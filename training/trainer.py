import os
from datetime import datetime
import numpy as np
import torch
from torch import nn
import pickle
import tqdm
import pathlib
import json

from torch.utils.tensorboard import SummaryWriter

from eval.evaluate import evaluate, print_eval_stats
from misc.utils import MinkLocParams, get_datetime
from models.loss import make_loss
from models.model_factory import model_factory


VERBOSE = False


def print_stats(stats, phase):
    if 'num_pairs' in stats:
        # For batch hard contrastive loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Pairs per batch (all/non-zero pos/non-zero neg): {:.1f}/{:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pairs'],
                       stats['pos_pairs_above_threshold'], stats['neg_pairs_above_threshold']))
    elif 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train(dataloaders, params: MinkLocParams, ckpt=None, debug=False, visualize=False):
    # Create model class
    s = get_datetime()
    now = datetime.now()
    now_strftime = now.strftime("%Y%m%d-%H%M%S")
    model = model_factory(params)
    start_epoch = 1
    if ckpt is not None:
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint)
        print("Loaded model from ", ckpt)
        start_epoch = int(ckpt.split('/')[-1].split('.')[0].split('epoch')[-1]) + 1
        print("Starting from epoch ", start_epoch)

    model_name = 'model_' + params.model_params.backbone + params.model_params.pooling + \
                 '_' + now_strftime.replace('-', '_')
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)
    pathlib.Path(model_pathname).mkdir(parents=False, exist_ok=True)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():

        device = torch.device(f"cuda:{params.model_params.gpu}")
        torch.cuda.set_device(device)
        model.to(device)

        ######### ddp

    else:
        device = "cpu"

    print('Model device: {}'.format(device))

    loss_fn = make_loss(params)

    # Training elements
    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.5)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

    # now = datetime.now()
    logdir = os.path.join("../tf_logs", now_strftime)
    writer = SummaryWriter(logdir)

    ###########################################################################
    #
    ###########################################################################

    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}

    for epoch in tqdm.tqdm(range(start_epoch, params.epochs + 1)):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch

            count_batches = 0
            for batch, positives_mask, negatives_mask in dataloaders[phase]:
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}

                if debug and count_batches > 2:
                    break

                # Move everything to the device except 'coords' which must stay on CPU
                batch = {e: batch[e].to(device) if e != 'coords' else batch[e] for e in batch}

                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()
                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue

                optimizer.zero_grad()
                if visualize:
                    #visualize_batch(batch)
                    pass

                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    torch.autograd.set_detect_anomaly(True)
                    embeddings = model(batch)
                    loss, temp_stats, _ = loss_fn(embeddings, positives_mask, negatives_mask)

                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)
                    batch_stats['loss'] = loss.item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_stats.append(batch_stats)
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

            # ******* PHASE END *******
            # Compute mean stats for the epoch
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # ******* EPOCH END *******

        if scheduler is not None:
            scheduler.step()

        loss_metrics = {'train': stats['train'][-1]['loss']}
        if 'val' in phases:
            loss_metrics['val'] = stats['val'][-1]['loss']
        writer.add_scalars('Loss', loss_metrics, epoch)

        if 'num_triplets' in stats['train'][-1]:
            nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}
            if 'val' in phases:
                nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
            writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

        elif 'num_pairs' in stats['train'][-1]:
            nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
                          'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
            if 'val' in phases:
                nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
                nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
            writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

        if params.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

        print('')

        if params.dataset_name != 'TUM' and epoch % 10 != 0 and epoch > 0 and epoch < params.epochs:
            continue

        # Save model weights
        final_model_path = os.path.join(model_pathname, f'epoch{epoch}.pth')
        torch.save(model.state_dict(), final_model_path)

        # Evaluate the final model
        model.eval()
        with torch.no_grad():
            final_eval_stats = evaluate(model, device, params)
        print(f'\nEpoch{epoch} model:')
        print_eval_stats(final_eval_stats)
        stats['eval'].append({f'epoch{epoch}': final_eval_stats})
        print('')
        for database_name in final_eval_stats:
            nz_metric1 = {f'{database_name}': final_eval_stats[database_name]['ave_one_percent_recall'].item()}
            nz_metric2 = {f'{database_name}': final_eval_stats[database_name]['ave_recall'][0].item()}
            writer.add_scalars('Eval Mean recall', nz_metric2, epoch)
            writer.add_scalars('Eval One percent recall', nz_metric1, epoch)
            writer.flush()

    stats = {'stats': stats, 'params': vars(params)}
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(stats, f, indent = 6, default=lambda o: getattr(o, '__dict__', str(o)))


def export_eval_stats(file_name, prefix, eval_stats, dataset_name):
    s = '\n' + prefix + '\n\n'
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        if dataset_name == 'USyd':
            ave_1p_recall = eval_stats['usyd']['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats['usyd']['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        elif dataset_name == 'IntensityOxford':
            ave_1p_recall = eval_stats['intensityOxford']['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats['intensityOxford']['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)
        elif dataset_name == 'TUM':
            ave_1p_recall = eval_stats['tum']['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats['tum']['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += 'Average 1% recall: {:0.2f}\n'.format(ave_1p_recall)
            s += 'Average Recall: {:0.2f}\n'.format(ave_recall)
        else:
            for ds in ['oxford', 'university', 'residential', 'business']:
                ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
                ave_1p_recall_l.append(ave_1p_recall)
                ave_recall = eval_stats[ds]['ave_recall'][0]
                ave_recall_l.append(ave_recall)
                s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += 'Mean 1% Recall @N: {:0.2f}\nMean Recall {:0.2f}\n\n '.format(mean_1p_recall, mean_recall)
        f.write(s)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
