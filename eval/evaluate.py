# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

import csv
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import random
import tqdm
import open3d as o3d
np.random.seed(0)
random.seed(0)

import sys
sys.path.append(os.path.dirname(os.getcwd()))

from misc.utils import MinkLocParams
from models.model_factory import model_factory
from datasets.dataset_utils import to_spherical

DEBUG = False


def evaluate(model, device, params, log=False, time_file=None):
    # Run evaluation on all eval datasets

    if DEBUG:
        params.eval_database_files = params.eval_database_files[0:1]
        params.eval_query_files = params.eval_query_files[0:1]

    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, params, database_sets, query_sets, log=log, time_file=time_file)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, params, database_sets, query_sets, log=False, time_file=None):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    print(f"Extracting database sets embeddings")
    for set in tqdm.tqdm(database_sets):
        database_embeddings.append(get_latent_vectors(model, set, device, params, time_file=time_file))

    print(f"\nExtracting query sets embeddings")
    for set in tqdm.tqdm(query_sets):
        query_embeddings.append(get_latent_vectors(model, set, device, params, time_file=time_file))

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                                                database_sets, log=log)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity}
    return stats

def pnv_preprocessing(xyz):
    vox_sz = 0.3
    while np.asarray(xyz).shape[0] > 4096:
        xyz = downsample_point_cloud(xyz, vox_sz)
        vox_sz += 0.01
    return np.asarray(xyz)

def downsample_point_cloud(xyz, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    return pcd_ds.points

def load_pc(filename, params):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
    file_path = os.path.join(params.dataset_folder, filename)

    if params.dataset_name == "USyd":
        pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])
    elif params.dataset_name == "IntensityOxford":
        pc = np.fromfile(file_path, dtype=np.float64).reshape([-1, 4])
    elif params.dataset_name == "Oxford":
        pc = np.fromfile(file_path, dtype=np.float64).reshape([-1, 3])
    elif params.dataset_name == 'TUM':
        pc = np.fromfile(file_path, dtype=np.float64).reshape([-1, 3])
        assert pc.size == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)

    # remove intensity for models which are not using it
    if params.model_params.version in ['MinkLoc3D', 'MinkLoc3D-S']:
        pc = pc[:, :3]
    # limit distance
    pc = pc[np.linalg.norm(pc[:, :3], axis=1) < params.max_distance]
    if params.dataset_name == "USyd":
        pc = pnv_preprocessing(pc)
    # convert to spherical coordinates in -S versions
    if params.model_params.version in ['MinkLoc3D-S', 'MinkLoc3D-SI']:
        pc_s = to_spherical(pc, params.dataset_name)
    else:
        pc_s = pc

    # shuffle points in case they are randomly subsampled later
    # np.random.seed(0)
    # np.random.shuffle(pc)

    pcs = [pc, pc_s]
    for idx, pc in  enumerate(pcs):
        pc = torch.tensor(pc, dtype=torch.float)
        padlen = params.num_points - len(pc)
        if padlen > 0:
            pc = torch.nn.functional.pad(pc, (0, 0, 0, padlen), "constant", 0)
        elif padlen < 0:
            pc = pc[:params.num_points]
        pcs[idx] = pc

    # return pc, pc_s
    return pcs[0], pcs[1]

def get_latent_vectors(model, set, device, params, time_file=None):
    # Adapted from original PointNetVLAD code

    """
    if DEBUG:
        embeddings = torch.randn(len(set), 256)
        return embeddings
    """
    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    model.eval()
    embeddings_l = []

    include_pnt, pnt2s = False, False
    for key in ['pointnet', 'pointnet_cross_attention']:
        if key in params.model_params.combine_params:
            include_pnt = True
            if params.model_params.combine_params[key]['pnt2s']:
                pnt2s = True
    include_pnt2s = True if include_pnt and pnt2s else False

    for i, elem_ndx in enumerate(set):
        x, x_s = load_pc(set[elem_ndx]["query"], params)
        # import code; code.interact(local=locals())
        with torch.no_grad():
            # models without intensity
            if params.model_params.version in ['MinkLoc3D', 'MinkLoc3D-S']:
                coords = ME.utils.sparse_quantize(coordinates=x_s, quantization_size=params.model_params.mink_quantization_size)
                coords_more = ME.utils.sparse_quantize(coordinates=x_s, quantization_size=[0.0001, 0.0001, 0.0001])
                bcoords = ME.utils.batched_coordinates([coords]).to(device)
                boords_more =  ME.utils.batched_coordinates([coords_more]).to(device)

                # Assign a dummy feature equal to 1 to each point
                # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32).to(device)

            # models with intensity - intensity value is averaged over voxel
            elif params.model_params.version in ['MinkLoc3D-I', 'MinkLoc3D-SI']:
                sparse_field = ME.TensorField(features=x_s[:, 3].reshape([-1, 1]),
                                              coordinates=ME.utils.batched_coordinates(
                                                  [x_s[:, :3] / np.array(params.model_params.mink_quantization_size)],
                                                  dtype=torch.int),
                                              quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                              minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED).sparse()
                feats = sparse_field.features.to(device)
                bcoords = sparse_field.coordinates.to(device)

            batch = {'coords': bcoords, 'features': feats}
            batch['coords_more'] = boords_more if params.model_params.version == 'MinkLoc3D-S' else None

            if include_pnt:
                pnt_coords = x_s if include_pnt2s else x
                batch['pnt_coords'] = pnt_coords.unsqueeze(dim=0).to(device)

            # import code; code.interact(local=locals())
            embedding = model(batch, time_file=time_file)

            # embedding is (1, 256) tensor
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]  # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        if log:
            # Log 10% of false positives (returned as the first element) for Oxford dataset
            # Check if there's a false positive returned as the first element
            if query_details['query'][:6] == 'oxford' and indices[0][0] not in true_neighbors and random.random() < 0.1:
                fp_ndx = indices[0][0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0, 0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
                                        (query_details['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(indices[0])):
                    if indices[0][k] in true_neighbors:
                        closest_pos_ndx = indices[0][k]
                        tp = database_sets[m][
                            closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[0][k]
                        tp_world_dist = np.sqrt((query_details['northing'] - tp['northing']) ** 2 +
                                                (query_details['easting'] - tp['easting']) ** 2)
                        break

                with open("log_fp.txt", "a") as f:
                    s = "{}, {}, {:0.2f}, {:0.2f}".format(query_details['query'], fp['query'], fp_emb_dist,
                                                          fp_world_dist)
                    if tp is None:
                        s += ', 0, 0, 0\n'
                    else:
                        s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                    f.write(s)

            if query_details['query'][:6] == 'oxford' and len(indices[0]) >= 5 and random.random() < 0.01:
                # For randomly selected 1% of queries save details of 5 best matches for later visualization
                s = "{}, ".format(query_details['query'])
                for k in range(min(len(indices[0]), 5)):
                    is_match = indices[0][k] in true_neighbors
                    e_ndx = indices[0][k]
                    e = database_sets[m][e_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                    e_emb_dist = distances[0][k]
                    s += ', {}, {:0.2f}, {}, '.format(e['query'], e_emb_dist, 1 if is_match else 0)
                s += '\n'
                out_file_name = "log_search_results.txt"
                with open(out_file_name, "a") as f:
                    f.write(s)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100

    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on USyd/IntensityOxford (described in MinkLoc3D-SI) '
                                                 'or PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true')
    parser.set_defaults(log=False)

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('Debug mode: {}'.format(args.debug))
    print('Visualize: {}'.format(args.visualize))
    print('Log search results: {}'.format(args.log))
    print('')

    params = MinkLocParams(args.config, args.model_config)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    w_list = w.split('/')
    logdir = '/'.join(w_list[:-1])
    epoch = w_list[-1].split('.')[0]
    time_file = os.path.join(logdir, f'{epoch}_time.csv')

    with open(time_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Total', 'Pointnet', 'Self-Attention', 'Cross-Attention-Linear', 'Cross-Attention-Dot'])

    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    stats = evaluate(model, device, params, args.log, time_file=time_file)
    print_eval_stats(stats)
