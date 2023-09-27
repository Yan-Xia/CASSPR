import os
import configparser
import time
from typing import Dict
import numpy as np


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MINKLOC3D-SI']

        self.model_params_path = model_params_path
        self.gpu = params.getint('gpu')
        self.backbone = params.get('backbone')
        assert self.backbone in ['MinkFPN'], 'Supported Backbone are: MinkFPN'
        self.pooling = params.get('pooling')
        assert self.pooling in ['Max', 'GeM', 'NetVlad', 'NetVlad_CG'], 'Supported Pooling are: Max, GeM, NetVlad, NetVlad_CG'
        self.output_dim = params.getint('output_dim', 256)  # Size of the final descriptor

        # Add gating as the last step
        if 'vlad' in self.pooling.lower():
            self.cluster_size = params.getint('cluster_size', 64)  # Size of NetVLAD cluster
            self.gating = params.getboolean('gating', True)  # Use gating after the NetVlad

        self.mink_quantization_size = [float(item) for item in params['mink_quantization_size'].split(',')]
        self.version = params['version']
        assert self.version in ['MinkLoc3D', 'MinkLoc3D-I', 'MinkLoc3D-S', 'MinkLoc3D-SI'], 'Supported versions ' \
                                                                                            'are: MinkLoc3D, ' \
                                                                                            'MinkLoc3D-I, ' \
                                                                                            'MinkLoc3D-S, ' \
                                                                                            'MinkLoc3D-SI '

        self.feature_size = params.getint('feature_size', 256)
        if 'planes' in params:
            self.planes = [int(e) for e in params['planes'].split(',')]
        else:
            self.planes = [32, 64, 64]

        if 'layers' in params:
            self.layers = [int(e) for e in params['layers'].split(',')]
        else:
            self.layers = [1, 1, 1]

        self.num_top_down = params.getint('num_top_down', 1)
        self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)

        combine_modules = ['POINTNET', 'SELF-ATTENTION', 'POINTNET-CROSS-ATTENTION'] \
                          if self.backbone == 'MinkFPN' else 'POINTNET'
        combine_modules = {} if self.version not in ['MinkLoc3D-S', 'MinkLoc3D-SI', 'MinkLoc3D'] else combine_modules
        self.get_combine_params(config, combine_modules)
        assert isinstance(self.combine_params, Dict)

    def get_combine_params(self, config, combine_modules):
        self.combine_params = {}
        pntnet_params = config['POINTNET'] if 'POINTNET' in combine_modules else None
        self_att_params = config['SELF-ATTENTION'] if 'SELF-ATTENTION' in combine_modules else None
        cross_att_params = config['POINTNET-CROSS-ATTENTION'] if 'POINTNET-CROSS-ATTENTION' in combine_modules else None

        with_pntnet = pntnet_params.getboolean('with_pntnet') if self_att_params is not None else None
        with_cross_att = cross_att_params.getboolean('with_cross_att') if cross_att_params is not None else None
        with_self_att = self_att_params.getboolean('with_self_att') if self_att_params is not None else None
        assert not(with_pntnet and with_cross_att), 'Options: with_pntnet or with_cross_att or Neither '

        if with_pntnet:
            pntnet_combine_params = {'pointnet': { 'pnt2s': pntnet_params.getboolean('pnt2s') }}
            self.combine_params = {**self.combine_params, **pntnet_combine_params}

        if with_self_att:
            assert self_att_params.getint('num_layers') >= 1, 'num_layers must be greater than 1'
            assert self_att_params.getint('num_layers') <= sum(self.layers)+1+self.num_top_down+1, 'num_layers should be <= sum(self.layers)+2+self.num_top_down'
            self_att_combine_params = {'self_attention': {  'linear_att': self_att_params.getboolean('linear_att'),
                                                            'num_layers': self_att_params.getint('num_layers'),
                                                            'kernel_size': self_att_params.getint('kernel_size'),
                                                            'stride': self_att_params.getint('stride'),
                                                            'dilation': self_att_params.getint('dilation'),
                                                            'num_heads': self_att_params.getint('num_heads') }}
            self.combine_params = {**self.combine_params, **self_att_combine_params}

        if with_cross_att:
            assert cross_att_params['attention_type'] in ['dot_prod', 'linear_attention'], 'Supported attention types: dot_prod, linear_attention'
            cross_att_combine_params = {"pointnet_cross_attention":
                                                           {"pnt2s": cross_att_params.getboolean('pnt2s'),
                                                            "nhead": cross_att_params.getint('num_heads'),
                                                            "d_feedforward": cross_att_params.getint("d_feedforward"),
                                                            "dropout": cross_att_params.getint("dropout"),
                                                            "transformer_act": cross_att_params['transformer_act'],
                                                            "pre_norm": cross_att_params.getboolean("pre_norm"),
                                                            "attention_type": cross_att_params['attention_type'],
                                                            "sa_val_has_pos_emb": cross_att_params.getboolean('sa_val_has_pos_emb'),
                                                            "ca_val_has_pos_emb": cross_att_params.getboolean('ca_val_has_pos_emb'),
                                                            "num_encoder_layers": cross_att_params.getint('num_encoder_layers'),
                                                            "transformer_encoder_has_pos_emb": cross_att_params.getboolean('transformer_encoder_has_pos_emb') }}
            self.combine_params = {**self.combine_params, **cross_att_combine_params}

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


def xyz_from_depth(depth_image, depth_intrinsic, depth_scale=1000.):
    # Return X, Y, Z coordinates from a depth map.
    # This mimics OpenCV cv2.rgbd.depthTo3d() function
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    # Construct (y, x) array with pixel coordinates
    y, x = np.meshgrid(range(depth_image.shape[0]), range(depth_image.shape[1]), sparse=False, indexing='ij')

    X = (x - cx) * depth_image / (fx * depth_scale)
    Y = (y - cy) * depth_image / (fy * depth_scale)
    xyz = np.stack([X, Y, depth_image / depth_scale], axis=2)
    xyz[depth_image == 0] = np.nan
    return xyz


class MinkLocParams:
    """
    Params for training MinkLoc models on Oxford dataset
    """

    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(
            model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.num_points = params.getint('num_points')
        self.max_distance = params.getint('max_distance')

        self.dataset_name = params.get('dataset_name')
        assert self.dataset_name in ['USyd', 'IntensityOxford', 'Oxford', 'TUM'], 'Dataset should be USyd, IntensityOxford ' \
                                                                           'or Oxford, TUM'

        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 128)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = params.getfloat('lr', 1e-3)

        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.normalize_embeddings = params.getboolean('normalize_embeddings',
                                                      True)  # Normalize embeddings during training and evaluation
        self.loss = params.get('loss')

        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)  # Margin used in loss function
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)  # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)

        if self.dataset_name == 'USyd':
            self.eval_database_files = ['usyd_evaluation_database.pickle']
            self.eval_query_files = ['usyd_evaluation_query.pickle']

        elif self.dataset_name == 'Oxford':
            self.eval_database_files = ['oxford_evaluation_database.pickle', 'business_evaluation_database.pickle',
                                        'residential_evaluation_database.pickle',
                                        'university_evaluation_database.pickle']
            self.eval_query_files = ['oxford_evaluation_query.pickle', 'business_evaluation_query.pickle',
                                     'residential_evaluation_query.pickle', 'university_evaluation_query.pickle']
        elif self.dataset_name == 'TUM':
            self.eval_database_files = ['tum_evaluation_frame_5m_database.pickle']
            self.eval_query_files = ['tum_evaluation_frame_5m_query.pickle']

        assert len(self.eval_database_files) == len(self.eval_query_files)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')
