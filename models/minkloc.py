import csv
from tkinter import W
from models.pointnet.PointNet import PointNetfeatv1
#################################################

import torch
from torch import nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from models.minkfpn import MinkFPN
from models.netvlad import MinkNetVladWrapper
import layers.pooling as layers_pooling

class PNT_GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, kernel=(4096,1)):
        super(PNT_GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.kernel = kernel

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size=self.kernel).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class MinkLoc(torch.nn.Module):
    def __init__(self, backbone, pooling, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size, num_points, combine_params, dataset_name='Usyd'):
        super().__init__()

        self.num_points = num_points
        self.with_pntnet = True if 'pointnet' in combine_params else False
        self.with_self_att = True if 'self_attention' in combine_params else False
        self.with_cross_att = True if 'pointnet_cross_attention' in combine_params else False
        self.planes = planes

        if self.with_pntnet or self.with_cross_att:
            self.pointnet = PointNetfeatv1(num_points=num_points,
                                           global_feat=True,
                                           feature_transform=True,
                                           max_pool=False,
                                           output_dim=feature_size if self.with_pntnet else planes[0])
            if self.with_pntnet:
                self.pntnet_pooling = PNT_GeM(kernel=(self.num_points,1))
        self.in_channels = in_channels
        self.feature_size = feature_size    # Size of local features produced by local feature extraction block
        self.output_dim = output_dim        # Dimensionality of the global descriptor
        if backbone == 'MinkFPN':
            self.backbone = MinkFPN(in_channels=in_channels,
                                    out_channels=self.feature_size,
                                    num_top_down=num_top_down,
                                    conv0_kernel_size=conv0_kernel_size,
                                    layers=layers, planes=planes,
                                    combine_params=combine_params,
                                    dataset_name=dataset_name)
        self.n_backbone_features = output_dim
        if pooling == 'Max':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = layers_pooling.MAC()
        elif pooling == 'GeM':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = layers_pooling.GeM()
        elif pooling == 'NetVlad':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, output_dim=self.output_dim,
                                              cluster_size=64, gating=False)
        elif pooling == 'NetVlad_CG':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, output_dim=self.output_dim,
                                              cluster_size=64, gating=True)

    def write_time_file(self, time_file, time):
        if time_file:
            all_num = True
            for element in time:
                if type(element) not in [int, float]:
                    all_num = False
            if all_num:
                time = [round(item, 5) for item in time]
                with open(time_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(time)
            else:
                print(f'skipped time: {time} ')


    def forward(self, batch, time_file=None):
        if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        feats = batch['features']
        feats = feats.to('cuda')
        coords = batch['coords']
        coords = coords.to('cuda')

        x = ME.SparseTensor(feats, coords)

        pointnet_time = 0
        if self.with_pntnet or self.with_cross_att:
            PNT_x = batch['pnt_coords']
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            PNT_feats = self.pointnet(PNT_x.unsqueeze(dim=1))
            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                pointnet_time = start.elapsed_time(end)

        if self.with_cross_att:
            PNT_x_list = [item for item in PNT_x]
            PNT_coords = ME.utils.batched_coordinates(PNT_x_list).to(PNT_x.device)
            assert type(self.backbone).__name__ == 'MinkFPN', 'backbone for cross attention should be MinkFPN'
            x, attention_time = self.backbone(x, PNT_coords, PNT_feats.squeeze(dim=-1).view(-1, self.planes[0]), time_file=time_file)
        else:
            x, attention_time = self.backbone(x, time_file)

        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.feature_size)
        x = self.pooling(x)
        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.output_dim)
        # x is (batch_size, output_dim) tensor

        if self.with_pntnet:
            y = self.pntnet_pooling(PNT_feats.view(-1, self.num_points, self.feature_size)).view(-1, self.feature_size)
            x = x + y

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
        else:
            total_time = 0
        time = [total_time, pointnet_time] + attention_time


        self.write_time_file(time_file, time)

        return x


    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print('Backbone parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print('Aggregation parameters: {}'.format(n_params))
        if hasattr(self.backbone, 'print_info'):
            self.backbone.print_info()
        if hasattr(self.pooling, 'print_info'):
            self.pooling.print_info()
