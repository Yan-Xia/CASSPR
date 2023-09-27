# Author: Jacek Komorowski
# Warsaw University of Technology

import models.minkloc as minkloc


def model_factory(params):
    in_channels = 1

    model = minkloc.MinkLoc(backbone=params.model_params.backbone,
                            pooling=params.model_params.pooling,
                            in_channels=in_channels,
                            feature_size=params.model_params.feature_size,
                            output_dim=params.model_params.output_dim,
                            planes=params.model_params.planes,
                            layers=params.model_params.layers,
                            num_top_down=params.model_params.num_top_down,
                            conv0_kernel_size=params.model_params.conv0_kernel_size,
                            num_points=params.num_points,
                            combine_params=params.model_params.combine_params,
                            dataset_name=params.dataset_name,)

    return model
