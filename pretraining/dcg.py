import torch
import torch.nn as nn
import numpy as np
import einops
import pretraining.tools as tools
import pretraining.modules as m
from pretraining.bra import BiLevelRoutingAttention
from pretraining.fem import BasicBlock
# from pretraining.mmpose.models.backbones.Hybrid_Transformer_CNN import BasicBlock

class DCG(nn.Module):
    def __init__(self, parameters):
        super(DCG, self).__init__()
        self.dn_resnet = m.DownsampleNetworkResNet18V1()

        # save parameters
        self.experiment_parameters = {
        "device_type": 'gpu',
        "gpu_number": 0,
        # model related hyper-parameters
        "cam_size": (7, 7),
        "K": 6,
        "crop_shape": (32, 32),
        "post_processing_dim":512,
        "num_classes":parameters.data.num_classes,
        "use_v1_global":True,
        "percent_t": 1.0,
        }
        self.cam_size = self.experiment_parameters["cam_size"]

        # construct networks
        # global network
        self.global_network = m.GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters, self)

        # detection module
        self.retrieve_roi_crops = m.RetrieveROIModule(self.experiment_parameters, self)


        self.bra = BiLevelRoutingAttention(dim=1024, n_win=2, topk=4)
        self.basicblock = BasicBlock(1024,1024)

        self.local_proj = nn.Linear(1024, 7)  # Project y_local to match y_global
        # detection network
        self.local_network = m.LocalNetwork(self.experiment_parameters, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.experiment_parameters, self)
        self.attention_module.add_layers()
        # fusion branch
        # self.fusion_dnn = nn.Linear(self.experiment_parameters["post_processing_dim"]+512, self.experiment_parameters["num_classes"], bias=False)


    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """
        Function that takes in the original image and cropping position and returns the crops
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        if self.experiment_parameters["device_type"] == "gpu":
            device = torch.device("cuda:{}".format(self.experiment_parameters["gpu_number"]))
            output = output.to(device)
        for i in range(batch_size):
            for j in range(num_crops):
                tools.crop_pytorch(x_original_pytorch[i, 0, :, :],
                                                    self.experiment_parameters["crop_shape"],
                                                    crop_positions[i,j,:],
                                                    output[i,j,:,:],
                                                    method=crop_method)
        return output


    def forward(self, x_original):
        """
        :param x_original: N,H,W,C numpy matrix
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x_original)

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # region proposal network
        small_x_locations = self.retrieve_roi_crops.forward(x_original, self.cam_size, self.saliency_map)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        # patch retriever
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        self.patches = crops_variable.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1) # shape of crops_variable: torch.Size([384, 1, 32, 32]) n c h w
        # print("shape of crops_variable:",crops_variable.shape)

        # ----------------------------------------------------------------
        
        # 1. 用ResNet提特征
        features = self.dn_resnet(crops_variable.expand(-1, 3, -1, -1))  # [N, 1024, 4, 4]
        features_nhwc = features.permute(0, 2, 3, 1)  # [N, 4, 4, 1024]

        # crops_variable = crops_variable.expand(-1, 3, -1 , -1)
        # crops_variable = crops_variable.permute(0, 2, 3, 1) # shape of crops_variable: torch.Size([384, 32, 32, 3]) n h w c
        # 2. BRA
        # print("features_nhwc.shape:", features_nhwc.shape)
        # print("n_win:", self.bra.n_win)
        h_crops = self.bra.forward(features_nhwc) # nhwc -> nhwc n 4 4 1024
        h_crops = einops.rearrange(h_crops, 'n h w c -> n c h w') # nhwc -> nchw  n 1024 4 4
        #3. FEM
        h_crops = self.basicblock.forward(h_crops) #nchw -> nchw
        #4. global average pooling
        h_crops = h_crops.mean(dim=2).mean(dim=2)
        h_crops = h_crops.view(batch_size, num_crops, -1)

        # self.y_local = h_crops
        # h_crops = self.bra(crops_variable)
        
        # h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1) # shape of h_crops: torch.Size([64, 6, 2048])
        # print("shape of h_crops:",h_crops.shape)
        # print("y_global shape:", self.y_global.shape)
        # print("h_crops shape:", h_crops.shape)
        # # MIL module
        # # y_local is not directly used during inference
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)
        # self.y_global = self.local_proj(self.y_global)
        # print("y_local shape:", self.y_local.shape)
        self.y_fusion = 0.5* (self.y_global+self.y_local)
        return self.y_fusion, self.y_global, self.y_local