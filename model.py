import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50
from torch.nn import functional as F
from modules import norm_layer, ChannelCompression, upsample, PredictLayer
from vit_seg_modeling import Transformer, CONFIGS

CHANNELS = [64, 256, 512, 1024]


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.config = config

        # Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')
        self.fuse_backbone = Transformer(config=CONFIGS['R50-ViT-B_16'], img_size=config.trainsize, vis=False)

        # fusion
        from modules import CrossAttentionFusionPool as FusionBlock

        from modules import Decoder as DecoderBlock

        fusions = []
        for i in range(4):
            fusions.append(FusionBlock(CHANNELS[i], config.dilation, config.kernel))
        self.fusion = nn.ModuleList(fusions)

        predicts, decoders = [], []
        for i in range(config.decoder_num):
            decoders.append(DecoderBlock(64))
            predicts.append(PredictLayer(64, config.trainsize))
        self.decoders = nn.ModuleList(decoders)
        self.predicts = nn.ModuleList(predicts)

        self.rgb_compress = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, bias=False),
            norm_layer(1024),
            nn.ReLU(inplace=True)
        )
        self.depth_compress = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, bias=False),
            norm_layer(1024),
            nn.ReLU(inplace=True)
        )
        self.compress1 = ChannelCompression(256, 64)
        self.compress2 = ChannelCompression(512, 64)  #2
        self.compress3 = ChannelCompression(768, 64)

        # Prediction
        self.high_predict = PredictLayer(64, config.trainsize)
        self.middle_predict = PredictLayer(64, config.trainsize)

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        # stage 0
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        x, x_depth, x_fused = self.fusion[0](x, x_depth)

        # stage 1
        x1 = self.resnet.layer1(x)
        x1_depth = self.resnet_depth.layer1(x_depth)
        x1_fused = self.fuse_backbone.embeddings.hybrid_model.body[0](x_fused)

        x1, x1_depth, x1_fused_add = self.fusion[1](x1, x1_depth)
        x1_fused = x1_fused_add + x1_fused

        # stage 2
        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)
        x2_fused = self.fuse_backbone.embeddings.hybrid_model.body[1](x1_fused)

        x2, x2_depth, x2_fused_add = self.fusion[2](x2, x2_depth)
        x2_fused = x2_fused_add + x2_fused

        # stage 3
        x3 = self.resnet.layer4(self.resnet.layer3(x2))
        x3_depth = self.resnet_depth.layer4(self.resnet_depth.layer3(x2_depth))
        x3_fused = self.fuse_backbone.embeddings.hybrid_model.body[2](x2_fused)

        x3 = self.rgb_compress(x3)
        x3_depth = self.depth_compress(x3_depth)
        x3, x3_depth, x3_fused_add = self.fusion[3](x3, x3_depth)
        x3_fused = x3_fused_add + x3_fused

        # delete intermediate features
        del x, x_depth, x_fused, x1, x1_depth, x2, x2_depth, x3, x3_depth
        del x1_fused_add, x2_fused_add, x3_fused_add

        # transformer part
        x3_fused = self.fuse_backbone.embeddings.patch_embeddings(x3_fused)
        x3_fused = x3_fused.flatten(2)
        x3_fused = x3_fused.transpose(-1, -2)
        x3_fused = x3_fused + self.fuse_backbone.embeddings.position_embeddings
        x3_fused = self.fuse_backbone.embeddings.dropout(x3_fused)

        x3_fused, _ = self.fuse_backbone.encoder(x3_fused)
        B, n_patch, hidden = x3_fused.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x3_fused = x3_fused.permute(0, 2, 1)
        x3_fused = x3_fused.contiguous().view(B, hidden, h, w)

        # channel compression
        x1_fused = self.compress1(x1_fused)
        x2_fused = self.compress2(x2_fused)
        x3_fused = self.compress3(x3_fused)

        # decoder
        decoder_output = []
        for i in range(self.config.decoder_num):
            x3_fused, x2_fused, x1_fused = self.decoders[i](x3_fused, x2_fused, x1_fused)
            decoder_output.append(self.predicts[i](x1_fused))

        # prediction
        middle_sal = self.middle_predict(x2_fused)
        high_sal = self.high_predict(x3_fused)
        return high_sal, middle_sal, decoder_output

    # initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v

        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v

        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)

        self.fuse_backbone.load_from(np.load('./models/imagenet21k_R50+ViT-B_16.npz'))


if __name__ == '__main__':
    model = Net().cuda()
    img = torch.randn((3, 3, 256, 256)).cuda()
    with torch.no_grad():
        res1 = model.resnet.maxpool(model.resnet.relu(model.resnet.conv1(img)))  # (3, 64, 64, 64)
        res2 = model.resnet.layer1(res1)  # (3, 256, 64, 64)
        res3 = model.resnet.layer2(res2)  # （3，512，32，32）
        res4 = model.resnet.layer3_1(res3)  # （3，1024，16，16）
        res5 = model.resnet.layer4_1(res4)  # （3，2048，8，8）

        features = [res1, res2, res3, res4, res5]
        for i in range(5):
            print(features[i].shape)
