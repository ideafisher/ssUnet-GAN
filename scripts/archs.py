import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from normalization import SPADE
from xresidualblock import xResidualBlock
from torch.nn import init
__all__ = ['UNet', 'NestedUNet', 'SSUNet', 'UNet_ori', 'UNet_B_SS','AttUNet', 'UNet_R_SS', 'UNet_R_SS_v2']

import os, sys
from torch import nn
sys.path.append('./image-segmentation/')

from efficientnet_pytorch import EfficientNet
import torchvision.models as models


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        semantic_nc = opt['num_class']
        opt_type = opt['opt_type']
        semantic_nc = 2
        self.learned_shortcut = (fin != fout)
        if opt_type == 1:
            fmiddle = min(fin, fout)
        else:
            fmiddle = max(fin, fout)
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        #if 'spectral' in opt.norm_G:
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        #    if self.learned_shortcut:
        #        self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        #spade_config_str = opt.norm_G.replace('spectral', '')
        spade_config_str = 'spadesyncbatch3x3'
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        #self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        #self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return torch.nn.functional.leaky_relu(x, 2e-1)



class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class xBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(xBasicBlock, self).__init__()

        self.conv1 = xResidualBlock(in_planes, planes, kernel_size=3, s=stride)
        self.conv2 = xResidualBlock(planes, planes, kernel_size=3, s=1)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                #nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                xResidualBlock(in_planes, self.expansion * planes, kernel_size=1, s=stride),
            )

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            )

        #self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv2.weight)
        init.xavier_normal_(self.shortcut[0].weight)


    def forward(self, x):
        if 1: # original RB(Resnet Block)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        else: #simplified RB(Resnet Block)
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out += self.shortcut(x)
            out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttUNet(nn.Module):
    #def __init__(self, img_ch=3, output_ch=1):
    def __init__(self, output_ch, img_ch=3, deep_supervision=False, **kwargs):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)


        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class UNet_B_SS(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        #nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [ 64, 128, 256, 512, 1024]
        # nb_filter = [64+28, 128+28, 256+28, 512+28, 1024+28]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        context = 'spadebatch3x3'  # 'spadeinstance3x3'
        ss_scale = 16
        spade_mid = num_classes
        self.SPADE0_0 = SPADE(context, nb_filter[0], spade_mid, nb_filter[0] / ss_scale)
        self.SPADE1_0 = SPADE(context, nb_filter[1], spade_mid, nb_filter[1] / ss_scale)
        self.SPADE2_0 = SPADE(context, nb_filter[2], spade_mid, nb_filter[2] / ss_scale)
        self.SPADE3_0 = SPADE(context, nb_filter[3], spade_mid, nb_filter[3] / ss_scale)


        self.SPADE4_0 = SPADE(context, nb_filter[4], spade_mid, nb_filter[4] / ss_scale)
        self.SPADE3_1 = SPADE(context, nb_filter[3], spade_mid, nb_filter[3] / ss_scale)
        self.SPADE2_2 = SPADE(context, nb_filter[2], spade_mid, nb_filter[2] / ss_scale)
        self.SPADE1_3 = SPADE(context, nb_filter[1], spade_mid, nb_filter[1] / ss_scale)
        self.SPADE0_4 = SPADE(context, nb_filter[0], spade_mid, nb_filter[0] / ss_scale)

        self.conv0_0 = Bottleneck(input_channels, nb_filter[0])
        self.conv1_0 = Bottleneck(nb_filter[0], nb_filter[1])
        self.conv2_0 = Bottleneck(nb_filter[1], nb_filter[2])
        self.conv3_0 = Bottleneck(nb_filter[2], nb_filter[3])
        self.conv4_0 = Bottleneck(nb_filter[3], nb_filter[4])


        self.conv3_1 = Bottleneck(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = Bottleneck(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = Bottleneck(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = Bottleneck(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_0 = self.SPADE0_0(x0_0, x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.SPADE1_0(x1_0, x1_0)
        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.SPADE2_0(x2_0, x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.SPADE3_0(x3_0, x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0))

        x4_0 = self.SPADE4_0(x4_0, x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.SPADE3_1(x3_1, x3_1)
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x2_2 = self.SPADE2_2(x2_2, x2_2)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x1_3 = self.SPADE1_3(x1_3, x1_3)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        x0_4 = self.SPADE0_4(x0_4, x0_4)
        output = self.final(x0_4)
        return output


class AttentiveCNN(nn.Module):
    def __init__(self,  model_info):
        super(AttentiveCNN, self).__init__()

        self.f_channel = 1408
        eff_net_flag = model_info['eff_flag']
        is_train = model_info['phase_train']
        if eff_net_flag == True:
            model_name = model_info['eff_model_name']
            # model_name= 'efficientnet-b2'
            pretrained_base = '../pretrained/normal/'
            image_size = EfficientNet.get_image_size(model_name)

            print('==> Building model.. : ', model_name)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if is_train == True:
                model = EfficientNet.from_pretrained(model_name, pretrained_base)
            else:
                model = EfficientNet.from_name(model_name)

            if model_name == 'efficientnet-b2':
                self.f_channel= 1408
            if model_name == 'efficientnet-b3':
                self.f_channel= 1536
            if model_name == 'efficientnet-b4':
                self.f_channel= 1792
            if model_name == 'efficientnet-b5':
                self.f_channel= 2048
            eff_conv = model.to(device)
            num_ftrs = model._fc.in_features
            self.eff_conv = eff_conv
            self.input_img_size = image_size
            self.eff_channel = 1024
            self.conv_a = nn.Conv2d(self.f_channel, self.eff_channel, kernel_size=1, bias=False)
        else:
            # ResNet-152 backend
            # resnet = models.resnet152( pretrained=True )
            resnet = models.resnet101(pretrained=True)
            modules = list(resnet.children())[:-2]  # delete the last fc layer and avg pool.
            resnet_conv = nn.Sequential(*modules)  # last conv feature
            self.resnet_conv = resnet_conv
            self.input_img_size = 224

        self.eff_net_flag = eff_net_flag

    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        resized_images = F.interpolate(images, size=(self.input_img_size, self.input_img_size), mode='bilinear')
        # Last conv layer feature map
        if self.eff_net_flag == True:
            A = self.eff_conv.extract_features(resized_images)
        else:
            A = self.resnet_conv( resized_images )
        A_out = self.conv_a(A)
        return A_out

six_step = True
class UNet_R_SS(nn.Module):
    def __init__(self, num_classes, input_channels=3,  deep_supervision=False, **kwargs):
        super().__init__()
        #nb_filter = [128, 256, 512, 768, 1024]
        six_step = True
        self.six_step = six_step
        if self.six_step == False:
            nb_filter = [ 64, 128, 256, 512, 1024]
        else:
            nb_filter = [64, 128, 256, 384, 512, 768]

        spade_mid = num_classes
        self.pool = nn.MaxPool2d(2, 2, return_indices=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        context = 'spadebatch3x3' #'spadeinstance3x3'
        ss_scale = 16
        self.conv0_0 = BasicBlock(input_channels,nb_filter[0])
        self.SPADE0_0 = SPADE(context, nb_filter[0], spade_mid,  nb_filter[0] / ss_scale)

        self.conv1_0 = BasicBlock(nb_filter[0], nb_filter[1])
        self.SPADE1_0 = SPADE(context, nb_filter[1], spade_mid, nb_filter[1] / ss_scale)

        self.conv2_0 = BasicBlock(nb_filter[1], nb_filter[2])
        self.SPADE2_0 = SPADE(context, nb_filter[2], spade_mid, nb_filter[2] / ss_scale)

        self.conv3_0 = BasicBlock(nb_filter[2], nb_filter[3])
        self.SPADE3_0 = SPADE(context, nb_filter[3], spade_mid, nb_filter[3] / ss_scale)

        self.conv4_0 = BasicBlock(nb_filter[3],  nb_filter[4])
        self.SPADE4_0 = SPADE(context, nb_filter[4], spade_mid, nb_filter[4] / ss_scale)
        if six_step == True:
            self.conv5_0 = BasicBlock(nb_filter[4],  nb_filter[5])
            self.SPADE5_0 = SPADE(context, nb_filter[5], spade_mid, nb_filter[5] / ss_scale)
            self.conv4_1 = BasicBlock(nb_filter[4] + nb_filter[5], nb_filter[4])
            self.SPADE4_1 = SPADE(context, nb_filter[4], spade_mid, nb_filter[4] / ss_scale)

        self.conv3_1 = BasicBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.SPADE3_1 = SPADE(context, nb_filter[3], spade_mid, nb_filter[3] / ss_scale)

        self.conv2_2 = BasicBlock(nb_filter[2] + nb_filter[3],nb_filter[2])
        self.SPADE2_2 = SPADE(context, nb_filter[2], spade_mid, nb_filter[2] / ss_scale)

        self.conv1_3 = BasicBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.SPADE1_3 = SPADE(context, nb_filter[1], spade_mid, nb_filter[1] / ss_scale)
        self.sp_up1_3 = SubPixelConvolutionalBlock(3, nb_filter[1], 2)

        self.conv0_4 = BasicBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.SPADE0_4 = SPADE(context, nb_filter[0], spade_mid, nb_filter[0] / ss_scale)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.final.weight, mode='fan_in')
        self.final.bias.data.fill_(0)


    def forward(self, input):

        x0_0 = self.conv0_0(input)
        x0_0 = self.SPADE0_0(x0_0, x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.SPADE1_0(x1_0, x1_0)
        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.SPADE2_0(x2_0, x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.SPADE3_0(x3_0, x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.SPADE4_0(x4_0, x4_0)
        if self.six_step == True:
            x5_0 = self.conv5_0(self.pool(x4_0))
            x5_0 = self.SPADE5_0(x5_0, x5_0)
            x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
            x4_1 = self.SPADE4_1(x4_1, x4_1)
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_1)], 1))
        else:
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.SPADE3_1(x3_1, x3_1)
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x2_2 = self.SPADE2_2(x2_2, x2_2)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x1_3 = self.SPADE1_3(x1_3, x1_3)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        x0_4 = self.SPADE0_4(x0_4, x0_4)

        output = self.final(x0_4)
        return output


class UNet_R_SS_v2(nn.Module):
    def __init__(self, num_classes, input_channels=3,  deep_supervision=False, **kwargs):
        super().__init__()
        #nb_filter = [128, 256, 512, 768, 1024]
        six_step = True
        self.six_step = six_step
        if self.six_step == False:
            nb_filter = [ 64, 128, 256, 512, 1024]
        else:
            nb_filter = [64, 128, 256, 384, 512, 768]

        spade_mid = num_classes
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        context = 'spadebatch3x3' #'spadeinstance3x3'
        ss_scale = 16
        self.conv0_0 = BasicBlock(input_channels,nb_filter[0])
        self.SPADE0_0 = SPADE(context, nb_filter[0], spade_mid,  nb_filter[0] / ss_scale)

        self.conv1_0 = BasicBlock(nb_filter[0], nb_filter[1])
        self.SPADE1_0 = SPADE(context, nb_filter[1], spade_mid, nb_filter[1] / ss_scale)

        self.conv2_0 = BasicBlock(nb_filter[1], nb_filter[2])
        self.SPADE2_0 = SPADE(context, nb_filter[2], spade_mid, nb_filter[2] / ss_scale)

        self.conv3_0 = BasicBlock(nb_filter[2], nb_filter[3])
        self.SPADE3_0 = SPADE(context, nb_filter[3], spade_mid, nb_filter[3] / ss_scale)

        self.conv4_0 = BasicBlock(nb_filter[3],  nb_filter[4])
        self.SPADE4_0 = SPADE(context, nb_filter[4], spade_mid, nb_filter[4] / ss_scale)

        self.conv5_0 = BasicBlock(nb_filter[4],  nb_filter[5])
        self.SPADE5_0 = SPADE(context, nb_filter[5], spade_mid, nb_filter[5] / ss_scale)
        self.conv_head5_0 = nn.Conv2d(nb_filter[5], nb_filter[4], kernel_size=1, stride=1, bias=False)

        self.conv4_1 = BasicBlock(nb_filter[4] + nb_filter[4], nb_filter[4])
        self.SPADE4_1 = SPADE(context, nb_filter[4], spade_mid, nb_filter[4] / ss_scale)
        self.conv_head4_1 = nn.Conv2d(nb_filter[4], nb_filter[3], kernel_size=1, stride=1, bias=False)

        self.conv3_1 = BasicBlock(nb_filter[3] + nb_filter[3], nb_filter[3])
        self.SPADE3_1 = SPADE(context, nb_filter[3], spade_mid, nb_filter[3] / ss_scale)
        self.conv_head3_1 = nn.Conv2d(nb_filter[3], nb_filter[2], kernel_size=1, stride=1, bias=False)

        self.conv2_1 = BasicBlock(nb_filter[2] + nb_filter[2],nb_filter[2])
        self.SPADE2_1 = SPADE(context, nb_filter[2], spade_mid, nb_filter[2] / ss_scale)
        #self.conv_head2_1 = nn.Conv2d(nb_filter[2], nb_filter[1], kernel_size=1, stride=1, bias=False)

        self.conv1_1 = BasicBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.SPADE1_1 = SPADE(context, nb_filter[1], spade_mid, nb_filter[1] / ss_scale)
        #self.conv_head1_1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1, bias=False)


        self.conv0_1 = BasicBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.SPADE0_1 = SPADE(context, nb_filter[0], spade_mid, nb_filter[0] / ss_scale)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.final.weight, mode='fan_in')
        self.final.bias.data.fill_(0)

    def forward(self, input):

        enc_0 = self.conv0_0(input)
        enc_0 = self.SPADE0_0(enc_0, enc_0)

        output0_0, indices0_0 = self.pool(enc_0)
        enc_1 = self.conv1_0(output0_0)
        enc_1 = self.SPADE1_0(enc_1, enc_1)

        output1_0, indices1_0 = self.pool(enc_1)
        enc_2_0 = self.conv2_0(output1_0)
        enc_2_0 = self.SPADE2_0(enc_2_0, enc_2_0)

        output2_0, indices2_0 = self.pool(enc_2_0)
        enc_3 = self.conv3_0(output2_0)
        enc_3 = self.SPADE3_0(enc_3, enc_3)

        output3_0, indices3_0 = self.pool(enc_3)
        enc_4 = self.conv4_0(output3_0)
        enc_4 = self.SPADE4_0(enc_4, enc_4)
        output4_0, indices4_0 = self.pool(enc_4)

        enc_5 = self.conv5_0(output4_0)
        enc_5 = self.SPADE5_0(enc_5, enc_5)
        enc_5 = self.conv_head5_0(enc_5)
        enc_5_up = self.unpool(enc_5, indices4_0)

        # decode + cat
        dec_4 = self.conv4_1(torch.cat([enc_4, enc_5_up], 1))
        dec_4 = self.SPADE4_1(dec_4, dec_4)
        dec_4 = self.conv_head4_1(dec_4)
        dec_4_up = self.unpool(dec_4, indices3_0)

        dec_3 = self.conv3_1(torch.cat([enc_3, dec_4_up], 1))
        dec_3 = self.SPADE3_1(dec_3, dec_3)
        dec_3 = self.conv_head3_1(dec_3)
        dec_3_up = self.unpool(dec_3, indices2_0)

        dec_2 = self.conv2_1(torch.cat([enc_2_0, dec_3_up], 1))
        dec_2 = self.SPADE2_1(dec_2, dec_2)

        dec_1 = self.conv1_1(torch.cat([enc_1, self.up(dec_2)], 1))
        dec_1 = self.SPADE1_1(dec_1, dec_1)

        dec_0 = self.conv0_1(torch.cat([enc_0, self.up(dec_1)], 1))
        dec_0 = self.SPADE0_1(dec_0, dec_0)

        output = self.final(dec_0)
        return output

class SSUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3,  deep_supervision=False, **kwargs):
        super().__init__()


        #nb_filter = [32+add_c, 64+add_c*2, 128+add_c*2, 256+add_c*2, 512+add_c*2]
        #nb_filter = [ 64, 128, 256, 512, 1024]
        nb_filter = [32, 64, 128, 256, 512] #512
        #nb_filter = [16, 32, 64, 128, 256] # 256
        #nb_filter = [8, 16, 32, 64, 128]
        opt = {}

        spade_mid = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        context = 'spadebatch3x3' #'spadeinstance3x3'
        ss_scale = 4
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.SPADE0_0 = SPADE(context, nb_filter[0], spade_mid, nb_filter[0] / ss_scale)

        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.SPADE1_0 = SPADE(context, nb_filter[1], spade_mid,  nb_filter[1] / ss_scale)

        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.SPADE2_0 = SPADE(context, nb_filter[2], spade_mid, nb_filter[2] / ss_scale)

        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.SPADE3_0 = SPADE(context, nb_filter[3], spade_mid, nb_filter[3] / ss_scale)

        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.SPADE4_0 = SPADE(context, nb_filter[4], spade_mid, nb_filter[4] / ss_scale)

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.SPADE3_1 = SPADE(context, nb_filter[3], spade_mid,  nb_filter[3] / ss_scale)

        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.SPADE2_2 = SPADE(context, nb_filter[2], spade_mid, nb_filter[2] / ss_scale)

        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.SPADE1_3 = SPADE(context, nb_filter[1], spade_mid,  nb_filter[1] / ss_scale)

        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.SPADE0_4 = SPADE(context, nb_filter[0], spade_mid, nb_filter[0] / ss_scale)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):

        x0_0 = self.conv0_0(input)
        x0_0 = self.SPADE0_0(x0_0, x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.SPADE1_0(x1_0, x1_0)
        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.SPADE2_0(x2_0, x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.SPADE3_0(x3_0, x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.SPADE4_0(x4_0, x4_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.SPADE3_1(x3_1, x3_1)
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x2_2 = self.SPADE2_2(x2_2, x2_2)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x1_3 = self.SPADE1_3(x1_3, x1_3)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        x0_4 = self.SPADE0_4(x0_4, x0_4)


        output = self.final(x0_4)
        return output

class ProgUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3,  deep_supervision=False, **kwargs):
        super().__init__()

        #nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [64, 128,  256, 512,1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final0 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final1 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=1)



    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output0 = self.final0(x0_4)
        output1 = self.final1(x1_3)
        output2 = self.final2(x2_2)
        output3 = self.final3(x3_1)
        return  [output0, output1, output2, output3]

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3,  deep_supervision=False, **kwargs):
        super().__init__()

        #nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [ 64, 128, 256, 512, 1024]


        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        #nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class UNet_ori(nn.Module):
    def __init__(self,num_classes, input_channels=3,  deep_supervision=False, **kwargs):
        super().__init__()
        #nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [64, 128, 256, 512, 1024]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=input_channels, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Up5 = up_conv(ch_in=nb_filter[4], ch_out=nb_filter[3])
        self.Up_conv5 = conv_block(ch_in=nb_filter[4], ch_out=nb_filter[3])

        self.Up4 = up_conv(ch_in=nb_filter[3], ch_out=nb_filter[2])
        self.Up_conv4 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[2])

        self.Up3 = up_conv(ch_in=nb_filter[2], ch_out=nb_filter[1])
        self.Up_conv3 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[1])

        self.Up2 = up_conv(ch_in=nb_filter[1], ch_out=nb_filter[0])
        self.Up_conv2 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1