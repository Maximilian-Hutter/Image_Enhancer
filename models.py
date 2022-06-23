from imp import init_frozen
from numpy import outer
import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange

from data_augmentation import lightmap_gen

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, activation_function, padding=1):
        super(ConvBlock, self).__init__()

        Layers = []
        Layers.append(nn.Conv2d(in_features, out_features, 3, padding=padding, bias=False)) # set bias true if result is not good
        if activation_function == "ReLU":
            Layers.append(nn.ReLU())
        if activation_function == "PReLU":
            Layers.append(nn.PReLU())
        if activation_function == "LReLU":
            Layers.append(nn.LeakyReLU())
        if activation_function == "SeLU":
            Layers.append(nn.SELU())
        if activation_function == "SiLU":
            Layers.append(nn.SiLU())

        Layers.append(nn.InstanceNorm2d(in_features))
        self.Block = nn.Sequential(*Layers)
        
    def forward(self, x):

        x = self.Block(x)

        return x

class UpConv(nn.Module):
    def __init__(self, in_features, kernel_size, scale_factor):
        super(UpConv, self).__init__()

        Layers = []
        Layers.append(nn.Upsample(scale_factor))
        Layers.append(nn.Conv2d(in_features, in_features, kernel_size, 1, padding=1,bias=False))
        Layers.append(nn.Conv2d(in_features, in_features*scale_factor, kernel_size, 1, padding=1,bias=False))

        self.Block = nn.Sequential(*Layers)
    def forward(self, x):

        #print(x.shape)
        x = self.Block(x)

        return x

class ResNet50LastLayer(nn.Module):
    def __init__(self, in_features, out_features, filters):
        super(ResNet50LastLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_features, filters, 1)
        self.conv2 = nn.Conv2d(filters, filters,3, padding=1)
        self.conv3 = nn.Conv2d(filters, out_features, 1, stride=2)    
        self.norm = nn.InstanceNorm2d(out_features)

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.norm(x)

        return x

class ResNet50Layer(nn.Module):
    def __init__(self, in_features,filters):
        super(ResNet50Layer,self).__init__()

        self.conv1 = nn.Conv2d(in_features, filters, 1)
        self.conv2 = nn.Conv2d(filters,filters,3,padding=1)
        self.conv3 = nn.Conv2d(filters, in_features, 1)
        self.norm = nn.InstanceNorm2d(in_features)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.norm(x)

        return x

class ResNet50SmallBlock(nn.Module):
    def __init__(self, in_features, out_features, filters, last_block=0):
        super(ResNet50SmallBlock, self).__init__()

        Layers = []
        self.last_block = last_block
        if last_block:
            Layers.append(ResNet50Layer(in_features, filters))
            Layers.append(ResNet50LastLayer(in_features, out_features, filters))
        else:
            Layers.append(ResNet50Layer(in_features, filters))
            Layers.append(ResNet50Layer(in_features, filters))

        self.block = nn.Sequential(*Layers)
        self.resconv = ResNet50LastLayer(in_features, out_features, filters)

    def forward(self, x):
        res = x
        out = self.block(x)
        #print(out.shape)
        if self.last_block:
            res = self.resconv(res)
    
        out = torch.add(out,res)# changes minibatch size to double the previous

        return out

class ResNet50Block(nn.Module):
    def __init__(self, in_features, out_features, filters,blocksnum):
        super(ResNet50Block, self).__init__()

        Layers = []
        for i in range(blocksnum):
            if i == (blocksnum-1):
                Layers.append(ResNet50SmallBlock(in_features, out_features, filters, last_block=1))
            else:
                Layers.append(ResNet50SmallBlock(in_features, out_features, filters))

        self.block = nn.Sequential(*Layers)

    def forward(self,x):
        out = self.block(x)
        res = out
        return out, res

class skipBlock(nn.Module):
    def __init__(self, in_feature):
        super(skipBlock, self).__init__()

        self.up = nn.Upsample(2)
        self.conv1 = nn.Conv2d(in_feature,in_feature*2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_feature*2, in_feature*2, 3,padding=1)

        self.conv3 = nn.Conv2d(in_feature, in_feature*2, 3,padding=2)

        self.convout = nn.Conv2d(in_feature*2, in_feature*2, 1)

    def forward(self, x, res):

        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        
        res = self.conv3(res)

        print(x.shape)
        print(res.shape)
        out = torch.add(x,res)

        out = self.convout(out)

        return out

class LightAttentionModule(nn.Module):
    def __init__(self,in_features, activation_function):
        super(LightAttentionModule, self).__init__()

        self.conv1 = ConvBlock(in_features, in_features, activation_function, 3)
        self.conv2 = ConvBlock(in_features, in_features, activation_function, 3)
        self.sigmoid = nn.Sigmoid()

        self.conv3 = ConvBlock(in_features, in_features, activation_function, 3)

        self.conv4 = ConvBlock(in_features, in_features, activation_function, 3)
        self.up = UpConv(in_features,3,2)
        self.conv5 = ConvBlock(in_features*2, in_features, activation_function, 3)

    def forward(self, x, y):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        y = self.conv3(x)

        x = torch.cat((x,y),0)

        out = self.conv4(x)
        out = self.up(out)
        out = self.conv5(out)

        return out

class ConvModule(nn.Module):
    def __init__(self, in_features):
        super(ConvModule,self).__init__()

        self.conv1 = nn.Conv2d(in_features, 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 4,3, stride=3)
        self.up1 = UpConv(4, 3, 2)
        self.up2 = UpConv(8, 3, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up1(x)
        out = self.up2(x)

        return out

class NeuralNet(nn.Module):
    def __init__(self, in_features, activation_function, filters):
        super(NeuralNet,self).__init__()

        self.preconv = ConvBlock(in_features, 512, "ReLU", padding=0)
        self.block1 = ResNet50Block(512, 256, filters, 3)
        self.block2 = ResNet50Block(256, 128, filters, 6)
        self.block3 = ResNet50Block(128, 64, filters, 4)
        self.block4 = ResNet50Block(64, 32, filters, 3)
        self.resconv = nn.Conv2d(32, 16, 7, 2)
        self.norm = nn.InstanceNorm2d(16)

        self.convmodule = ConvModule(16)
        
        self.skip = skipBlock(16)
        self.skip2 = skipBlock(32)
        self.skip3 = skipBlock(64)
        self.skip4 = skipBlock(128)
        

        
        #self.up3 = UpConv()

        self.LAM = LightAttentionModule(in_features, activation_function)
        #self.dimcorrect = DimensionalityCorrection()
        self.up = UpConv(64, 3, 2)
        self.conv = ConvBlock(128,128,activation_function)
        self.up2 = UpConv(128, 3, 2)

        self.skip5 = skipBlock(256)
        self.shuffle = nn.PixelShuffle((in_features / 256))

    def forward(self,x, lightmap):

        x = self.preconv(x)
        # Res Net Feature Extract
        x, res1 = self.block1(x)
        x, res2 = self.block2(x)
        x, res3 = self.block3(x)
        x, res4 = self.block4(x)
        x = self.resconv(x)
        res5 = x
        x = self.norm(x)
        # Lowest point in U net
        x = self.convmodule(x)

        # decoder
        x = self.skip(x, res5)
        x = self.skip2(x, res4)
        x = self.skip3(x, res3)
        x = self.skip4(x, res2)

        lightmap = self.dimcorrect(lightmap)
        lightmap = self.up(lightmap)
        lightmap = self.conv(lightmap)
        lightmap = self.up2(lightmap)
        #lightmap = self.up3(lightmap)

        out = self.LAM(lightmap, x)
        out = self.skip5(out, res1)
        out = self.shuffle(out)

        return out

###############
#   https://arxiv.org/pdf/2109.06688v1.pdf LAM
#   model idea
#   feature extraction
#   light attention using and augmented picture
#   
