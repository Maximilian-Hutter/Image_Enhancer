from imp import init_frozen
from numpy import outer
import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange

from data_augmentation import lightmap_gen

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, activation_function, padding=1, stride=1):
        super(ConvBlock, self).__init__()

        Layers = []
        Layers.append(nn.Conv2d(in_features, out_features, 3,stride=stride, padding=padding, bias=False)) # set bias true if result is not good
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
    def __init__(self, in_features,out_features, scale_factor):
        super(UpConv, self).__init__()

        Layers = []
        Layers.append(nn.Upsample(scale_factor = scale_factor))
        Layers.append(nn.Conv2d(in_features,out_features, 3, 1, padding=1))

        self.Block = nn.Sequential(*Layers)
    def forward(self, x):

        #print(x.shape)
        x = self.Block(x)
        #print(x.shape)

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
    def __init__(self, in_features, out_features, size):
        super(skipBlock, self).__init__()

        size = int(size)

        self.up = nn.Upsample(size=size)
        self.conv1 = nn.Conv2d(in_features,out_features, 3, padding=1)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, padding=1)

        self.conv3 = nn.Conv2d(out_features, out_features, 3,padding=1)

        self.convout = nn.Conv2d(out_features, out_features, 1)

    def forward(self, x, res):

        #print(x.shape)
        x = self.up(x)
        #print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        
        res = self.conv3(res)
        #res = self.up(res)
        #print(x.shape)
        #print(res.shape)
        out = torch.add(x,res)

        out = self.convout(out)

        return out

class LightAttentionModule(nn.Module):
    def __init__(self,in_features, out_features, filters, activation_function):
        super(LightAttentionModule, self).__init__()

        self.lightmap = LightMapAttention(in_features, filters, filters, activation_function)

        self.conv3 = ConvBlock(in_features, filters, activation_function)

        self.conv4 = ConvBlock(filters, filters, activation_function, stride=2)
        self.up = UpConv(filters,out_features,2)
        self.conv5 = ConvBlock(out_features, out_features, activation_function)

    def forward(self, x, y):

        x = self.lightmap(x)

        y = self.conv3(y)

        out = torch.mul(x,y)

        out = self.conv4(x)
        out = self.up(out)
        out = self.conv5(out)

        return out

class LightMapAttention(nn.Module):
    def __init__(self, in_features, out_features, filters, activation_function):
        super(LightMapAttention, self).__init__()

        self.conv1 = ConvBlock(in_features, filters, activation_function)
        self.conv2 = ConvBlock(filters, out_features, activation_function)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        return x

class ConvModule(nn.Module):
    def __init__(self, in_features):
        super(ConvModule,self).__init__()

        self.conv1 = nn.Conv2d(in_features, in_features*2, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_features*2, in_features*4,3, stride=2, padding=1)
        self.up1 = UpConv(in_features*4, in_features*2, 2)
        self.up2 = UpConv(in_features*2, in_features, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        #print(x.shape)
        x = self.up1(x)
        #print(x.shape)
        out = self.up2(x)
        #print(out.shape)
        return out

class LightmapPath(nn.Module):
    def __init__(self, in_features, out_features,size):
        super(LightmapPath, self).__init__()

        # WIP
        size = int(size)
        self.conv = nn.Conv2d(in_features, out_features, 3, padding=1)
        self.up = nn.Upsample(size)

    def forward(self, x):

        x = self.conv(x)
        x = self.up(x)

        return x

class NeuralNet(nn.Module):
    def __init__(self, in_features, activation_function, filters, size=448):
        super(NeuralNet,self).__init__()

        self.preconv = nn.Conv2d(in_features, 64, 7, stride=1, padding=3)
        self.block1 = ResNet50Block(64, 128, filters, 3)
        self.block2 = ResNet50Block(128, 256, filters, 4)
        self.block3 = ResNet50Block(256, 512, filters, 6)
        self.block4 = ResNet50Block(512, 1024, filters, 3)
        self.block5 = ResNet50Block(1024, 2048, filters, 1)
        #self.resconv = nn.Conv2d(32, 16, 3, stride=2)
        self.norm = nn.InstanceNorm2d(2048)

        self.convmodule = ConvModule(2048)
        
        self.skip = skipBlock(2048, 1024, size/16)
        self.skip2 = skipBlock(1024, 512, size/8)
        self.skip3 = skipBlock(512, 256, size/4)
        self.skip4 = skipBlock(256, 128, size/2)

        self.lightmappath = LightmapPath(1, 128, size/2)

        self.LAM = LightAttentionModule(128,128,filters,  activation_function)
        self.upsample = UpConv(in_features, in_features, scale_factor=2)
        self.skip5 = skipBlock(128, 3, size)

    def forward(self,x, lightmap):

        res = x
        x = self.preconv(x)
        #res0 = x
        #print(x.shape)
        # Res Net Feature Extract
        x, res1 = self.block1(x)
        #print(x.shape)

        x, res2 = self.block2(x)
        #print(x.shape)

        x, res3 = self.block3(x)
        #print(x.shape)

        x, res4 = self.block4(x)
        #print(x.shape)

        x, res5 = self.block5(x)
        #res5 = x
        x = self.norm(x)
        # Lowest point in U net
        x = self.convmodule(x)

        #print(x.shape)
        # decoder
        #print("res:")
        #print(res5.shape)
        #print(x.shape)
        x = self.skip(x, res4)
        #print(x.shape)
        x = self.skip2(x, res3)
        #print(x.shape)

        lightmap = self.lightmappath(lightmap)

        x = self.skip3(x, res2)
        #print(x.shape)
        x = self.skip4(x, res1)
        #print(x.shape)
        out = self.LAM(lightmap, x)
        #print(out.shape)
        out = self.skip5(out, res)
        #print(out.shape)
        #out = torch.add(out, res)

        return out

###############
#   https://arxiv.org/pdf/2109.06688v1.pdf LAM
#   model idea
#   feature extraction
#   light attention using and augmented picture
#   
