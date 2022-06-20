import torch
import torch.nn as nn

class ResNet50Layer(nn.Module):
    def __init__(self, in_features, filters, out_features, first_block=0, last_block=0):
        super(ResNet50Layer,self).__init__()

        print(first_block)
        if first_block == 1:
            self.conv1 = nn.Conv2d(in_features, filters,1)
        else:
            self.conv1 = nn.Conv2d(filters, filters, 1)

        if last_block == 1:
            self.conv3 = nn.Conv2d(filters, out_features, 1)
        else:
            self.conv3 = nn.Conv2d(filters, filters,1)

        if first_block:
            self.conv2 = nn.Conv2d(filters, filters, 3, stride=2)
        else:
            self.conv2 = nn.Conv2d(filters, filters, 3, stride=1)
        
        self.norm = nn.InstanceNorm2d(filters)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)

        return out

class ResNet50SmallBlock(nn.Module):
    def __init__(self, in_features, out_features, filters, first_block=0, last_block=0):
        super(ResNet50SmallBlock, self).__init__()

        Layers = []
        if first_block:
            Layers.append(ResNet50Layer(in_features, out_features, filters, first_block, last_block))
            Layers.append(ResNet50Layer(in_features, out_features, filters,first_block, last_block))  
        else:  
            Layers.append(ResNet50Layer(in_features, out_features, filters,first_block, last_block))
            Layers.append(ResNet50Layer(in_features, out_features, filters,first_block, last_block))

        self.block = nn.Sequential(*Layers)

    def forward(self, x):
        res = x
        out = self.block(x)
        torch.cat((out,res),0)

        return out

class ResNet50Block(nn.Module):
    def __init__(self, in_features, out_features, filters,blocksnum):
        super(ResNet50Block, self).__init__()

        Layers = []
        for i in range(blocksnum):
            if i == 0:
                Layers.append(ResNet50SmallBlock(in_features, out_features, filters, first_block=1))
            elif i == blocksnum:
                Layers.append(ResNet50SmallBlock(in_features, out_features, filters, last_block = 1))
            else:
                Layers.append(ResNet50SmallBlock(in_features, out_features, filters))

        self.block = nn.Sequential(*Layers)

    def forward(self,x):
        out = self.block(x)
        res = out
        return out, res
