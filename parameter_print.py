from models import NeuralNet
import numpy as np
import torch
import torch.nn as nn
import argparse
import socket
import torch.backends.cudnn as cudnn
from torchsummary import summary
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch dummmyname')
    parser.add_argument('--ray_tune', type=bool, default=False, help=("Use ray tune to tune parameters"))
    parser.add_argument('--train_data_path', type=str, help=("path for the data"))
    parser.add_argument('--R_kernel', type=int, default=3, help=("set kernel size for Red channel"))
    parser.add_argument('--G_kernel', type=int, default=5, help=("set kernel size for Green channel"))
    parser.add_argument('--B_kernel', type=int, default=7, help=("set kernel size for Blue channel"))
    parser.add_argument('--D_kernel', type=int, default=7, help=("set kernel size for Depth channel"))
    parser.add_argument('--activation', type=str, default="PReLU", help=("set activation function"))
    parser.add_argument('--imgheight', type=int, default=312, help=("set the height of the image in pixels"))
    parser.add_argument('--imgwidth', type=int, default=312, help=("set the width of the image in pixels"))
    parser.add_argument('--imgchannels', type=int, default=3, help=("set the channels of the Image (default = RGBD -> 4)"))
    parser.add_argument('--augment_data', type=bool, default=False, help=("if true augment train data"))
    parser.add_argument('--use_patches', type=bool, default=True, help=("if true use cropped parts of trained image"))
    parser.add_argument('--batchsize', type=int, default=4, help=("set batch Size"))
    parser.add_argument('--gpu_mode', type=bool, default=True) 
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--disc_save_folder',type=str, default='weights/', help='Location to save Discriminator')
    parser.add_argument('--mini_batch',type=int, default=16, help='mini batch size')
    parser.add_argument('--sample_interval',type=int, default=100, help='Number of epochs for learning rate decay')
    parser.add_argument('--resume',type=bool, default=False, help='resume training/ load checkpoint')
    parser.add_argument('--beta1',type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2',type=float, default=0.999, help='decay of first order momentum of gradient')
    parser.add_argument('--model_type', type=str, default="Contrast_expander", help="set type of model")
    parser.add_argument('--net_depth', type=int, default=None, help="set the U-Net depth of the Network")
    parser.add_argument('--Unet', type=bool, default=False, help="set if using U-Net structure")
    parser.add_argument('--stride', type=int, default=1, help="set stride of CBAM")
    parser.add_argument('--padding', type=int, default=1, help="set padding of CBAM")
    parser.add_argument('--filters', type=int, default=32, help="set number of filters")

    parser.add_argument
    opt = parser.parse_args()
    np.random.seed(opt.seed)    # set seed to default 123 or opt
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(opt)  # print the chosen parameters

    # defining shapes

    Net = NeuralNet(in_features=opt.imgchannels, activation_function=opt.activation, filters=opt.filters)#.to("cuda:0")

    summary(Net, [(opt.imgchannels, opt.imgheight, opt.imgwidth), (opt.imgchannels, opt.imgheight, opt.imgwidth)])

    # pytorch_params = sum(p.numel() for p in Net.parameters())
    # print("Network parameters: {}".format(pytorch_params))

    # def print_network(net):
    #     num_params = 0
    #     for param in net.parameters():
    #         num_params += param.numel()
    #     print(net)
    #     print('Total number of parameters: %d' % num_params)

    # print('===> Building Model ')
    # Net = Net


    # print('----------------Network architecture----------------')
    # print_network(Net)
    # print('----------------------------------------------------')