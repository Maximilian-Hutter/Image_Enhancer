import numpy as np
import torch
import torch as nn
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import socket
import argparse
from prefetch_generator import BackgroundGenerator
import time
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import itertools
import os
import logging

# import own dataset
from get_data import ImageDataset
from models import NeuralNet

# replace Contrast_expander with the final name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch dummmyname')
    parser.add_argument('--ray_tune', type=bool, default=False, help=("Use ray tune to tune parameters"))
    parser.add_argument('--train_data_path', type=str, help=("path for the data"))
    parser.add_argument('--R_kernel', type=int, default=3, help=("set kernel size for Red channel"))
    parser.add_argument('--G_kernel', type=int, default=5, help=("set kernel size for Green channel"))
    parser.add_argument('--B_kernel', type=int, default=7, help=("set kernel size for Blue channel"))
    parser.add_argument('--D_kernel', type=int, default=7, help=("set kernel size for Depth channel"))
    parser.add_argument('--activation', type=str, default="PReLU", help=("set activation function"))
    parser.add_argument('--imgheight', type=int, default=1944, help=("set the height of the image in pixels"))
    parser.add_argument('--imgwidth', type=int, default=2592, help=("set the width of the image in pixels"))
    parser.add_argument('--imgchannels', type=int, default=4, help=("set the channels of the Image (default = RGBD -> 4)"))
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
    parser.add_argument('--filters', type=int, default=8, help="set number of filters")
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

    parser.add_argument
    opt = parser.parse_args()
    np.random.seed(opt.seed)    # set seed to default 123 or opt
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    gpus_list = range(opt.gpus)
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(opt)  # print the chosen parameters

    # defining shapes
    imgshape = (opt.imgchannels, opt.imgheight, opt.imgwidth)

    # dataloader
    print('==> Loading Datasets')
    dataloader = DataLoader(ImageDataset(opt.train_data_path, opt.use_patches, opt.augment_data), batch_size=opt.batchSize, shuffle=True, num_workers=opt.threads)

    # instantiate model

    #Generator = ESRGANplus(opt.channels, filters=opt.filters,hr_shape=hr_shape, n_resblock = opt.n_resblock, upsample = opt.upsample)
    Net = NeuralNet(in_features=opt.imgchannels, filters=opt.filters, stride=opt.stride, padding=opt.padding,depth=opt.net_depth, U_Net=opt.Unet, activation_function=opt.activation)

    #parameters
    pytorch_params = sum(p.numel() for p in Net.parameters())
    print("Network parameters: {}".format(pytorch_params))

    # loss
    #adverserial_criterion = torch.nn.BCEWithLogitsLoss()
    loss = MSELoss()

    # run on gpu
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    if cuda:
        Net = Net.cuda(gpus_list[0])
        loss = loss.cuda(gpus_list[0])

    optimizer = optim.Adam(Net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    # load checkpoint/load model
    star_n_iter = 0
    start_epoch = 0
    if opt.resume:
        checkpoint = torch.load(opt.save_folder) ## look at what to load
        start_epoch = checkpoint['epoch']
        start_n_iter = checkpoint['n_iter']
        optimizer.load_state_dict(checkpoint['optim'])
        print("last checkpoint restored")

    def checkpointG(epoch):
        model_out_path = opt.save_folder+opt.activation+opt.model_type+".pth".format(epoch)
        torch.save(Net.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # Tensorboard
    writer = SummaryWriter()

    # define Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    for epoch in range(start_epoch, opt.nEpochs):
        epoch_loss = 0
        Net.train()
        start_time = time.time()

        for i, imgs in enumerate(BackgroundGenerator(dataloader,1)):

            img = Variable(imgs["img"].type(Tensor))
            recovered = Variable(imgs["recovered"].type(Tensor))
            depth = Variable(imgs["depth"].type(Tensor))    # merge depth and image to RGBD form

            if cuda:    # put variables to gpu
                img = img.to(gpus_list[0])
                recovered = recovered.to(gpus_list[0])
                depth = depth.to(gpus_list[0])

            img = torch.cat((img, depth),0) # in der ersten dimension depth anhängen

            prepare_time = time.time() - start_time

            # start train
            optimizer.zero_grad()

            generated_image = Net(img)

            Loss = loss(generated_image, recovered)
            epoch_loss += Loss.data

            Loss.backward()
            optimizer.step()

            #compute time and compute efficiency and print information
            process_time = time.time() - start_time
            print("process time: {}, Number of Iteration {}/{}".format(process_time,i , (len(dataloader)-1)))
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))
            start_time = time.time()

    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    print('===> Building Model ', opt.model_type)
    if opt.model_type == 'Contrast_expander':
        Net = Net


    print('----------------Network architecture----------------')
    print_network(Net)
    print('----------------------------------------------------')