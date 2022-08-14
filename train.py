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
from data_augmentation import lightmap_gen
from criterion import luminance_criterion, abs_criterion, color_criterion
import tqdm

# import own dataset
from get_data import ImageDataset
from models import NeuralNet

# replace Contrast_expander with the final name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch HDRnet')
    parser.add_argument('--ray_tune', type=bool, default=False, help=("Use ray tune to tune parameters"))
    parser.add_argument('--train_data_path', type=str, default="C:/Data/NTIRE", help=("path for the data"))
    parser.add_argument('--valid_data_path', type=str, default="C:/Data/NTIRE/valid", help=("path for the data"))
    parser.add_argument('--valid_intervall', type=int, default=20, help=("intervall for validation loop"))
    parser.add_argument('--activation', type=str, default="PReLU", help=("set activation function"))
    parser.add_argument('--imgheight', type=int, default=448, help=("set the height of the image in pixels"))
    parser.add_argument('--imgwidth', type=int, default=448, help=("set the width of the image in pixels"))
    parser.add_argument('--imgchannels', type=int, default=3, help=("set the channels of the Image (default = RGB)"))
    parser.add_argument('--augment_data', type=bool, default=False, help=("if true augment train data"))
    parser.add_argument('--batchsize', type=int, default=4, help=("set batch Size"))
    parser.add_argument('--gpu_mode', type=bool, default=True) 
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--mini_batch',type=int, default=16, help='mini batch size')
    parser.add_argument('--sample_interval',type=int, default=100, help='Number of epochs for learning rate decay')
    parser.add_argument('--resume',type=bool, default=False, help='resume training/ load checkpoint')
    parser.add_argument('--model_type', type=str, default="HDR", help="set type of model")
    parser.add_argument('--filters', type=int, default=64, help="set number of filters")
    parser.add_argument('--beta1',type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2',type=float, default=0.999, help='decay of first order momentum of gradient')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--nEpochs', type=int, default=250, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0004, help="learning rate")
    parser.add_argument('--snapshots', type=int, default=10, help="number of epochs until a checkpoint is made")
    parser.add_argument('--loss_weight', type=float, default=1, help="set weight for loss addition")
    parser.add_argument
    opt = parser.parse_args()
    np.random.seed(opt.seed)    # set seed to default 123 or opt
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    gpus_list = range(opt.gpus)
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(opt)  # print the chosen parameters

    # dataloader
    print('==> Loading Datasets')
    dataloader = DataLoader(ImageDataset(opt.train_data_path, opt.augment_data), batch_size=opt.batchsize, shuffle=True, num_workers=opt.threads)
    #testloader = DataLoader(ImageDataset(opt.valid_data_path, opt.augment_data), batch_size=opt.batchsize, shuffle=True, num_workers=opt.threads)

    # instantiate model

    #Generator = ESRGANplus(opt.channels, filters=opt.filters,hr_shape=hr_shape, n_resblock = opt.n_resblock, upsample = opt.upsample)
    Net = NeuralNet(in_features=opt.imgchannels,activation_function=opt.activation, filters=opt.filters)

    #parameters
    pytorch_params = sum(p.numel() for p in Net.parameters())
    print("Network parameters: {}".format(pytorch_params))

    # loss
    #adverserial_criterion = torch.nn.BCEWithLogitsLoss()
    #abs_crit = abs_criterion()
    #lum_crit = luminance_criterion()
    mse = MSELoss(reduction="sum")

    # run on gpu
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    if cuda:
        Net = Net.cuda(gpus_list[0])
        mse = mse.cuda(gpus_list[0])
        #abs_crit = abs_crit.cuda(gpus_list[0])
        #lum_crit = lum_crit.cuda(gpus_list[0])

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
        model_out_path = opt.save_folder+str(epoch)+opt.model_type+".pth".format(epoch)
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
        correct = 0
        for i, imgs in enumerate(BackgroundGenerator(dataloader,1)):

            img = Variable(imgs["img"].type(Tensor))
            lightmap = Variable(imgs["lightmap"].type(Tensor))
            label = Variable(imgs["label"].type(Tensor))

            if cuda:    # put variables to gpu
                img = img.to(gpus_list[0])
                lightmap = lightmap.to(gpus_list[0])
                label = label.to(gpus_list[0])

            prepare_time = time.time() - start_time

            # start train
            optimizer.zero_grad()

            generated_image = Net(img, lightmap)

            #lum_loss = luminance_criterion(generated_image, label)    
            #abs_loss = abs_criterion(generated_image, label)      
            #color_loss = color_criterion(generated_image, label)  

            #Loss = lum_loss +  abs_loss.mean() + color_loss
            #Loss = abs_loss.mean()
            Loss = mse(generated_image, label)
            #Loss = lum_loss
            train_acc = torch.sum(generated_image == label)
            epoch_loss += Loss.data
            Loss.sum().backward()
            optimizer.step()

            #compute time and compute efficiency and print information
            process_time = time.time() - start_time
            print("process time: {}, Number of Iteration {}/{}".format(process_time,i , (len(dataloader)-1)))
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))
            
            # if epoch % opt.valid_intervall == 0:
            #     test()
            
            start_time = time.time()

        # def test():
        #     Net.eval()
        #     correct=0
        #     total=0

        #     with torch.no_grad():
        #         for imgs in tqdm(testloader):

        #             if cuda:
        #                 img = Variable(imgs["img"].type(Tensor)).to(gpus_list[0])
        #                 lightmap = Variable(imgs["lightmap"].type(Tensor)).to(gpus_list[0])
        #                 label = Variable(imgs["label"].type(Tensor)).to(gpus_list[0])
        #             else:
        #                 label = Variable(imgs["label"].type(Tensor))
        #                 lightmap = Variable(imgs["lightmap"].type(Tensor))
        #                 img = Variable(imgs["img"].type(Tensor))

        #             outputs=Net(img, lightmap)

        #             lum_loss = luminance_criterion(generated_image, label)    
        #             abs_loss = abs_criterion(generated_image, label)      
        #             color_loss = color_criterion(generated_image, label)  

        #             loss = lum_loss +  abs_loss.mean() + color_loss

        #             predicted = outputs
                    
        #             total += label.size(0)
        #             correct += predicted.eq(label).sum().item()

        #     accu=100.*correct/total

        #     valid_loss = loss.item() * img.size(0)

        #     print('Test Loss: %.3f | Accuracy: %.3f'%(valid_loss, accu)) 


        
        if (epoch+1) % (opt.snapshots) == 0:
            checkpointG(epoch)

        Accuracy = 100*train_acc / len(dataloader)
        writer.add_scalar('loss', Loss.sum(), global_step=epoch)
        writer.add_scalar('accuracy',Accuracy, global_step=epoch)
        print("===> Epoch {} Complete: Avg. loss: {:.4f} Accuracy {}".format(epoch, ((epoch_loss/2) / len(dataloader)), Accuracy))

    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    print('===> Building Model ', opt.model_type)
    if opt.model_type == 'HDR':
        Net = Net


    print('----------------Network architecture----------------')
    print_network(Net)
    print('----------------------------------------------------')