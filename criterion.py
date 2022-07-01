import torch


def luminance_criterion(gen_img, target, alpha=0.05):

    loss =  torch.mean(torch.mean((gen_img - target)**2, [1, 2, 3]) - alpha * torch.pow(torch.mean(gen_img - target, [1, 2, 3]), 2))
    
    return loss

def abs_criterion(gen_img, target):

    loss = torch.abs(gen_img - target)

    return loss