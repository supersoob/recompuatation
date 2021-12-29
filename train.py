import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import pandas as pd
from get_mem import _add_memory_hooks
import sys
import torch.optim as optim
import argparse
from configparser import ConfigParser

from memonger import vgg_org_imagenet as get_vgg_org
from memonger import vgg_time_imagenet as get_vgg_time
from memonger import vgg_mem_imagenet as get_vgg_mem

from memonger import resnet_org_cifar as get_resnet_org

from memonger import inception_org as get_incep_org
from memonger import inception_time as get_incep_time
from memonger import inception_mem as get_incep_mem


GPU_NUM = 1
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def plot_mem(
        df,
        exps=None,
        normalize_call_idx=True,
        normalize_mem_all=True,
        filter_fwd=False,
        return_df=False,
        output_file=None
):
    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp == exp]

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            #df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2 ** 20

        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]
            # df_ = df_[df_.call_idx < df_[df_.layer_idx=='bwd'].call_idx.min()]

        plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)
        if output_file:
            plot.get_figure().savefig(output_file)

    if return_df:
        return df_


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--batch-size', '-bs', type=int, help='integer for batch size')
    parser.add_argument('--batch-num', '-bn', type=int, help='number of batch size')
    parser.add_argument('--epoch-size', '-e', type=int, default=1)
    parser.add_argument('--break-point', '-bp', type=int, default=100, help='iteration break point')
    parser.add_argument('--model', '-m', type=str, help='model for training')
    parser.add_argument('--dataset','-data', type=str, default='imagenet', help='dataset for training (cifar/imagenet)')
    parser.add_argument('--version','-v', type=str, default='None', help='setting for training (time/memory)')

    args = parser.parse_args()

    batch_size = args.batch_size
    batch_num = args.batch_num
    epoch_size = args.epoch_size
    break_point = args.break_point
    model = args.model
    dataset = args.dataset
    version = args.version


    config = ConfigParser()

    config['setting'] = {
        'model' : model,
        'dataset' : dataset,
        'batch_size' : str(batch_size)
    }

    with open(f'./conf_{batch_num}.ini','w') as f:
        config.write(f)


    # model loading
    global net

    if model == "resnet50":
        net = get_resnet_org.ResNet50()
    elif model == "resnet101":
        net = get_resnet_org.ResNet101()
    elif model == "vgg16":
        if version == "time":
            net = get_vgg_time.vgg16()
        elif version == "memory":
            net = get_vgg_mem.vgg16()
        else:
            net = get_vgg_org.vgg16()
    elif model == "vgg16_bn":
        if version == "time":
            net = get_vgg_time.vgg16_bn()
        elif version == "memory":
            net = get_vgg_mem.vgg16_bn()
        else:
            net = get_vgg_org.vgg16_bn()
    elif model == "inception":
        if version == "time":
            net = get_incep_time.inception_v3()
        elif version == "memory":
            net = get_incep_mem.inception_v3()
        else:
            net = get_incep_org.inception_v3()


    if version == "memory":
        break_point = 1

    net.to('cuda')

    # dataset loading
    global trainloader

    if dataset == "imagenet" and model!="inception":
        data_path = "./data/imagenet-bird"
        traindir = os.path.join(data_path, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        trainloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transform), batch_size=batch_size, shuffle=False, num_workers=0)

    elif dataset == "imagenet" and model=="inception":
        data_path = "./data/imagenet-bird"
        traindir = os.path.join(data_path, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize,
            ])
        trainloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    
    elif dataset == "cifar":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR100(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False, num_workers=0)


    # optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #memory plot setting
    #mem_log_total=[]
    #exp="vgg_bn"
    #print(net)

    # switch to train mode
    net.train()
    
    end = time.time()
    
    # train
    for epoch in range(epoch_size):

        # progress bar
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(trainloader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))


        for i, (images, target) in enumerate(trainloader):
            if i > break_point:
                break
            
            data_time.update(time.time() - end)

            images = images.to('cuda')
            target = target.to('cuda')

            if model == "inception":
                output, aux = net(images)
            else:
                output = net(images)
            
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

    print('Finished Training')


