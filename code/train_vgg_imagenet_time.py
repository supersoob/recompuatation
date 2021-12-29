import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

from memonger import SublinearSequential
from memonger import vgg_org_imagenet as get_vgg_org


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

GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

#with torch.cuda.device (GPU_NUM):
net = get_vgg_org.vgg16()
net.to('cuda')

bs = int(sys.argv[1])


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
    datasets.ImageFolder(traindir, transform), batch_size=bs, shuffle=False, num_workers=0)



"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=bs, shuffle=False, num_workers=0)
"""


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epoch = 1
batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
losses = AverageMeter('Loss', ':.4e')
top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')
progress = ProgressMeter(
    len(trainloader),
    [batch_time, data_time, losses, top1, top5],
    prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
net.train()

mem_log_total=[]

exp="org"




#with torch.autograd.profiler.profile(use_cuda=True) as prof:
for i, (images, target) in enumerate(trainloader):
    if i>50:
        break

    images = images.to('cuda')
    target = target.to('cuda')

    a = time.perf_counter()
    output = net(images)
    torch.cuda.synchronize()
    b = time.perf_counter()

    print(f"[{i}] forward pass time : ", b-a)

    loss = criterion(output, target)
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
   
    #b = time.perf_counter()
     
    #print(b-a)


print('Finished Training')


