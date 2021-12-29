import torch
import torch.nn as nn
from .memonger import SublinearSequential_every
import time

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        cl_layers = [*list(classifier.children())]
        self.classifier_ckpt = SublinearSequential_every(*cl_layers)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier_ckpt(x)
    
        return x

    
    def forward_time(self, x):
    #def forward(self, x):
        #output = self.features(x)

        func_features = list(self.features.children())
        
        for i in range(0,len(func_features)):
            st = time.perf_counter()
            x = func_features[i](x)
            torch.cuda.synchronize()
            en = time.perf_counter()
            print(format(en-st, '.20f'))
        
        
        x = x.view(x.size()[0], -1)
        
        func_classifier = list(self.classifier.children())
        
        for i in range(0,len(func_classifier)):
            st = time.perf_counter()
            x = func_classifier[i](x)
            torch.cuda.synchronize()
            en = time.perf_counter()
            print(format(en-st, '.20f'))
            
        return x

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return SublinearSequential_every(*layers)
    #return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16():
    return VGG(make_layers(cfg['D'], batch_norm=False))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))
