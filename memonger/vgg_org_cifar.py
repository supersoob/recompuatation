import torch
import torch.nn as nn
import time
from memory_profiler import profile

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

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            #nn.ReLU(), 
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            #nn.ReLU(), 
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

   
    #def forward(self, x):
    #    out = self.features(x)
    #    out = out.view(out.size()[0], -1)
    #    out = self.classifier(out)
    
    #    return out
   
    #def forward_mem(self,x):
    def forward(self,x):

        func_features = list(self.features.children())
        
        x = func_features[0](x) 
        print(x.element_size() * x.nelement())
        x = func_features[1](x) 
        print( x.element_size() * x.nelement())
        x = func_features[2](x) 
        print(x.element_size() * x.nelement())
        x = func_features[3](x) 
        print( x.element_size() * x.nelement())
        x = func_features[4](x) 
        print( x.element_size() * x.nelement())
        x = func_features[5](x) 
        print( x.element_size() * x.nelement())
        x = func_features[6](x) 
        print( x.element_size() * x.nelement())
        x = func_features[7](x) 
        print( x.element_size() * x.nelement())
        x = func_features[8](x) 
        print( x.element_size() * x.nelement())
        x = func_features[9](x) 
        print( x.element_size() * x.nelement())
        x = func_features[10](x) 
        print( x.element_size() * x.nelement())
        x = func_features[11](x) 
        print( x.element_size() * x.nelement())
        x = func_features[12](x) 
        print(x.element_size() * x.nelement())
        x = func_features[13](x) 
        print( x.element_size() * x.nelement())
        x = func_features[14](x) 
        print( x.element_size() * x.nelement())
        x = func_features[15](x) 
        print( x.element_size() * x.nelement())
        x = func_features[16](x) 
        print( x.element_size() * x.nelement())
        x = func_features[17](x) 
        print( x.element_size() * x.nelement())
        x = func_features[18](x) 
        print( x.element_size() * x.nelement())
        x = func_features[19](x) 
        print( x.element_size() * x.nelement())
        x = func_features[20](x) 
        print( x.element_size() * x.nelement())
        x = func_features[21](x) 
        print( x.element_size() * x.nelement())
        x = func_features[22](x) 
        print( x.element_size() * x.nelement())
        x = func_features[23](x) 
        print( x.element_size() * x.nelement())
        x = func_features[24](x) 
        print( x.element_size() * x.nelement())
        x = func_features[25](x) 
        print( x.element_size() * x.nelement())
        x = func_features[26](x) 
        print( x.element_size() * x.nelement())
        x = func_features[27](x) 
        print(x.element_size() * x.nelement())
        x = func_features[28](x) 
        print( x.element_size() * x.nelement())
        x = func_features[29](x) 
        print( x.element_size() * x.nelement())
        x = func_features[30](x) 
        print( x.element_size() * x.nelement())

        x = x.view(x.size()[0], -1)
        
        print( x.element_size() * x.nelement())

        func_classifier = list(self.classifier.children())
            
        x = func_classifier[0](x) 
        print( x.element_size() * x.nelement())
        x = func_classifier[1](x) 
        print( x.element_size() * x.nelement())
        x = func_classifier[2](x) 
        print( x.element_size() * x.nelement())
        x = func_classifier[3](x) 
        print( x.element_size() * x.nelement())
        x = func_classifier[4](x) 
        print( x.element_size() * x.nelement())
        x = func_classifier[5](x) 
        print( x.element_size() * x.nelement())
        x = func_classifier[6](x) 
        print( x.element_size() * x.nelement())
        
        return x

    def forward_time(self,x):
    #def forward(self, x):

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
    
    return nn.Sequential(*layers)

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
