'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

#from .memonger import SublinearSequential

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    
     
    def forward(self, x):
        """
        a = time.perf_counter()
        out = self.conv1(x)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'conv1 time : {b-a}')
        
        a = time.perf_counter()
        out = self.bn1(out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'bn1 time : {b-a}')

        a = time.perf_counter()
        out = F.relu(out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'relu1 time : {b-a}')

        a = time.perf_counter()
        out = self.conv2(out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'conv2 time : {b-a}')
        
        a = time.perf_counter()
        out = self.bn2(out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'bn2 time : {b-a}')
        
        a = time.perf_counter()
        out = F.relu(out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'relu2 time : {b-a}')
        
        a = time.perf_counter()
        out = self.conv3(out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'conv3 time : {b-a}')
        
        a = time.perf_counter()
        out = self.bn3(out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'bn3 time : {b-a}')
        
        a = time.perf_counter()
        out += self.shortcut(x)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'shortcut time : {b-a}')

        a = time.perf_counter()
        out = F.relu(out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'relu time : {b-a}')
        """

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


   
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

    #def forward(self, x):
    def forward_to_profile(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        functions = list(self.layer1.children())
        a = time.perf_counter()
        out = functions[0](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 1 time : {b-a}')

        a = time.perf_counter()
        out = functions[1](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 2 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[2](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 3 time : {b-a}')
       
        
        functions = list(self.layer2.children())
        a = time.perf_counter()
        out = functions[0](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 4 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[1](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 5 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[2](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 6 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[3](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 7 time : {b-a}')
       


        functions = list(self.layer3.children())
        
        a = time.perf_counter()
        out = functions[0](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 8 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[1](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 9 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[2](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 10 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[3](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 11 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[4](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 12 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[5](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 13 time : {b-a}')
        
        """
        a = time.perf_counter()
        out = functions[6](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 14 time : {b-a}')
        
         
        a = time.perf_counter()
        out = functions[7](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 15 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[8](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 16 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[9](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 17 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[10](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 18 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[11](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 19 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[12](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 20 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[13](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 21 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[14](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 22 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[15](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 23 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[16](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 24 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[17](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 25 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[18](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 26 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[19](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 27 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[20](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 28 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[21](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 29 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[22](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print (f'residual 30 time : {b-a}')
        """
         

        functions = list(self.layer4.children())
        a = time.perf_counter()
        out = functions[0](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print(f'residaul 14 time : {b-a}')
        #print (f'residual 31 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[1](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print(f'residaul 15 time : {b-a}')
        #print (f'residual 32 time : {b-a}')
        
        a = time.perf_counter()
        out = functions[2](out)
        torch.cuda.synchronize()
        b = time.perf_counter()
        print(f'residual 16 time : {b-a}')
        #print (f'residual 33 time : {b-a}')


        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def forward_to_profile_inside_res(self, x):
        
        
        out = F.relu(self.bn1(self.conv1(x)))
        functions = list(self.layer1.children())      

        out = functions[0](out)
        print("residual 1 done")
        out = functions[1](out)
        print("residual 2 done")
        out = functions[2](out)
        print("residual 3 done")

        functions = list(self.layer2.children())
        
        out = functions[0](out)
        print("residual 4 done")
        out = functions[1](out)
        print("residual 5 done")
        out = functions[2](out)
        print("residual 6 done")
        out = functions[3](out)
        print("residual 7 done")

        functions = list(self.layer3.children())
        out = functions[0](out)
        print("residual 8 done")
       
        out = functions[1](out)
        print("residual 9 done")
       
        out = functions[2](out)
        print("residual 10 done")
        
        out = functions[3](out)
        print("residual 11 done")
       
        out = functions[4](out)
        print("residual 12 done")
       
        out = functions[5](out)
        print("residual 13 done")
       
        out = functions[6](out)
        print("residual 14 done")
       
        out = functions[7](out)
        print("residual 15 done")
        
        out = functions[8](out)
        print("residual 16 done")
       
        out = functions[9](out)
        print("residual 17 done")
        
        out = functions[10](out)
        print("residual 18 done")
        
        out = functions[11](out)
        print("residual 19 done")
        
        out = functions[12](out)
        print("residual 20 done")
        
        out = functions[13](out)
        print("residual 21 done")
        
        out = functions[14](out)
        print("residual 22 done")
        
        out = functions[15](out)
        print("residual 23 done")
        
        out = functions[16](out)
        print("residual 24 done")
        
        out = functions[17](out)
        print("residual 25 done")
        
        out = functions[18](out)
        print("residual 26 done")
        
        out = functions[19](out)
        print("residual 27 done")
        
        out = functions[20](out)
        print("residual 28 done")

        out = functions[21](out)
        print("residual 29 done")
        
        out = functions[22](out)
        print("residual 30 done")

        functions = list(self.layer4.children())
        out = functions[0](out)
        print("residual 31 done")

        out = functions[1](out)
        print("residual 32 done")

        out = functions[2](out)
        print("residual 33 done")

        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
        
        

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
