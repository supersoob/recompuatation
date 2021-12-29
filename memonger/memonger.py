from math import sqrt, log
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
#from torch.utils.checkpoint import checkpoint
from .checkpoint import checkpoint


def reforwad_momentum_fix(origin_momentum):
    return (1 - sqrt(1 - origin_momentum))

#def reforwad_momentum_fix_every(origin_momentum):
#    return (1 - ((1 - origin_momentum)/2))

#def reforwad_momentum_fix_every_before_mod(origin_momentum):
#    return (1 - ((1 - origin_momentum)/3))


class SublinearSequential(nn.Sequential):
    def __init__(self, *args):
        super(SublinearSequential, self).__init__(*args)
        self.reforward = False
        self.momentum_dict = {}
        self.set_reforward(True)

    def set_reforward(self, enabled=True):
        if not self.reforward and enabled:
            print("Rescale BN Momemtum for re-forwarding purpose")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    self.momentum_dict[n] = m.momentum
                    m.momentum = reforwad_momentum_fix(self.momentum_dict[n])
        if self.reforward and not enabled:
            print("Re-store BN Momemtum")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    m.momentum = self.momentum_dict[n]
        self.reforward = enabled

    def forward(self, input):
        
        if self.reforward:
            return self.sublinear_forward(input)
        else:
            return self.normal_forward(input)

    def normal_forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def sublinear_forward(self, input):
        def run_function(start, end, functions):
            def forward(*inputs):
                input = inputs[0]
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input

            return forward

        functions = list(self.children())
        segments = int(sqrt(len(functions)))
        #print("segments : ", segments)
        
        segment_size = len(functions) // segments
        
        # the last chunk has to be non-volatile
        end = -1
        if not isinstance(input, tuple):
            inputs = (input,)
        for start in range(0, segment_size * (segments - 1), segment_size):
            end = start + segment_size - 1
            inputs = checkpoint(run_function(start, end, functions), *inputs)
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
        # output = run_function(end + 1, len(functions) - 1, functions)(*inputs)
        output = checkpoint(run_function(end + 1, len(functions) - 1, functions), *inputs)
        return output



class SublinearSequential_every(nn.Sequential):
    def __init__(self, *args):
        super(SublinearSequential_every, self).__init__(*args)
        self.reforward = False
        self.momentum_dict = {}
        self.set_reforward(True)

    def set_reforward(self, enabled=True):
        if not self.reforward and enabled:
            print("Rescale BN Momemtum for re-forwarding purpose")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    self.momentum_dict[n] = m.momentum
                    m.momentum = reforwad_momentum_fix(self.momentum_dict[n])
        if self.reforward and not enabled:
            print("Re-store BN Momemtum")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    m.momentum = self.momentum_dict[n]
        self.reforward = enabled

    def forward(self, input):
        #return self.normal_forward(input)
        if self.reforward:
            return self.sublinear_forward(input)
        else:
            return self.normal_forward(input)

    def normal_forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def sublinear_forward(self, input):
        def run_function(start, end, functions):
            def forward(*inputs):
                input = inputs[0]
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input

            return forward

        functions = list(self.children())

        segments = round(len(functions)/2) 
        segment_size = 2
        
        #print("func : ", functions)        
        print("len(f) : ", len(functions))
        print("# of seg : ", segments)
        print("size : ", segment_size)

        # the last chunk has to be non-volatile
        end = -1
        if not isinstance(input, tuple):
            inputs = (input,)

#assuming that segment_size <=3
#if wants segment_size = 1, output -> checkpoint(run_function(,,,))
        if segments == 1:
            start = 0
            end = start + segment_size - 2
            inputs = run_function(start, end, functions)(*inputs)
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
            output = checkpoint(run_function(end+1,end+1,functions), *inputs)
            return output

        else:
            for start in range(0, segment_size * (segments -1), segment_size):
                end = start + segment_size - 2
                #print("end : ", end)
                inputs = run_function(start, end, functions)(*inputs)
                if not isinstance(inputs, tuple):
                    inputs = (inputs,)
                inputs = checkpoint(run_function(end + 1, end + 1, functions), *inputs)
                if not isinstance(inputs, tuple):
                    inputs = (inputs,)
                #print("end : ", end)
            #output = checkpoint(run_function(end + 2, len(functions) - 1, functions),*inputs)
            output = run_function(end + 2, len(functions) - 1, functions)(*inputs)
            return output
     
        """
        for start in range(0, segment_size * (segments - 1), segment_size):
            end = start + segment_size - 1
            inputs = checkpoint(run_function(start, end, functions), *inputs)
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
        #output = run_function(end + 1, len(functions) - 1, functions)(*inputs)
        output = checkpoint(run_function(end + 1, len(functions) - 1, functions), *inputs)
        return output
        """


class SublinearSequential_every_before_mod(nn.Sequential):
    def __init__(self, *args):
        super(SublinearSequential_every_before_mod, self).__init__(*args)
        self.reforward = False
        self.momentum_dict = {}
        self.set_reforward(True)

    def set_reforward(self, enabled=True):
        if not self.reforward and enabled:
            print("Rescale BN Momemtum for re-forwarding purpose")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    self.momentum_dict[n] = m.momentum
                    m.momentum = reforwad_momentum_fix(self.momentum_dict[n])
        if self.reforward and not enabled:
            print("Re-store BN Momemtum")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    m.momentum = self.momentum_dict[n]
        self.reforward = enabled

    def forward(self, input):
        if self.reforward:
            return self.sublinear_forward(input)
        else:
            return self.normal_forward(input)

    def normal_forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def sublinear_forward(self, input):
        def run_function(start, end, functions):
            def forward(*inputs):
                input = inputs[0]
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input

            return forward

        functions = list(self.children())
        print("len(functions) : ", len(functions))

        segments = len(functions) // 1
        segment_size = 1

        # the last chunk has to be non-volatile
        end = -1
        if not isinstance(input, tuple):
            inputs = (input,)
        for start in range(0, segment_size * (segments - 1), segment_size):
            end = start + segment_size - 1
            inputs = checkpoint(run_function(start, end, functions), *inputs)
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
        #output = run_function(end + 1, len(functions) - 1, functions)(*inputs)
        output = checkpoint(run_function(end + 1, len(functions) - 1, functions), *inputs)
        return output


class SublinearSequential_no_ckpt(nn.Sequential):
    def __init__(self, *args):
        super(SublinearSequential_no_ckpt, self).__init__(*args)
        self.reforward = False
        self.momentum_dict = {}
        self.set_reforward(True)

    def set_reforward(self, enabled=True):
        if not self.reforward and enabled:
            print("Rescale BN Momemtum for re-forwarding purpose")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    self.momentum_dict[n] = m.momentum
                    m.momentum = reforwad_momentum_fix(self.momentum_dict[n])
        if self.reforward and not enabled:
            print("Re-store BN Momemtum")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    m.momentum = self.momentum_dict[n]
        self.reforward = enabled

    def forward(self, input):
        if self.reforward:
            return self.sublinear_forward(input)
        else:
            return self.normal_forward(input)

    def normal_forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def sublinear_forward(self, input):
        def run_function(start, end, functions):
            def forward(*inputs):
                print(start, end, inputs)
                input = inputs[0]
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input

            return forward

        functions = list(self.children())
        print("len(functions) : ", len(functions))

        segments = 1
        segment_size = len(functions) // segments
           

        # the last chunk has to be non-volatile
        end = -1

     
        if not isinstance(input, tuple):
            inputs = (input,)
       
        #print(inputs)
 
        
        
        for start in range(0, segment_size * (segments - 1), segment_size):
            end = start + segment_size - 1
            inputs = checkpoint(run_function(start, end, functions), *inputs)
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
        #output = run_function(end + 1, len(functions) - 1, functions)(*inputs)
        output = checkpoint(run_function(end + 1, len(functions) - 1, functions), *inputs)
        return output
        

