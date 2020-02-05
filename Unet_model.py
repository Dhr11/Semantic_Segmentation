import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from layer_blocks import conv_down,conv_up

class Unet(nn.Module):
    def __init__(self,levels=5,filter_offset=4,channels=3,n_class=21,do_batchnorm=True,deconv=True):
        super(Unet,self).__init__()
        self.levels = levels
        self.channels = channels
        self.n_class = n_class
        self.deconv = deconv
        self.do_batchnorm = do_batchnorm
        self.filter_offset = filter_offset
        self.filters = [self.channels] + [2**(i+self.filter_offset) for i in range(self.levels)]
        print(self.filters)
        
        self.down_layers=nn.ModuleList()
        for i in range(levels-1):
            self.down_layers.append(conv_down(self.filters[i],self.filters[i+1],self.do_batchnorm,2))
        
        self.center_layer = conv_down(self.filters[-2],self.filters[-1],self.do_batchnorm,None)

        self.up_layers = nn.ModuleList()
        for i in range(levels,1,-1):
            tmp = 0 if i == levels else self.filters[i]
            self.up_layers.append(conv_up(tmp+self.filters[i],self.filters[i-1],self.deconv))

        self.fin = nn.Conv2d(2*self.filters[1],self.n_class,1)
        self.maxp = nn.MaxPool2d(kernel_size=2)
    def forward(self,inp):
        out = inp
        down = []
        for layer in self.down_layers:
            out = layer(out)
            #print(out.shape)
            down.append(out)
            out = self.maxp(out)
        #print(out.shape)    
        out = self.center_layer(out)
        #print("center",out.shape)
        index = -1
        for layer in self.up_layers:
            #print("up before",out.shape)    
            out = layer(out)
            #print(out.shape)    
            offset = out.size()[2] - down[index].size()[2]
            offset2 = out.size()[3] - down[index].size()[3]
            padding = 2 * [offset // 2, offset2 // 2]
            out2 = F.pad(down[index], padding)
            out = torch.cat([out,out2],dim=1)
            index-=1
        return self.fin(out)    
