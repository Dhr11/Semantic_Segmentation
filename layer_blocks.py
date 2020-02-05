import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_down(nn.Module):
    def __init__(self,size_in,size_out,do_BatchNorm=True,maxpool_size=2):
        super(conv_down,self).__init__()
        self.maxpool_size = maxpool_size
        if do_BatchNorm:
            self.conv1 =nn.Sequential(nn.Conv2d(size_in,size_out,3,1,1),nn.BatchNorm2d(size_out),nn.ReLU())
            self.conv2 =nn.Sequential(nn.Conv2d(size_out,size_out,3,1,1),nn.BatchNorm2d(size_out),nn.ReLU())
            
        else:
            self.conv1 =nn.Sequential(nn.Conv2d(size_in,size_out,3,1,1),nn.ReLU())
            self.conv2 =nn.Sequential(nn.Conv2d(size_out,size_out,3,1,1),nn.ReLU())
            
        #self.maxp = nn.MaxPool2d(kernel_size=maxpool_size)
    def forward(self,inp):
        out = self.conv1(inp)
        out = self.conv2(out)
        return out
        """
        if self.maxpool_size is None: 
            return out
        else :
            return self.maxp(out)    
        """    

class conv_up(nn.Module):
    def __init__(self,size_in,size_out,deconv=True):
        super(conv_up,self).__init__()
        self.deconv = deconv
        self.conv = conv_down(size_in,size_out,False,None)
        self.upconv = nn.ConvTranspose2d(size_out, size_out, kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,inp1):
        out = self.conv(inp1)
        if self.deconv:
            return self.upconv(out)
        else:
            return self.upsample(out)        