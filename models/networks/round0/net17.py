import torch.nn as nn
import torch
#network feature: [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]]]]
class net17(nn.Module):
    def __init__(self,w):
        super(net17,self).__init__()
        def ConvFactory(inp,oup,kernel,stride,pad):
            return nn.Sequential(
                nn.Conv2d(inp,oup,kernel,stride,pad),
                nn.BatchNorm2d(oup),
                nn.ReLU()
            )
        def depthwise_separable_conv(inp, oup, kernel,stride,pad):
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel, stride=stride,padding=pad, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, kernel_size=1),
                nn.BatchNorm2d(oup),
                nn.ReLU()
            )
        def Dilated_conv(inp, oup, kernel,stride,pad):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=kernel,stride=stride, padding=pad, dilation=2),
                nn.BatchNorm2d(oup),
                nn.ReLU()
            )
        def Fac_conv(inp, oup, kernel,stride,pad):
            return nn.Sequential(
                ConvFactory(inp,oup,1,stride,0),
                ConvFactory(oup, oup, (1,kernel),1,(0,pad)),
                ConvFactory(oup, oup, (kernel,1),1,(pad,0))
            )
        def MaxPool(inp, oup, kernel,stride,padding):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1),
                nn.BatchNorm2d(oup),
                nn.MaxPool2d(3,stride=stride, padding=1)
            )
        def AvgPool(inp, oup, kernel,stride,padding):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1),
                nn.BatchNorm2d(oup),
                nn.AvgPool2d(3,stride=stride, padding=1)
            )
        self.channel=w
        self.sml=nn.Linear(1*4*w*4*4,10)
        self.conv1=nn.Sequential(nn.Conv2d(3,w,3,1,1),)
        self.block1_U1_l1=nn.Sequential(MaxPool(3,w,3, stride=1, padding=1))
        self.block1_U1_l2=nn.Sequential(MaxPool(3,w,3, stride=1, padding=1))
        self.block2_U1_l1=nn.Sequential(MaxPool(1*w,w,3, stride=2, padding=1))
        self.block2_U1_l2=nn.Sequential(MaxPool(1*w,w,3, stride=2, padding=1))
        self.block3_U1_l1=nn.Sequential(MaxPool(1*w,2*w,3, stride=1, padding=1))
        self.block3_U1_l2=nn.Sequential(MaxPool(1*w,2*w,3, stride=1, padding=1))
        self.block4_U1_l1=nn.Sequential(MaxPool(1*2*w,2*w,3, stride=2, padding=1))
        self.block4_U1_l2=nn.Sequential(MaxPool(1*2*w,2*w,3, stride=2, padding=1))
        self.block5_U1_l1=nn.Sequential(MaxPool(1*2*w,4*w,3, stride=1, padding=1))
        self.block5_U1_l2=nn.Sequential(MaxPool(1*2*w,4*w,3, stride=1, padding=1))
        self.lmx=nn.Sequential(nn.MaxPool2d(2))
    def forward(self, x):
        U1_l1=self.block1_U1_l1(x)
        U1_l2=self.block1_U1_l2(x)
        U1=U1_l1+U1_l2
        b1=U1
        U1_l1=self.block2_U1_l1(b1)
        U1_l2=self.block2_U1_l2(b1)
        U1=U1_l1+U1_l2
        b2=U1
        U1_l1=self.block3_U1_l1(b2)
        U1_l2=self.block3_U1_l2(b2)
        U1=U1_l1+U1_l2
        b3=U1
        U1_l1=self.block4_U1_l1(b3)
        U1_l2=self.block4_U1_l2(b3)
        U1=U1_l1+U1_l2
        b4=U1
        U1_l1=self.block5_U1_l1(b4)
        U1_l2=self.block5_U1_l2(b4)
        U1=U1_l1+U1_l2
        b5=U1

        last=self.lmx(b5)
        x=last.view(-1,1*4*self.channel*4*4)
        x=self.sml(x)
        return x