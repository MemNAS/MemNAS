import torch.nn as nn
import torch
#network feature: [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 1]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 1]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 1]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 1]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 1]]]]
class net216(nn.Module):
    def __init__(self,w):
        super(net216,self).__init__()
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
        self.sml=nn.Linear(3*4*w*4*4,10)
        self.conv1=nn.Sequential(nn.Conv2d(3,w,3,1,1),)
        self.block1_U1_l1=nn.Sequential(Fac_conv(3,w,7,1,3),)
        self.block1_U1_l2=nn.Sequential(depthwise_separable_conv(3,w,3,1,1),)

        self.block1_U2_l1=nn.Sequential(Fac_conv(w,w,7,1,3),)
        self.block1_U2_l2=nn.Sequential(depthwise_separable_conv(w,w,3,1,1),)

        self.block2_U1_l1=nn.Sequential(Fac_conv(3*w,w,7,2,3),)
        self.block2_U1_l2=nn.Sequential(depthwise_separable_conv(3*w,w,3,2,1),)

        self.block2_U2_l1=nn.Sequential(Fac_conv(w,w,7,1,3),)
        self.block2_U2_l2=nn.Sequential(depthwise_separable_conv(w,w,3,1,1),)

        self.block3_U1_l1=nn.Sequential(Fac_conv(3*w,2*w,7,1,3),)
        self.block3_U1_l2=nn.Sequential(depthwise_separable_conv(3*w,2*w,3,1,1),)

        self.block3_U2_l1=nn.Sequential(Fac_conv(2*w,2*w,7,1,3),)
        self.block3_U2_l2=nn.Sequential(depthwise_separable_conv(2*w,2*w,3,1,1),)

        self.block4_U1_l1=nn.Sequential(Fac_conv(3*2*w,2*w,7,2,3),)
        self.block4_U1_l2=nn.Sequential(depthwise_separable_conv(3*2*w,2*w,3,2,1),)

        self.block4_U2_l1=nn.Sequential(Fac_conv(2*w,2*w,7,1,3),)
        self.block4_U2_l2=nn.Sequential(depthwise_separable_conv(2*w,2*w,3,1,1),)

        self.block5_U1_l1=nn.Sequential(Fac_conv(3*2*w,4*w,7,1,3),)
        self.block5_U1_l2=nn.Sequential(depthwise_separable_conv(3*2*w,4*w,3,1,1),)

        self.block5_U2_l1=nn.Sequential(Fac_conv(4*w,4*w,7,1,3),)
        self.block5_U2_l2=nn.Sequential(depthwise_separable_conv(4*w,4*w,3,1,1),)

        self.lmx=nn.Sequential(nn.MaxPool2d(2))
    def forward(self, x):
        U1_l1=self.block1_U1_l1(x)
        U1_l2=self.block1_U1_l2(x)
        U1=U1_l1+U1_l2
        U2_l1=self.block1_U2_l1(U1)
        U2_l2=self.block1_U2_l2(U1)
        U2=torch.cat((U2_l1,U2_l2),1)
        b1=torch.cat((U1,U2),1)
        U1_l1=self.block2_U1_l1(b1)
        U1_l2=self.block2_U1_l2(b1)
        U1=U1_l1+U1_l2
        U2_l1=self.block2_U2_l1(U1)
        U2_l2=self.block2_U2_l2(U1)
        U2=torch.cat((U2_l1,U2_l2),1)
        b2=torch.cat((U1,U2),1)
        U1_l1=self.block3_U1_l1(b2)
        U1_l2=self.block3_U1_l2(b2)
        U1=U1_l1+U1_l2
        U2_l1=self.block3_U2_l1(U1)
        U2_l2=self.block3_U2_l2(U1)
        U2=torch.cat((U2_l1,U2_l2),1)
        b3=torch.cat((U1,U2),1)
        U1_l1=self.block4_U1_l1(b3)
        U1_l2=self.block4_U1_l2(b3)
        U1=U1_l1+U1_l2
        U2_l1=self.block4_U2_l1(U1)
        U2_l2=self.block4_U2_l2(U1)
        U2=torch.cat((U2_l1,U2_l2),1)
        b4=torch.cat((U1,U2),1)
        U1_l1=self.block5_U1_l1(b4)
        U1_l2=self.block5_U1_l2(b4)
        U1=U1_l1+U1_l2
        U2_l1=self.block5_U2_l1(U1)
        U2_l2=self.block5_U2_l2(U1)
        U2=torch.cat((U2_l1,U2_l2),1)
        b5=torch.cat((U1,U2),1)

        last=self.lmx(b5)
        x=last.view(-1,3*4*self.channel*4*4)
        x=self.sml(x)
        return x