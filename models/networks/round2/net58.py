import torch.nn as nn
import torch
#network feature: [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [1, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [1, 0]]]]
class net58(nn.Module):
    def __init__(self,w):
        super(net58,self).__init__()
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
        self.channel1=w
        self.channel2=w
        self.channel3=w
        self.channel4=w
        self.channel5=w
        self.sml=nn.Linear(3*self.channel5*4*4,10)
        self.conv1=nn.Sequential(nn.Conv2d(3,w,3,1,1),)
        self.block1_U1_l1=nn.Sequential(Fac_conv(3,self.channel1,7,1,3),)
        self.block1_U1_l2=nn.Sequential(depthwise_separable_conv(3,self.channel1,3,1,1),)

        self.block1_U2_l1=nn.Sequential(Fac_conv(self.channel1,self.channel1,7,1,3),)
        self.block1_U2_l2=nn.Sequential(MaxPool(3,self.channel1,3, stride=1, padding=1))

        self.block1_U3_l1=nn.Sequential(AvgPool(self.channel1,self.channel1,3, stride=1, padding=1))
        self.block1_U3_l2=nn.Sequential(ConvFactory(3,self.channel1,3,1,1),)

        self.block2_U1_l1=nn.Sequential(Fac_conv(3*self.channel1,self.channel2,7,2,3),)
        self.block2_U1_l2=nn.Sequential(depthwise_separable_conv(3*self.channel1,self.channel2,3,2,1),)

        self.block2_U2_l1=nn.Sequential(Fac_conv(self.channel2,self.channel2,7,1,3),)
        self.block2_U2_l2=nn.Sequential(MaxPool(3*self.channel1,self.channel2,3, stride=2, padding=1))

        self.block2_U3_l1=nn.Sequential(AvgPool(self.channel2,self.channel2,3, stride=1, padding=1))
        self.block2_U3_l2=nn.Sequential(ConvFactory(3*self.channel1,self.channel2,3,2,1),)

        self.block3_U1_l1=nn.Sequential(Fac_conv(3*self.channel2,self.channel3,7,1,3),)
        self.block3_U1_l2=nn.Sequential(depthwise_separable_conv(3*self.channel2,self.channel3,3,1,1),)

        self.block3_U2_l1=nn.Sequential(Fac_conv(self.channel3,self.channel3,7,1,3),)
        self.block3_U2_l2=nn.Sequential(MaxPool(3*self.channel2,self.channel3,3, stride=1, padding=1))

        self.block3_U3_l1=nn.Sequential(AvgPool(self.channel3,self.channel3,3, stride=1, padding=1))
        self.block3_U3_l2=nn.Sequential(ConvFactory(3*self.channel2,self.channel3,3,1,1),)

        self.block4_U1_l1=nn.Sequential(Fac_conv(3*self.channel3,self.channel4,7,2,3),)
        self.block4_U1_l2=nn.Sequential(depthwise_separable_conv(3*self.channel3,self.channel4,3,2,1),)

        self.block4_U2_l1=nn.Sequential(Fac_conv(self.channel4,self.channel4,7,1,3),)
        self.block4_U2_l2=nn.Sequential(MaxPool(3*self.channel3,self.channel4,3, stride=2, padding=1))

        self.block4_U3_l1=nn.Sequential(AvgPool(self.channel4,self.channel4,3, stride=1, padding=1))
        self.block4_U3_l2=nn.Sequential(ConvFactory(3*self.channel3,self.channel4,3,2,1),)

        self.block5_U1_l1=nn.Sequential(Fac_conv(3*self.channel4,self.channel5,7,1,3),)
        self.block5_U1_l2=nn.Sequential(depthwise_separable_conv(3*self.channel4,self.channel5,3,1,1),)

        self.block5_U2_l1=nn.Sequential(Fac_conv(self.channel5,self.channel5,7,1,3),)
        self.block5_U2_l2=nn.Sequential(MaxPool(3*self.channel4,self.channel5,3, stride=1, padding=1))

        self.block5_U3_l1=nn.Sequential(AvgPool(self.channel5,self.channel5,3, stride=1, padding=1))
        self.block5_U3_l2=nn.Sequential(ConvFactory(3*self.channel4,self.channel5,3,1,1),)

        self.lmx=nn.Sequential(nn.MaxPool2d(2))
    def forward(self, x):
        U1_l1=self.block1_U1_l1(x)
        U1_l2=self.block1_U1_l2(x)
        U1=U1_l1+U1_l2
        U2_l1=self.block1_U2_l1(U1)
        U2_l2=self.block1_U2_l2(x)
        U2=U2_l1+U2_l2
        U3_l1=self.block1_U3_l1(U1)
        U3_l2=self.block1_U3_l2(x)
        U3=U3_l1+U3_l2
        b1=torch.cat((U1,U2,U3),1)
        U1_l1=self.block2_U1_l1(b1)
        U1_l2=self.block2_U1_l2(b1)
        U1=U1_l1+U1_l2
        U2_l1=self.block2_U2_l1(U1)
        U2_l2=self.block2_U2_l2(b1)
        U2=U2_l1+U2_l2
        U3_l1=self.block2_U3_l1(U1)
        U3_l2=self.block2_U3_l2(b1)
        U3=U3_l1+U3_l2
        b2=torch.cat((U1,U2,U3),1)
        U1_l1=self.block3_U1_l1(b2)
        U1_l2=self.block3_U1_l2(b2)
        U1=U1_l1+U1_l2
        U2_l1=self.block3_U2_l1(U1)
        U2_l2=self.block3_U2_l2(b2)
        U2=U2_l1+U2_l2
        U3_l1=self.block3_U3_l1(U1)
        U3_l2=self.block3_U3_l2(b2)
        U3=U3_l1+U3_l2
        b3=torch.cat((U1,U2,U3),1)
        U1_l1=self.block4_U1_l1(b3)
        U1_l2=self.block4_U1_l2(b3)
        U1=U1_l1+U1_l2
        U2_l1=self.block4_U2_l1(U1)
        U2_l2=self.block4_U2_l2(b3)
        U2=U2_l1+U2_l2
        U3_l1=self.block4_U3_l1(U1)
        U3_l2=self.block4_U3_l2(b3)
        U3=U3_l1+U3_l2
        b4=torch.cat((U1,U2,U3),1)
        U1_l1=self.block5_U1_l1(b4)
        U1_l2=self.block5_U1_l2(b4)
        U1=U1_l1+U1_l2
        U2_l1=self.block5_U2_l1(U1)
        U2_l2=self.block5_U2_l2(b4)
        U2=U2_l1+U2_l2
        U3_l1=self.block5_U3_l1(U1)
        U3_l2=self.block5_U3_l2(b4)
        U3=U3_l1+U3_l2
        b5=torch.cat((U1,U2,U3),1)

        last=self.lmx(b5)
        x=last.view(-1,3*self.channel5*4*4)
        x=self.sml(x)
        return x