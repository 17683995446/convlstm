"""residual attention network in pytorch



[1] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang

    Residual Attention Network for Image Classification
    https://arxiv.org/abs/1704.06904
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#"""The Attention Module is built by pre-activation Residual Unit [11] with the
#number of channels in each stage is the same as ResNet [10]."""



class SA_Attn_Mem(nn.Module):
    # SAM 自注意力模块
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.layer_q = nn.Conv3d(input_dim, hidden_dim, (1,1,1),(1,1,1))
        self.layer_k = nn.Conv3d(input_dim, hidden_dim, (1,1,1),(1,1,1))

        self.layer_v = nn.Conv3d(input_dim, hidden_dim, (1,1,1),(1,1,1))

        self.layer_z = nn.Conv3d(input_dim * 2, input_dim * 2, (1,1,1),(1,1,1))
        self.layer_m = nn.Conv3d(input_dim * 3, input_dim * 3, (1,1,1),(1,1,1))

    def forward(self, h):
        batch_size, channels, T, H, W = h.shape
        # **********************  feature aggregation ******************** #

        # Use 1x1 convolution for Q,K,V Generation
        K_h = self.layer_k(h)
        K_h = K_h.view(batch_size, self.hidden_dim,T * H * W)

        Q_h = self.layer_q(h)
        Q_h = Q_h.view(batch_size, self.hidden_dim,T * H * W)
        Q_h = Q_h.transpose(1, 2)

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim,T * H * W)


        # **********************  hidden h attention ******************** #
        # [batch_size,H*W,H*W]
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)

        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim,T, H, W)
        # **********************  memory m attention ******************** #
        # [batch_size,H*W,H*W]
        out = Z_h

        return out




class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x




class DA(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(DA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c,  _ ,_,_ = x.size()
        x =  self.avg_pool(x).view(b,c)
        x = self.fc(x).view(b,c,1,1,1)
        return x




class SA_Attn_Mem(nn.Module):
    # SAM 自注意力模块
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.layer_q = nn.Conv3d(input_dim, hidden_dim, (1,1,1),(1,1,1))
        self.layer_k = nn.Conv3d(input_dim, hidden_dim, (1,1,1),(1,1,1))

        self.layer_v = nn.Conv3d(input_dim, hidden_dim, (1,1,1),(1,1,1))

        self.layer_z = nn.Conv3d(input_dim * 2, input_dim * 2, (1,1,1),(1,1,1))
        self.layer_m = nn.Conv3d(input_dim * 3, input_dim * 3, (1,1,1),(1,1,1))

    def forward(self, h):
        batch_size, channels, T, H, W = h.shape
        # **********************  feature aggregation ******************** #
        # print(batch_size, channels, T, H, W)
        # Use 1x1 convolution for Q,K,V Generation
        K_h = self.layer_k(h)
        K_h = K_h.view(batch_size, self.hidden_dim,T * H * W)

        Q_h = self.layer_q(h)
        Q_h = Q_h.view(batch_size, self.hidden_dim,T * H * W)
        Q_h = Q_h.transpose(1, 2)

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim,T * H * W)


        # **********************  hidden h attention ******************** #
        # [batch_size,H*W,H*W]
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)

        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim,T, H, W)
        # **********************  memory m attention ******************** #
        # [batch_size,H*W,H*W]
        out = Z_h

        return out






class PreActResidualUnit(nn.Module):
    """PreAct Residual Unit
    Args:
        in_channels: residual unit input channel number
        out_channels: residual unit output channel numebr
        stride: stride of residual unit when stride = 2, downsample the featuremap
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()


        self.radix=8
        self.cardinality = 8
        group = 1
        if out_channels>=64:
            bottleneck_channels = int(out_channels / 2)
        else:
            bottleneck_channels = out_channels
            self.radix=1
            self.cardinality = 1
        group = bottleneck_channels
        self.da_layer = DA(in_channels=out_channels,out_channels=out_channels)



        self.relu = nn.ReLU(inplace=True)
        inter_channels = max(out_channels//4, 64)
        radix_in_channels = max(in_channels//self.radix, 1)
        self.fc1 = nn.Conv3d(out_channels, inter_channels, (1,1,1),(1,1,1), groups=self.cardinality)
        self.fc2 = nn.Conv3d(inter_channels, out_channels*self.radix, (1,1,1),(1,1,1), groups=self.cardinality)
        self.fc3 = nn.Conv3d(radix_in_channels, inter_channels, (1,1,1),(1,1,1), groups=self.cardinality)
        self.fc4 = nn.Conv3d(inter_channels, out_channels, (1,1,1),(1,1,1), groups=self.cardinality)
        self.rsoftmax = rSoftMax(self.radix, self.cardinality)



        self.residual_function = nn.Sequential(
            #1x1x1 conv
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, bottleneck_channels, (1,1,1), (1,1,1)),

            #1x3x3 conv
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, bottleneck_channels, (3,11,11),(1,1,1),(1,15,15),dilation=(1,3,3),groups=group),

            #3x1x1 conv
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, bottleneck_channels, (5,3,3),(1,1,1),(2,1,1),dilation=(1,1,1),groups=group),

            #1x3x3 conv
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, bottleneck_channels, (3,11,11),(1,1,1),(1,15,15),dilation=(1,3,3),groups=group),

            #3x1x1 conv
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, bottleneck_channels, (5,3,3),(1,1,1),(2,1,1),dilation=(1,1,1),groups=group),#dilation=(2,1,1)

            #1x3x3 conv
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, bottleneck_channels, (3,11,11),(1,1,1),(1,15,15),dilation=(1,3,3),groups=group),

            #3x1x1 conv
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, bottleneck_channels, (5,3,3),(1,1,1),(2,1,1),dilation=(1,1,1),groups=group),#dilation=(2,1,1)


            #1x1 conv
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, out_channels*self.radix, (1,1,1),(1,1,1))
        )

        self.shortcut = nn.Sequential()
        if stride != 2 or (in_channels != out_channels):
            self.shortcut = nn.Conv3d(in_channels, out_channels, (1,1,1), (1,1,1))

        self.sigmoid = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1,1,1),(1,1,1),groups=out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1,1,1),(1,1,1),groups=out_channels),
            nn.Sigmoid()
        )

        self.bn = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1,1,1),(1,1,1),groups=out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1,1,1),(1,1,1),groups=out_channels),
        )



    def forward(self, x):
        # res = self.residual_function(x)
        shortcut = self.shortcut(x)
        # # print("res:",res.cpu().data.numpy().shape)
        # print("shortcut:",shortcut.cpu().data.numpy().shape)

        # x_d =  self.da_layer(shortcut).expand_as(res).contiguous()
        # # print("x_d:",x_d.cpu().data.numpy().shape)
        # x_s = res*x_d
        # x_s = self.sigmoid(x_s)

        # res = x_s * shortcut
        batch,rchannel = x.shape[:2]
        if rchannel>=64:
            # print("x:",x.cpu().data.numpy().shape)
            x = self.residual_function(x)
            # print("x:",x.cpu().data.numpy().shape)
            batch,rchannel = x.shape[:2]
            splited = torch.split(x,rchannel//self.radix,dim=1)
            gap = sum(splited)
            batch,rchannel = gap.shape[:2]
            # print("gap0:",gap.cpu().data.numpy().shape)
            gap = F.adaptive_avg_pool3d(gap, 1)
            # print("gap1:",gap.cpu().data.numpy().shape)
            gap = self.fc1(gap)
            gap = self.relu(gap)
            atten = self.fc2(gap)
            # print("atten0:",atten.cpu().data.numpy().shape)
            atten = self.rsoftmax(atten).view(batch, -1, 1, 1 ,1)
            attens = torch.split(atten, rchannel, dim=1)
            # attens_list = []
            # for att in attens:
            #   att = self.fc3(att)
            #   att = self.relu(att)
            #   att = self.fc4(att)
            #   att = self.rsoftmax(att).view(batch, -1, 1, 1 ,1)
            #   attens_list.append(att)
            # attens = attens_list
            # print("atten:",atten.cpu().data.numpy().shape)
            out = sum([att*split for (att, split) in zip(attens, splited)])
            # print("out:",out.cpu().data.numpy().shape)
            res = out


        else:
            res = self.residual_function(x)
            # print("res:",res.cpu().data.numpy().shape)
            # print("shortcut:",shortcut.cpu().data.numpy().shape)
            x_d = self.da_layer(shortcut).expand_as(res).contiguous()
            # print("x_d:",x_d.cpu().data.numpy().shape)
            x_s = res*x_d
            x_s = self.sigmoid(x_s)
            res = x_s * shortcut
            res = self.bn(res)
        # print("res_before:",res.cpu().data.numpy().shape)

        # print("res:",res.cpu().data.numpy().shape)
        return res + shortcut

class AttentionModule1(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        #"""The hyperparameter p denotes the number of preprocessing Residual
        #Units before splitting into trunk branch and mask branch. t denotes
        #the number of Residual Units in trunk branch. r denotes the number of
        #Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown4 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup4 = self._make_residual(in_channels, out_channels, r)

        self.shortcut_short = PreActResidualUnit(in_channels, out_channels, 1)
        self.shortcut_long = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1,1,1),(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1,1,1),(1,1,1)),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        ###We make the size of the smallest output map in each mask branch 7*7 to be consistent
        #with the smallest trunk output map size.
        ###Thus 3,2,1 max-pooling layers are used in mask branch with input size 56 * 56, 28 * 28, 14 * 14 respectively.
        x = self.pre(x)
        input_size = (x.size(2), x.size(3),x.size(4))

        x_t = self.trunk(x)

        #first downsample out 28
        x_s = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        x_s = self.soft_resdown1(x_s)

        #28 shortcut
        shape1 = (x_s.size(2), x_s.size(3),x_s.size(4))
        shortcut_long = self.shortcut_long(x_s)

        #seccond downsample out 14
        x_s = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        x_s = self.soft_resdown2(x_s)

        #14 shortcut
        shape2 = (x_s.size(2), x_s.size(3),x_s.size(4))
        shortcut_short = self.soft_resdown3(x_s)

        #third downsample out 7
        x_s = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        x_s = self.soft_resdown3(x_s)

        #mid
        x_s = self.soft_resdown4(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape2)
        x_s += shortcut_short

        #second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut_long

        #thrid upsample out 54
        x_s = self.soft_resup4(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class AttentionModule2(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        #"""The hyperparameter p denotes the number of preprocessing Residual
        #Units before splitting into trunk branch and mask branch. t denotes
        #the number of Residual Units in trunk branch. r denotes the number of
        #Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)

        self.shortcut = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3),x.size(4))

        x_t = self.trunk(x)

        #first downsample out 14
        x_s = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        #14 shortcut
        shape1 = (x_s.size(2), x_s.size(3),x_s.size(4))
        shortcut = self.shortcut(x_s)

        #seccond downsample out 7
        x_s = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        x_s = self.soft_resdown2(x_s)

        #mid
        x_s = self.soft_resdown3(x_s)
        x_s = self.soft_resup1(x_s)

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut

        #second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class AttentionModule3(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()

        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)

        self.shortcut = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1,1,1),(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, (1,1,1),(1,1,1)),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

        # self.da_layer = DA(in_channels=in_channels,out_channels=out_channels)
        self.sa_attn_mem = SA_Attn_Mem(input_dim=out_channels,hidden_dim=out_channels)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3),x.size(4))

        x_t = self.trunk(x)

        #first downsample out 14
        x_s = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        # x_s = x
        x_s = self.soft_resdown1(x_s)

        #mid
        temp = x_s
        shape1 = (x_s.size(2), x_s.size(3),x_s.size(4))
        x_s = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        x_s = self.soft_resdown2(x_s)

        # x_s = self.sa_attn_mem(x_s)

        x_s = self.soft_resup1(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += temp

        #first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        # x = x_s
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        return nn.Sequential(*layers)

class Attention(nn.Module):
    """residual attention netowrk
    Args:
        block_num: attention module number for each stage
    """

    def __init__(self, block_num, class_num=100):

        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )


        self.stage1 = self._make_stage(64, 64, block_num[0], AttentionModule1)
        self.stage2 = self._make_stage(64, 64, block_num[1], AttentionModule2)
        self.stage3 = self._make_stage(64, 64, block_num[2], AttentionModule3)
        self.stage4 = nn.Sequential(
            # PreActResidualUnit(64, 64, 1),
            # PreActResidualUnit(64, 64, 1),
            # PreActResidualUnit(64, 64, 1),
            PreActResidualUnit(64, 1, 1)
        )
        # self.avg = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(2048, 100)

    def forward(self, x):
        x = self.pre_conv(x)
        # x = self.stage1(x)
        # x = self.stage2(x)
        # x = self.stage3(x)
        x = self.stage4(x)
        # x = self.avg(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)

        return x

    def _make_stage(self, in_channels, out_channels, num, block):

        layers = []
        layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        for _ in range(num):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

def attention56():
    return Attention([1, 1, 1])

def attention92():
    return Attention([1, 2, 3])

