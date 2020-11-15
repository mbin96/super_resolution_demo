## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from . import common

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return UNETPP(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        self.CA = (CALayer(n_feat, reduction))
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


#orig
class UBLK(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_resblocks, reduction):
        super(UBLK, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x



class UpCat(nn.Module):
    def __init__(self, conv, n_feat, out_feat, kernel_size, CA):
        super(UpCat, self).__init__()
        self.up = UpSampler(conv, n_feat, out_feat, kernel_size)
        # self.CA = (CALayer(CA))
        
    def forward(self, tensorList, smallTensor):
        
        smallTensor = self.up(smallTensor)
        # 업스케일된거 크기 맞춰주기
        tensorList.append(smallTensor[:,:,:tensorList[0].size()[2],:tensorList[0].size()[3]]) 
        result = torch.cat(tensorList,dim = 1)
        # result = self.CA(result)
        return result

class UpSampler(nn.Module):
    def __init__(self, conv, n_feat, out_feat, kernel_size):
        super(UpSampler, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, 4*out_feat, kernel_size))
        modules_body.append(nn.ReLU(True))
        modules_body.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class DownSampler(nn.Module):
    def __init__(self, conv, n_feat, out_feat, kernel_size):
        super(DownSampler, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        modules_body.append(nn.ReLU(True))
        modules_body.append(nn.Conv2d(n_feat, out_feat, 1, stride=2))
        modules_body.append(nn.ReLU(True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class D_DownBlock(torch.nn.Module):
    def __init__(self, n_feat, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, act = torch.nn.ReLU(True), norm=None):
        super(D_DownBlock, self).__init__()
      
        self.conv = torch.nn.Conv2d(n_feat*num_stages, n_feat, 1, 1, 0, bias=bias)
        
        self.act = act
        
        upConv1 = [torch.nn.ConvTranspose2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]
        downConv1 = [torch.nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]
        downConv2 = [torch.nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]

        self.down_conv1 = nn.Sequential(downConv1)
        self.down_conv2 = nn.Sequential(upConv1)
        self.down_conv3 = nn.Sequential(downConv2)

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.down_conv1(x)
    	l0 = self.down_conv2(h0)
    	h1 = self.down_conv3(l0 - x)
    	return h1 + h0

class DownBlock(torch.nn.Module):
    def __init__(self, n_feat, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True,act = torch.nn.ReLU(True), norm=None):
        super(DownBlock, self).__init__()
            
        # self.conv = torch.nn.Conv2d(n_feat*num_stages, n_feat, 1, 1, 0, bias=bias)
        
        self.act = act

        upConv1 = [torch.nn.ConvTranspose2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]
        downConv1 = [torch.nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]
        downConv2 = [torch.nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]

        self.down_conv1 = nn.Sequential(downConv1)
        self.down_conv2 = nn.Sequential(upConv1)
        self.down_conv3 = nn.Sequential(downConv2)

    def forward(self, x):
    	# x = self.conv(x)
    	h0 = self.down_conv1(x)
    	l0 = self.down_conv2(h0)
    	h1 = self.down_conv3(l0 - x)
    	return h1 + h0

class D_UpBlock(torch.nn.Module):
    def __init__(self, n_feat, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, act = torch.nn.ReLU(True), norm=None):
        super(D_UpBlock, self).__init__()
      
        self.conv = torch.nn.Conv2d(n_feat*num_stages, n_feat, 1, 1, 0, bias=bias)
        
        self.act = act
        
        upConv1 = [torch.nn.ConvTranspose2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]
        upConv2 = [torch.nn.ConvTranspose2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]
        downConv1 = [torch.nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]

        self.up_conv1 = nn.Sequential(upConv1)
        self.up_conv2 = nn.Sequential(downConv1)
        self.up_conv3 = nn.Sequential(upConv2)

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class UpBlock(torch.nn.Module):
    def __init__(self, n_feat, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True,act = torch.nn.ReLU(True), norm=None):
        super(UpBlock, self).__init__()
            
        
        # self.conv = torch.nn.Conv2d(n_feat*num_stages, n_feat, 1, 1, 0, bias=bias)

        self.act = act

        upConv1 = [torch.nn.ConvTranspose2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]
        upConv2 = [torch.nn.ConvTranspose2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]
        downConv1 = [torch.nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), self.act]

        self.up_conv1 = nn.Sequential(upConv1)
        self.up_conv2 = nn.Sequential(downConv1)
        self.up_conv3 = nn.Sequential(upConv2)

    def forward(self, x):
    	# x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

## Residual Channel Attention Network (RCAN)
class UNETPP(nn.Module):
    def __init__(self, scale = 2, conv=common.default_conv):
        super(UNETPP, self).__init__()
        # self.piramid = args.piramid
        # n_resgroups = args.n_resgroups
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        # reduction = args.reduction 
        reduction = 16
        # scale = args.scale[0]
        # act = nn.ReLU(True)
        # self.interpolation = args.interpolation
        #RGB mean for DIV2K
        self.sub_mean = common.MeanShift(255)
        #upscale with bilinear        
        self.biUp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        #define head module
        #feature extrector
        self.fe = nn.Conv2d(3, n_feats, kernel_size, padding = 1)


        #L1
        self.blk0_0 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)
        
        #L2
        self.blk1_0 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)
        self.blk0_1 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)

        #L3
        self.blk2_0 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)
        self.blk1_1 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)
        self.blk0_2 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)
        
        #L4 - 1
        self.blk3_0 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)
        self.blk2_1 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)
        self.blk1_2 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)
        self.blk0_3 = UBLK(conv, n_feats, kernel_size, n_resblocks, reduction)

        

        self.dw1_0 = DownSampler(conv, n_feats, n_feats ,kernel_size)
        self.dw2_0 = DownSampler(conv, n_feats, n_feats ,kernel_size)
        self.dw3_0 = DownSampler(conv, n_feats, n_feats ,kernel_size)
        
        
        if scale == 4:
            module_up_bp_1 = [UpSampler(conv, n_feats, n_feats ,kernel_size)]
            module_up_bp_2 = [UpSampler(conv, n_feats, n_feats ,kernel_size)]
            module_up_bp_3 = [UpSampler(conv, n_feats, n_feats ,kernel_size)]
            module_up_bp_4 = [UpSampler(conv, n_feats, n_feats ,kernel_size)]
        
            module_up_bp_1.append(UpSampler(conv, n_feats, n_feats ,kernel_size))
            module_up_bp_2.append(UpSampler(conv, n_feats, n_feats ,kernel_size))
            module_up_bp_3.append(UpSampler(conv, n_feats, n_feats ,kernel_size))
            module_up_bp_4.append(UpSampler(conv, n_feats, n_feats ,kernel_size))

            self.up_bp_1 = nn.Sequential(*module_up_bp_1)
            self.up_bp_2 = nn.Sequential(*module_up_bp_2)
            self.up_bp_3 = nn.Sequential(*module_up_bp_3)
            self.up_bp_4 = nn.Sequential(*module_up_bp_4)
        else:
            self.up_bp_1 = UpSampler(conv, n_feats, n_feats ,kernel_size)
            self.up_bp_2 = UpSampler(conv, n_feats, n_feats ,kernel_size)
            self.up_bp_3 = UpSampler(conv, n_feats, n_feats ,kernel_size)
            self.up_bp_4 = UpSampler(conv, n_feats, n_feats ,kernel_size)
        



        #L1
        
        #L2
        self.up1_0 = UpCat(conv, n_feats, n_feats , kernel_size, n_feats*2)
        
        #L3
        self.up2_0 = UpCat(conv, n_feats, n_feats ,kernel_size, n_feats*2)
        self.up1_1 = UpCat(conv, n_feats, n_feats ,kernel_size,n_feats*3)
        
        #L4
        self.up3_0 = UpCat(conv, n_feats, n_feats ,kernel_size, n_feats*2)
        self.up2_1 = UpCat(conv, n_feats, n_feats ,kernel_size, n_feats*3)
        self.up1_2 = UpCat(conv, n_feats, n_feats ,kernel_size, n_feats*4)

        
        #L2
        self.ff0_1 = nn.Conv2d(n_feats * 2, n_feats, 1)
        
        #L3
        self.ff1_1 = nn.Conv2d(n_feats * 2, n_feats, 1) 
        self.ff0_2 = nn.Conv2d(n_feats * 3, n_feats, 1)
        
        #L4
        self.ff2_1 = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.ff1_2 = nn.Conv2d(n_feats * 3, n_feats, 1)
        self.ff0_3 = nn.Conv2d(n_feats * 4, n_feats, 1)
        
        self.fff = nn.Conv2d(n_feats * 3, n_feats, 1)
        self.ffX2_3 = nn.Conv2d(n_feats * 4, n_feats, 1) 
        
        
        # modules_body.append(conv(n_feats, n_feats, kernel_size))

        ## define tail module
        modules_tail = []
        
        # concat
        modules_tail.append(nn.Conv2d(n_feats * 8, n_feats * 8, kernel_size, padding=1))
        modules_tail.append(nn.ReLU(inplace=True))
        modules_tail.append(nn.PixelShuffle(2)) 

        modules_tail.append(nn.Conv2d(n_feats * 4, n_feats * 4, kernel_size, padding=1))
        modules_tail.append(nn.ReLU(inplace=True))
        modules_tail.append(nn.PixelShuffle(2)) 
        
        modules_tail.append(nn.Conv2d(n_feats*2, n_feats*2, kernel_size, padding=1))
        modules_tail.append(nn.ReLU(inplace=True))
        modules_tail.append(nn.PixelShuffle(2)) 
        
        modules_tail.append(nn.Conv2d(n_feats//2, 3, kernel_size, padding=1))
        
        self.concat_w_in= nn.Conv2d(n_feats*2, n_feats, kernel_size, padding=1)
        self.color= nn.Conv2d(n_feats, 3, kernel_size, padding=1)
        modules_up = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.output_bp = conv(4*n_feats, 3, kernel_size)
        self.add_mean = common.MeanShift(255, sign=1)
        
        self.up = nn.Sequential(*modules_up)

    def forward(self, x):

        x = self.sub_mean(x)
        
        x = self.fe(x)
        

        #L1
        x0_0 = self.blk0_0(x)

        xbp_1 = self.up_bp_1(x0_0) 
        
        #L2
        x1_0 = self.blk1_0(self.dw1_0(x0_0))

        x0_1 = self.blk0_1(self.ff0_1(self.up1_0([x0_0], x1_0)))

        xbp_2 = self.up_bp_2(x0_1) 

        #L3
        x2_0 = self.blk2_0(self.dw2_0(x1_0))
        
        x1_1 = self.blk1_1(self.ff1_1(self.up2_0([x1_0], x2_0)))

        x0_2 = self.blk0_2(self.ff0_2(self.up1_1([x0_1,x0_0],x1_1)))


        xbp_3 = self.up_bp_3(x0_2)
        
        #끝부분 있게
        x3_0 = self.blk3_0(self.dw3_0(x2_0))
        x2_1 = self.blk2_1(self.ff2_1(self.up3_0([x2_0], x3_0)))
        

        x1_2 = self.blk1_2(self.ff1_2(self.up2_1([x1_1,x1_0], x2_1)))

        x0_3 = self.blk0_3(self.ff0_3(self.up1_2([x0_2,x0_1,x0_0],x1_2)))
        xbp_4 = self.up_bp_4(x0_3)




        x = self.output_bp(torch.cat((xbp_1, xbp_2, xbp_3, xbp_4),dim = 1))

        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
