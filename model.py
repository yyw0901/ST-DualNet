import argparse
import math
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


# --- [新增] DINOv2 适配器 ---
class DINOv2Wrapper(nn.Module):
    def __init__(self, version='dinov2_vits14'):
        super(DINOv2Wrapper, self).__init__()
        print(f"Loading {version} from torch.hub ...")
        # 自动下载并加载 DINOv2
        self.transformer = torch.hub.load('facebookresearch/dinov2', version)

        # 锁定参数（可选）：如果你显存不够，可以把下面两行取消注释，冻结 DINO 参数
        # for param in self.transformer.parameters():
        #     param.requires_grad = False

        # 获取维度 (Small=384, Base=768)
        self.embed_dim = self.transformer.embed_dim

    def forward(self, x):
        # x: [Batch, 3, 224, 224]

        # 1. 提取特征
        # forward_features 返回字典，我们需要 patch tokens
        out = self.transformer.forward_features(x)
        patch_tokens = out['x_norm_patchtokens']  # [B, N, C] -> [B, 256, 384]

        # 2. 还原空间维度 (Reshape)
        # 将序列变回图片网格： (B, N, C) -> (B, C, H, W)
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)  # 256开根号 = 16

        # permute(0, 2, 1) 把 C 换到中间
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return x






def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))

        #x = self.classifier(x)
        return x

# Define the VGG16-based part Model
class ft_net_VGG16_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=8, row = False):
        super(ft_net_VGG16_LPN, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.avgpool = nn.AdaptiveAvgPool2d((1,block))
        self.maxpool = nn.AdaptiveMaxPool2d((1,block))
        if row:  # row partition the ground view image
            self.avgpool = nn.AdaptiveAvgPool2d((block,1))
            self.maxpool = nn.AdaptiveMaxPool2d((block,1))
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)

        # --- 1. 这里只负责池化，不管形状 ---
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
        elif self.pool == 'max':
            x = self.maxpool(x)

        # --- 2. 【关键】所有人都要走这里！(注意缩进和 if 对齐) ---
        # 先压平 [Batch, 2048, 2]
        x = x.view(x.size(0), x.size(1), -1)

        # 调换顺序 [Batch, 2, 2048]
        x = x.permute(0, 2, 1)

        # 合并 [Batch*2, 2048]
        x = x.contiguous().view(-1, 2048)

        # --- 3. 进分类器 ---
        x = self.classifier(x)
        return x
# Define vgg16 based square ring partition for satellite images of cvusa/cvact
class ft_net_VGG16_LPN_R(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=4):
        super(ft_net_VGG16_LPN_R, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x
    # VGGNet's output: 8*8 part:4*4, 6*6, 8*8
    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

# resnet50 backbone
# resnet50 backbone
class ft_net_cvusa_LPN(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6, row=False):
        super(ft_net_cvusa_LPN, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        self.avgpool = nn.AdaptiveAvgPool2d((1, block))
        self.maxpool = nn.AdaptiveMaxPool2d((1, block))
        if row:
            self.avgpool = nn.AdaptiveAvgPool2d((block, 1))
            self.maxpool = nn.AdaptiveMaxPool2d((block, 1))
        if init_model != None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # --- 修复后的 LPN 逻辑 ---
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)

        # 直接返回特征，不进分类器
        return x

    # --- 补上这个缺失的工具函数 ---
    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)
class ft_net_cvusa_LPN_R(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=6):
        super(ft_net_cvusa_LPN_R, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            # print(x.size())
            x = x.view(x.size(0), x.size(1), -1)
            # print(x)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        #x = self.classifier(x)
        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

# Define the ResNet50-based Model
# model.py (正确示范)
# model.py 中的 ft_net 类
# model.py (请完全替换 class ft_net)
class ft_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, ibn=False, LPN=True, block=4, pool='avg'):
        super(ft_net, self).__init__()
        # 1. 加载 ResNet50
        model_ft = models.resnet50(pretrained=True)

        # --- 关键修复：确保定义了 self.model ---
        self.model = model_ft
        self.block = block
        self.LPN = LPN  # 记录是否开启 LPN

        # 2. 定义 LPN 切分池化
        # (block, 1) 表示把高度切成 block 份
        if LPN:
            self.avgpool = nn.AdaptiveAvgPool2d((block, 1))
            self.maxpool = nn.AdaptiveMaxPool2d((block, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # 3. 定义分类器
        if LPN:
            for i in range(self.block):
                name = 'classifier' + str(i)
                # ResNet50 输出特征维度是 2048
                setattr(self, name, ClassBlock(2048, class_num, droprate))
        else:
            self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        # ResNet 前向传播
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # x shape: [B, 2048, H, W]

        # LPN 逻辑
        if self.LPN:
            # 池化 -> [B, 2048, block, 1]
            x = self.avgpool(x)
            # 挤掉宽度 -> [B, 2048, block]
            x = x.view(x.size(0), x.size(1), self.block)

            # --- 【关键修改】测试模式下，直接返回特征！---
            if not self.training:
                # 把特征 permute 成 [B, block, 2048] 然后拉平 -> [B, block*2048]
                return x.permute(0, 2, 1).contiguous().view(x.size(0), -1)

            # --- 训练模式下，算分类分数 ---
            logits_list = []
            for i in range(self.block):
                part_feature = x[:, :, i]  # [B, 2048]
                name = 'classifier' + str(i)
                classifier = getattr(self, name)
                logit = classifier(part_feature)
                logits_list.append(logit)

            return logits_list

        else:
            # 普通 ResNet 逻辑 (如果不开启 LPN)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            if not self.training:
                return x  # 测试返特征
            x = self.classifier(x)
            return x  # 训练返分数
# Define the ResNet50-based part Model
# --- 请直接替换 model.py 里的 ft_net_LPN 类 ---
class ft_net_LPN(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=4):
        super(ft_net_LPN, self).__init__()
        # 1. 使用 DINOv2
        self.model = DINOv2Wrapper('dinov2_vits14')

        # 2. 维度对齐 (384 -> 2048)
        self.projector = nn.Conv2d(384, 2048, kernel_size=1, bias=False)
        init.kaiming_normal_(self.projector.weight, mode='fan_out')

        self.pool = pool
        self.block = block

        # 3. 【核心修正】强制使用水平条带切分 (Stripes)
        # (block, 1) 意思是在高度上切 block 刀，宽度保持完整
        self.avgpool = nn.AdaptiveAvgPool2d((block, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((block, 1))

        if init_model != None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        # DINOv2 提取
        x = self.model(x)
        # 升维
        x = self.projector(x)

        # LPN 切分
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.maxpool(x)
            x = x.view(x.size(0), x.size(1), -1)
        return x

    # get_part_pool 函数保持不变，不用动
    def get_part_pool(self, x, pool='avg', no_overlap=True):
        # ... (保持原代码不变) ...
        # (为了节省篇幅，这里不重复粘贴 get_part_pool 的内容，请保留你原来的代码)
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

# For cvusa/cvact
class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=True, block=2):
        super(two_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        self.sqr = True # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.
        if VGG16:
            if LPN:
                # satelite
                self.model_1 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block)
                if self.sqr:
                    self.model_1 = ft_net_VGG16_LPN_R(class_num, stride=stride, pool=pool, block=block)
            else:
                self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
                # self.vgg1 = models.vgg16_bn(pretrained=True)
                # self.model_1 = SAFA()
                # self.model_1 = SAFA_FC(64, 32, 8)
        else:
            #resnet50 LPN cvusa/cvact
            self.model_1 =  ft_net_cvusa_LPN(class_num, stride=stride, pool = pool, block=block)
            if self.sqr:
                self.model_1 = ft_net_cvusa_LPN_R(class_num, stride=stride, pool=pool, block=block)
            self.block = self.model_1.block
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                if LPN:
                    #street
                    self.model_2 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block, row = self.sqr)
                else:
                    self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
                    # self.vgg2 = models.vgg16_bn(pretrained=True)
                    # self.model_2 = SAFA()
                    # self.model_2 = SAFA_FC(64, 32, 8)
            else:
                self.model_2 =  ft_net_cvusa_LPN(class_num, stride = stride, pool = pool, block=block, row = self.sqr)
        if LPN:
            if VGG16:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(1024, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(512, class_num, droprate))
            else:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(4096, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(2048, class_num, droprate))
        else:    
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool =='avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)
            if VGG16:
                self.classifier = ClassBlock(512, class_num, droprate)
                # self.classifier = ClassBlock(4096, class_num, droprate, num_bottleneck=512) #safa 情况下
                if pool =='avg+max':
                    self.classifier = ClassBlock(1024, class_num, droprate)

        # 找到 model.py 里 three_view_net 类的 forward 函数，全部替换为：
        def forward(self, x1, x2=None, x3=None, x4=None):  # <--- 【重点1】必须加上 =None
            if self.LPN:
                # --- 分支 1：卫星 (Satellite) ---
                if x1 is None:
                    y1 = None
                else:
                    x1 = self.model_1(x1)
                    if self.training:
                        y1 = self.part_classifier(x1)  # 训练时：走分类器
                    else:
                        y1 = x1  # 测试时：直接输出特征 [B, 2048, 4]

                # --- 分支 2：街景 (Street) ---
                if x2 is None:
                    y2 = None
                else:
                    x2 = self.model_2(x2)
                    if self.training:
                        y2 = self.part_classifier(x2)
                    else:
                        y2 = x2

                # --- 分支 3：无人机 (Drone) ---
                if x3 is None:
                    y3 = None
                else:
                    x3 = self.model_3(x3)  # <--- 之前报错可能是在这里
                    if self.training:
                        y3 = self.part_classifier(x3)
                    else:
                        y3 = x3  # 测试时：直接输出特征

                return y1, y2, y3
            else:
                # 非 LPN 模式的代码（如果没用到可以不管，但为了完整性保留）
                if x1 is None:
                    y1 = None
                else:
                    x1 = self.model_1(x1)

                if x2 is None:
                    y2 = None
                else:
                    x2 = self.model_2(x2)

                if x3 is None:
                    y3 = None
                else:
                    x3 = self.model_3(x3)
                return y1, y2, y3
    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            # part[i] = torch.squeeze(x[:,:,i])
            part[i] = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y

class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=6):
        super(three_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
        elif LPN:
            self.model_1 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            self.model_2 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            # self.block = self.model_1.block
        else: 
            self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        if share_weight:
            self.model_3 = self.model_1
        else:
            if VGG16:
                self.model_3 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            elif LPN:
                self.model_3 =  ft_net_LPN(class_num, stride = stride, pool = pool, block = block)
            else:
                self.model_3 =  ft_net(class_num, stride = stride, pool = pool)
        if LPN:
            if pool == 'avg+max':
                for i in range(self.block):
                    name = 'classifier'+str(i)
                    setattr(self, name, ClassBlock(4096, class_num, droprate))
            else:
                for i in range(self.block):
                    name = 'classifier'+str(i)
                    setattr(self, name, ClassBlock(2048, class_num, droprate))
        else:    
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool =='avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if self.LPN:
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                if self.training:
                    y1 = self.part_classifier(x1)
                else:
                    y1 = x1

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                if self.training:
                    y2 = self.part_classifier(x2)
                else:
                    y2 = x2  # <--- 必须有这个

            if x3 is None:
                y3 = None
            else:
                x3 = self.model_3(x3)
                if self.training:
                    y3 = self.part_classifier(x3)
                else:
                    y3 = x3  # <--- 必须有这个

            return y1, y2, y3
        else:
            # 非 LPN 模式的备用代码（不需要动）
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
            if x3 is None:
                y3 = None
            else:
                x3 = self.model_3(x3)
            return y1, y2, y3

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:,:,i].view(x.size(0),-1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(701, droprate=0.5, pool='avg', stride=1, VGG16=False, LPN=True, block=8)

    # net = three_view_net(701, droprate=0.5, pool='avg', stride=1, share_weight=True, LPN=True, block=2)
    # net.eval()

    # net = ft_net_VGG16_LPN_R(701)
    # net = ft_net_cvusa_LPN(701, stride=1)
    # net = ft_net(701)

    print(net)

    input = Variable(torch.FloatTensor(2, 3, 256, 256))
    output1,output2 = net(input,input)
    # output1,output2,output3 = net(input,input,input)
    # output1 = net(input)
    # print('net output size:')
    # print(output1.shape)
    # print(output.shape)
    for i in range(len(output1)):
        print(output1[i].shape)
    # x = torch.randn(2,512,8,8)
    # x_shape = x.shape
    # pool = AzimuthPool2d(x_shape, 8)
    # out = pool(x)
    # print(out.shape)


# =====================================================================
#  将以下代码粘贴到 model.py 的最下方
# =====================================================================

# 确保文件头部导入了 models，如果没有，请在 model.py 最上面加一行：
# from torchvision import models

class TwoStreamNet(nn.Module):
    def __init__(self, class_num, droprate, stride=1, pool='avg', block=4):
        super(TwoStreamNet, self).__init__()
        # 分支 1：ResNet
        self.model_1 = ft_net(class_num, droprate, stride=stride, pool=pool, block=block)

        # 分支 2：DINOv2 (请确认你的 DINO 类名，比如 ft_net_LPN 或 ft_net_dinov2)
        # 根据之前的报错，你的 DINO 类里应该有 transformer 层
        self.dino_branch = ft_net_LPN(class_num, droprate, stride=stride, pool=pool, block=block)

    def forward(self, x1, x2=None, x3=None):
        # ------------------------------------------------------------
        # A. 处理 DINO 分支 (修正点：必须分开跑，不能一次传三个)
        # ------------------------------------------------------------
        if x2 is not None and x3 is not None:
            # --- 3 View Training (卫星 + 街景 + 无人机) ---
            # 关键修改：一张一张传进去
            out_dino_1 = self.dino_branch(x1)  # 跑卫星
            out_dino_2 = self.dino_branch(x2)  # 跑街景
            out_dino_3 = self.dino_branch(x3)  # 跑无人机
            # 打包成 tuple 返回
            ret_dino = (out_dino_1, out_dino_2, out_dino_3)

        elif x2 is not None:
            # --- 2 View Training ---
            out_dino_1 = self.dino_branch(x1)
            out_dino_2 = self.dino_branch(x2)
            ret_dino = (out_dino_1, out_dino_2)

        else:
            # --- Testing (测试模式) ---
            ret_dino = self.dino_branch(x1)

        # ------------------------------------------------------------
        # B. 处理 CNN 分支 (ResNet50)
        # ------------------------------------------------------------
        if self.training:
            # 定义一个内部函数，帮我们把 CNN 输出伪装成 LPN 的 list 格式
            def run_cnn_branch(img):
                features = self.cnn_backbone(img)  # [B, 2048, H, W]
                features = self.avgpool(features)  # [B, 2048, 1, 1]
                features = features.view(features.size(0), -1)  # [B, 2048]
                logits = self.cnn_classifier(features)  # [B, 701]
                # 复制 block 份，为了配合 LPN Loss 的接口
                return [logits for _ in range(self.block)]

            # 分别对 3 张图跑 CNN
            out_cnn_1 = run_cnn_branch(x1)

            if x2 is not None:
                out_cnn_2 = run_cnn_branch(x2)
            else:
                out_cnn_2 = None

            if x3 is not None:
                out_cnn_3 = run_cnn_branch(x3)
            else:
                out_cnn_3 = None

            # 根据 View 数量返回
            if x3 is not None:
                return ret_dino, (out_cnn_1, out_cnn_2, out_cnn_3)
            elif x2 is not None:
                return ret_dino, (out_cnn_1, out_cnn_2)
            else:
                # 理论上训练时不会进这里
                return ret_dino, (out_cnn_1,)

        else:
            # --- 测试模式 ---
            # 直接返回 DINO 特征
            return ret_dino
