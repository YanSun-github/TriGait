import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple
from torchvision.ops import RoIAlign
from einops import rearrange


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()




class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs



class Attention(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim, bias=False)
        self.proj_drop=nn.Dropout(proj_drop)

    def forward(self,x):
        #n,c,t,vv=x.size()
        #print("inp", x.shape)
        #x = x.permute(0,2,3,1).reshape(n*t,vv,c).contiguous()
        #print("gai",x.shape)
        B, N, C = x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).contiguous()
        q,k,v=qkv[0],qkv[1],qkv[2]
        #print("q:",q.shape)

        attn = (q @ k.transpose(-2,-1).contiguous())*self.scale
        #print("attn1:",attn.shape)
        attn = attn.softmax(dim=1)
        #print("attn2:", attn.shape)
        attn = self.attn_drop(attn)
        #print("attn3:", attn.shape)

        x=(attn@v).transpose(1,2).reshape(B,N,C).contiguous()
        #print("attn4:", x.shape)
        x= self.proj(x)
        #print("attn5:", x.shape)
        x=self.proj_drop(x)#.reshape(n, t, vv, c).permute(0,3,1,2).contiguous()
        #print("attn6:", x.shape)
        return x



class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv2d(gate_channel // reduction_ratio,
                                             gate_channel // reduction_ratio,
                                             kernel_size=3,
                                             padding=dilation_val,
                                             dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):

        att = 1+torch.sigmoid(self.gate_s(in_tensor).expand_as(in_tensor))
        x = att*in_tensor
        return x

class GLSpatialGate(nn.Module):
    def __init__(self, gate_channel, having=3, fm_sign=True):
        super(GLSpatialGate, self).__init__()
        self.halving = having
        self.fm_sign = fm_sign
        self.global_SpatialGate = SpatialGate(gate_channel)

        self.local_SpatialGate0 = SpatialGate(gate_channel)
        self.local_SpatialGate1 = SpatialGate(gate_channel)
        self.local_SpatialGate2 = SpatialGate(gate_channel)
        self.local_SpatialGate3 = SpatialGate(gate_channel)
        self.local_SpatialGate4 = SpatialGate(gate_channel)
        self.local_SpatialGate5 = SpatialGate(gate_channel)
        self.local_SpatialGate6 = SpatialGate(gate_channel)
        self.local_SpatialGate7 = SpatialGate(gate_channel)

    def forward(self, x):
        gob_feat = self.global_SpatialGate(x)
        if self.halving == 0:
            lcl_feat = self.local_SpatialGate(x)
        else:
            h = x.size(2)
            split_size = int(h // 2 ** self.halving)
            lcl_feat = x.split(split_size, 2)
            lcl = [self.local_SpatialGate0(lcl_feat[0]), self.local_SpatialGate1(lcl_feat[1]),
                   self.local_SpatialGate2(lcl_feat[2]),
                   self.local_SpatialGate3(lcl_feat[3]), self.local_SpatialGate4(lcl_feat[4]),
                   self.local_SpatialGate5(lcl_feat[5]),
                   self.local_SpatialGate6(lcl_feat[6]), self.local_SpatialGate7(lcl_feat[7])]
            lcl_feat = torch.cat([_ for _ in lcl], 2)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=2))
        return feat



class TemporalGate(nn.Module):
    def __init__(self,in_channels,out_channels,bin_num=[64]):
        super(TemporalGate, self).__init__()
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        self.dilate_conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(1,0), dilation=(1,1))

        self.dilate_conv_2 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(2,0), dilation=(2,1))

        self.dilate_conv_4 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(4,0), dilation=(4,1))

    def forward(self,x):
        #in [n,c,s,h,w]
        #out [n,c,p]
        n,c,s,h,w = x.shape
        x = x.permute(0,2,1,3,4).contiguous().view(n*s,c,h,w)
        x = self.HPP(x)  #ns,c,t,p
        x = x.view(n,s,c,-1).permute(0,2,1,3).contiguous()

        t_f = self.dilate_conv_1(x).permute(0, 2, 1, 3).contiguous().unsqueeze(-1)
        t_s = self.dilate_conv_2(x).permute(0, 2, 1, 3).contiguous().unsqueeze(-1)+t_f
        t_l = self.dilate_conv_4(x).permute(0, 2, 1, 3).contiguous().unsqueeze(-1)+t_s

        t = torch.cat([t_f, t_s, t_l], -1)
        return t

def conv1d(in_planes, out_planes, kernel_size, has_bias=False, **kwargs):
    return nn.Conv1d(in_planes, out_planes, kernel_size, bias=has_bias, **kwargs)

class MTA(nn.Module):
    def __init__(self, in_planes, part_num=64, div=16):
        super(MTA, self).__init__()
        self.in_planes = in_planes
        self.part_num = part_num
        self.div = div

        self.in_conv = conv1d(part_num * 3 * in_planes, part_num * 3 * in_planes // div, 1, False, groups=part_num)

        self.out_conv = conv1d(part_num * 3 * in_planes // div, part_num * 3 * in_planes, 1, False, groups=part_num)

        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, t):
        n, s, c, h, _ = t.size()
        t = t.permute(0, 3, 4, 2, 1).contiguous().view(n, h * 3 * c, s)

        t_inter = self.leaky_relu(self.in_conv(t))
        t_attention = self.out_conv(t_inter).sigmoid()

        weighted_sum = (t_attention * t).view(n, h, 3, c, s).sum(2).sum(-1) / t_attention.view(n, h, 3, c, s).sum(
            2).sum(-1)
        weighted_sum = self.leaky_relu(weighted_sum).permute(0, 2, 1).contiguous()

        return weighted_sum


class JSA(nn.Module):
    def __init__(self, channle,head_num = 8,res = True):
        super(JSA, self).__init__()
        self.bn0 = nn.BatchNorm2d(channle)
        self.atten = Attention(channle, head_num, qkv_bias=False)
        self.mish = nn.Mish(inplace=True)
        self.bn1 = nn.BatchNorm2d(channle)
        self.res = res

    def forward(self, x):
        """
            x  : [n, c, t, v]
            ret: [n, c, p]
        """
        x = self.bn0(x)
        n, c, t, vv = x.size()
        # print("inp", x.shape)
        x = x.permute(0, 2, 3, 1).reshape(n * t, vv, c).contiguous()
        x = self.atten(x) + x if self.res else self.atten(x)
        x = x.reshape(n, t, vv, c).permute(0,3,1,2).contiguous()
        x = self.bn1(self.mish(x))
        return x



class TCN(nn.Module):
    def __init__(self,in_c, out_c,kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),res = True):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(nn.Dropout(),
                      nn.Conv2d(in_c, out_c, kernel_size, stride, padding,
                                bias=False),
                      nn.Mish(),
                      nn.BatchNorm2d(out_c))


    def forward(self, x):
        """
            x  : [n, in_c, t, v]
            ret: [n, out_c, t,v]
        """
        return self.tcn(x)

class JSATCN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)):
        super(JSATCN, self).__init__()
        self.f = False
        if in_c !=out_c:
            self.cc = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                    nn.Mish(),
                                    nn.BatchNorm2d(out_c)
                                    )
            self.f = True

        self.tcn = TCN(in_c,out_c,kernel_size, stride, padding)
        self.st = JSA(out_c)



    def forward(self, x):
        """
            x  : [n, c, t, v]
            ret: [n, c, p]
        """
        f_t = self.tcn(x)
        x = self.cc(x) if self.f else x
        f_st = x + f_t
        f_st = self.st(f_st)
        return x + f_st


class DividPart(nn.Module):
    def __init__(self, head=[0,1,2,3,4], shoulder=[5,6], elbow=[7,8], wrist=[9,10], hip=[11,12], knee=[13,14], ankle=[15,16],if_ou = False):
        super(DividPart, self).__init__()
        self.shoulder = shoulder
        self.hip = hip
        self.knee = knee
        self.head = head
        self.elbow = elbow
        self.wrist = wrist
        self.ankle = ankle
        self.if_ou = if_ou

    def mm(self, part, bottom=0, top=0, ratio=1, op=False):
        '''n, s, v'''
        if op:
            max_value = torch.ceil(torch.div(part.max(1)[0].max(1)[0] - top, bottom - top) * 64).int()
            min_value = torch.floor(torch.div(part.min(1)[0].min(1)[0] - top, bottom - top) * 64).int()
        else:
            max_value = part.max(1)[0].max(1)[0]
            min_value = part.min(1)[0].min(1)[0]
        return max_value, min_value

    def Aligh(self, pose, if_ou=False, node=17):
        y = pose[:, :, 0].unsqueeze(2).repeat(1, 1, node)
        # print("y",pose)
        pose = pose - y
        # print("pose-y",pose)
        shoulder = [5, 6]
        if if_ou:
            shoulder = [2, 5]
        ratio = (pose[:, :, shoulder].mean(-1, keepdim=True).repeat(1, 1, node))
        # print("ratio",ratio)
        pose = pose.div(ratio)
        mi = pose.min(-1, keepdim=True)[0]
        pose = pose - mi
        return pose

    def forward(self, poses):
        '''
            poses: [n, c, s, v]
            outs: [n, 4]
        '''
        # poses = poses[:,:2,:,:]
        n, c, s, v = poses.shape

        poses = self.Aligh(poses[:, 1, :, :],self.if_ou, v)  # nsv
        bottom, top = self.mm(poses)
        ratio = bottom - top  # torch.div(bottom - top, 64)#(bottom - top) // 64

        head_max, head_min = self.mm(poses[:, :, self.head], bottom, top, ratio, True)
        shoulder_max, shoulder_min = self.mm(poses[:, :, self.shoulder], bottom, top, ratio, True)
        elbow_max, elbow_min = self.mm(poses[:, :, self.elbow], bottom, top, ratio, True)
        wrist_max, wrist_min = self.mm(poses[:, :, self.wrist], bottom, top, ratio, True)
        hip_max, hip_min = self.mm(poses[:, :, self.hip], bottom, top, ratio, True)
        knee_max, knee_min = self.mm(poses[:, :, self.knee], bottom, top, ratio, True)
        ankle_max, ankle_min = self.mm(poses[:, :, self.ankle], bottom, top, ratio, True)
        ma = torch.stack([head_max, shoulder_max, elbow_max, wrist_max, hip_max, knee_max, ankle_max])
        mi = torch.stack([head_min, shoulder_min, elbow_min, wrist_min, hip_min, knee_min, ankle_min])
        c = torch.nonzero(ma<=mi)
        for i in range(len(c)):
            ma[c[i][0], c[i][1]] = (c[i][0] + 1) * 9
            mi[c[i][0], c[i][1]] = c[i][0] * 9
        d = torch.nonzero(ma - mi > 30)
        for i in range(len(d)):
            ma[d[i][0], d[i][1]] = (d[i][0] + 1) * 9
            mi[d[i][0], d[i][1]] = d[i][0] * 9
        return ma, mi



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),   # dim=1024, hidden_dim=2048
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention2(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Early_fuse(nn.Module):
    def __init__(self , sil_c, pose_c,  out_c, atten_depth=4, max_h=11, pose = [[0,1,2,3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]):
        super().__init__()


        self.pool1 = nn.AdaptiveAvgPool1d(2)
        self.pose = pose
        self.fuse_dim = int(sil_c * max_h + pose_c * 2)
        self.max_h = max_h
        self.out_c =out_c

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.fuse_dim),
            nn.Linear(self.fuse_dim, out_c),
            nn.LayerNorm(out_c),
        )

        
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([])

        for _ in range(atten_depth):
            self.layers.append(nn.ModuleList([
                PreNorm(out_c, Attention2(out_c, heads=8, dim_head=64, dropout=0.1)),
                PreNorm(out_c, FeedForward(out_c, 1024, dropout=0.1))
            ]))
       

    def forward(self,sil,pose,max,min):
        
        n, c, h, w = sil.shape  # n c h w

        p, n = max.shape
        silpart = torch.zeros((p, n, c, self.max_h)).cuda()
        for i in range(n):
            sil_tmp = sil[i, ...]  # chw

            for j in range(p):
                p_feat = sil_tmp[:, min[j, i]:max[j, i], :].view(c, self.max_h, -1)  # s c max_h _
                silpart[j, i, ...] = p_feat.mean(-1) + p_feat.max(-1)[0]  # 7 n s c max_h
        p, n, c, h = silpart.shape

        silpart = silpart.permute(1,  2, 3, 0).contiguous().view(n, c * h, p)
        

        n,c,v = pose.shape
        node = 2
        posepart = torch.zeros((p, n, c, node)).cuda()
        for i in range(p):
            posepart[i,...] = F.leaky_relu(self.pool1(pose[:,:,self.pose[i]]))
        p,n,c,v = posepart.shape
        posepart = posepart.permute(1,2,3,0).contiguous().view(n,c*v,p)
        

        fuse = torch.cat([silpart,posepart],1).permute(0,2,1).contiguous()  #ncp->npc
        fuse = self.to_patch_embedding(fuse)

        
        fuse = self.dropout(fuse)
        for attn, ff in self.layers:
            fuse = attn(fuse) + fuse
            fuse = ff(fuse) + fuse
        return fuse

class Late_fuse(nn.Module):
    def __init__(self, p=5):
        super().__init__()
        self.p = nn.Parameter(
            torch.ones(1) * p)
    def forward(self,sil,pose):
        return torch.cat([sil*self.p, pose], 1)


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False
