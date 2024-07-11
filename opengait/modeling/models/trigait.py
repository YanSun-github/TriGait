import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper,SeparateBNNecks,HorizontalPoolingPyramid
from ..modules import TCN,JSATCN
from ..modules import GLSpatialGate, TemporalGate,MTA
from ..modules import DividPart, Early_fuse, Late_fuse



class backbone_casiab(nn.Module):
    def __init__(self,in_c = [32, 64, 128,256]):
        super(backbone_casiab, self).__init__()
        # 3D Convolution
        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU(inplace=True)
        )


        self.ConvA0 = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU(inplace=True)
        )


        self.ConvA1 = nn.Sequential(

            BasicConv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[1]),
            nn.ReLU(inplace=True),
        )

        self.ConvA2 = nn.Sequential(
            BasicConv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[2]),
            nn.ReLU(inplace=True),
        )

        self.ConvA3 = nn.Sequential(
            BasicConv3d(in_c[2], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[3]),
            nn.ReLU(inplace=True),
        )


    def forward(self,inputs):
        x = self.conv3d(inputs)
        sil = x.clone()
        x = self.ConvA0(x)      
        x = self.ConvA1(x)
        x = self.ConvA2(x)
        x = self.ConvA3(x)
        return x, sil



class backbone_gait3d(nn.Module):
    def __init__(self,in_c = [64, 128,256, 512]):
        super(backbone_gait3d, self).__init__()
        # 3D Convolution
        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU(inplace=True)
        )


        self.ConvA0 = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU(inplace=True)
        )


        self.ConvA1 = nn.Sequential(

            BasicConv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[1]),
            BasicConv3d(in_c[1], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[1]),
            nn.ReLU(inplace=True),
        )

        self.ConvA2 = nn.Sequential(
            BasicConv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[2]),
            BasicConv3d(in_c[2], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[2]),
            nn.ReLU(inplace=True),
        )

        self.ConvA3 = nn.Sequential(
            BasicConv3d(in_c[2], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[3]),
            BasicConv3d(in_c[3], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[3]),
            nn.ReLU(inplace=True),
        )


    def forward(self,inputs):
        x = self.conv3d(inputs)
        sil = x.clone()
        x = self.ConvA0(x)
        x = self.ConvA1(x)
        x = self.ConvA2(x)
        x = self.ConvA3(x)
        return x,sil


class backbone_oumvlp(nn.Module):
    def __init__(self,in_c = [64, 128, 256,512]):
        super(backbone_oumvlp, self).__init__()

        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU()
        )


        self.ConvA0 = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU()
        )


        self.ConvA1 = nn.Sequential(

            BasicConv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[1]),
            nn.ReLU(),
        )

        self.ConvA2 = nn.Sequential(
            BasicConv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[2]),
            nn.ReLU(),
        )

        self.ConvA3 = nn.Sequential(
            BasicConv3d(in_c[2], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[3]),
            nn.ReLU(),
        )



    def forward(self,inputs):
        x = self.conv3d(inputs)
        sil = x.clone()
        x = self.ConvA0(x)
        x = self.ConvA1(x)
        x = self.ConvA2(x)
        x = self.ConvA3(x)
        return x, sil




class TriGait(BaseModel):
    def __init__(self, *args, **kargs):
        super(TriGait, self).__init__(*args, **kargs)


    def build_network(self, model_cfg):
        sil_channels = model_cfg['channels']

        if model_cfg['Dataset']=='CASIA-B':
            self.ES = backbone_casiab(sil_channels)
        elif model_cfg['Dataset']=='OUMVLP':
            self.ES = backbone_oumvlp(sil_channels)
        else: self.ES = backbone_gait3d(sil_channels)

       
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])

        self.attention_s = GLSpatialGate(sil_channels[-1])
        self.attention_t = TemporalGate(sil_channels[-1], sil_channels[-1],bin_num=model_cfg["bin_num"])
        self.mta = MTA(sil_channels[-1], part_num=model_cfg["bin_num"][0])




        # pose branch

        pose_cfg = model_cfg['pos_cfg']
        in_c2 = model_cfg['pos_cfg']['in_channels']

        self.BN2d = nn.BatchNorm2d(in_c2[0])

        self.jsatcn = nn.Sequential(JSATCN(in_c2[0],in_c2[1]),
                                    JSATCN(in_c2[1], in_c2[1]),
                                    JSATCN(in_c2[1],in_c2[2]),
                                    JSATCN(in_c2[2],in_c2[3]))

        self.Avg = nn.AdaptiveAvgPool1d(model_cfg["bin_num"])
        self.TPmean = PackSequenceWrapper(torch.mean)

        #fuse
        self.posepretreatment = TCN(in_c2[0], in_c2[1])
        self.cc = nn.Sequential(
            nn.Conv2d(in_c2[0], in_c2[1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.Mish(),
            nn.BatchNorm2d(in_c2[1])
            )

        self.divid = DividPart(pose_cfg['head'], pose_cfg['shoulder'], pose_cfg['elbow'], pose_cfg['wrist'],
                               pose_cfg['hip'], pose_cfg['knee'], pose_cfg['ankle'], if_ou=pose_cfg[
                'if_ou'])
        fuse_channel = model_cfg['SeparateFCs']['in_channels']
        self.early_fuse = Early_fuse(sil_channels[0], in_c2[1], out_c=fuse_channel, atten_depth=2)

        self.late_fuse = Late_fuse()

        
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

    

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL

        sils = ipts[0].unsqueeze(1)  ## [n, 1,s, h, w]
        poses = ipts[1].permute(0, 3, 1, 2).contiguous()  # [n, s, v, c]->n,c,s,v
        del ipts


        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)
            poses = poses.repeat(1, 1, repeat, 1)



        '''extract sil features '''


        f,fuse_sil = self.ES(sils)

        f_s = self.TP(f, seqL, options={"dim": 2})[0]
        f_s = self.attention_s(f_s)
        f_s = self.HPP(f_s)  # [n, c, p]
        f_te = self.attention_t(f)
        f_te = self.mta(f_te)

        outs = torch.cat((f_s,f_te),1)



        '''extract skeleton features'''


    
        x = self.BN2d(poses) 
        fuse_pose = self.posepretreatment(x)+self.cc(x)
        x = self.jsatcn(x)
        n, c, _, _ = x.shape  # n,c,t,v
        x = self.TPmean(x, seqL, options={"dim": 2})

        x = self.Avg(x)  # ncp


        '''early fuse'''
          #[n,c1,p]

        ma, mi = self.divid(poses)

        fuse_pose = self.TPmean(fuse_pose, seqL, options={"dim": 2})
        fuse_sil = self.TPmean(fuse_sil, seqL, options={"dim": 2})


        fuse_early = self.early_fuse(fuse_sil,fuse_pose,ma,mi)  #n,p,c
        fuse_early = fuse_early.permute(0,2,1).contiguous() #ncp

        fuse_late = self.late_fuse(outs,x)
        fuse = torch.cat([fuse_late, fuse_early], 2)

        
        embed_1 = self.FCs(fuse)  # [n, c, p]
        embed_2, logi = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1



        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            },
            'logits_feat': {
                'logits': outs
            }
        }
        return retval