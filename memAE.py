import math
import torch
from torch.nn import functional as F
import torch.nn as nn
from basic_modules import *
from twfinch import *
# from sspcab import *
# from Memory import *
import copy


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)     #大于0，硬收缩
    return output


class Memory(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, choices=None, hard_shrink=True):
        super(Memory, self).__init__()
        self.mem_dim = mem_dim   #插槽
        self.fea_dim = fea_dim
        self.choice = choices
        self.hard_shrink = hard_shrink
        self.memMatrix = nn.Parameter(torch.empty(mem_dim, fea_dim))
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)      # 随机化参数

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        att_weight = F.linear(input=x, weight=self.memMatrix)  # [N,C] by [M,C]^T --> [N,M] 线性变换
        att_weight = F.softmax(att_weight, dim=1)  # NxM

        # if use hard shrinkage
        if self.shrink_thres > 0 and self.hard_shrink == True:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # [N,M]
            # normalize
            att_weight = F.normalize(att_weight, p=1, dim=1)  # [N,M]

        # out slot
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C] N是查询项的数目，C与记忆槽的维数相同

        return dict(out=out, att_weight=att_weight,mem=self.memMatrix)

        # return dict(output=out, att=att_weight)


class ML_MemAE_SC(nn.Module):
    def __init__(self, num_in_ch, features_root,
                 mem_dim, shrink_thres,
                 mem_usage, skip_ops, hard_shrink_opt):
        super(ML_MemAE_SC, self).__init__()
        self.num_in_ch = num_in_ch

        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.hard_shrink_opt = hard_shrink_opt
        self.mem_usage = mem_usage
        self.num_mem = sum(mem_usage)
        self.skip_ops = skip_ops


        self.in_conv = inconv(num_in_ch, features_root)         #double_conv
        self.down_1 = down(features_root, features_root * 2)                #Conv3d  double_conv
        self.down_2 = down(features_root * 2, features_root * 4)
        self.down_3 = down(features_root * 4, features_root * 8)

        # memory modules
        self.mem1 = Memory(mem_dim=self.mem_dim, fea_dim=features_root * 2,# * 8 * 128 *128,
                           shrink_thres=self.shrink_thres, hard_shrink=self.hard_shrink_opt) if self.mem_usage[1] else None
        self.mem2 = Memory(mem_dim=self.mem_dim, fea_dim=features_root * 4, #* 4 * 64 * 64,
                           shrink_thres=self.shrink_thres, hard_shrink=self.hard_shrink_opt) if self.mem_usage[2] else None
        self.mem3 = Memory(mem_dim=self.mem_dim, fea_dim=features_root * 8, #* 2 * 32 * 32,
                           shrink_thres=self.shrink_thres, hard_shrink=self.hard_shrink_opt) if self.mem_usage[3] else None
        self.mem3_ano = Memory(mem_dim=self.mem_dim, fea_dim=features_root * 8,  # * 2 * 32 * 32,
                           shrink_thres=self.shrink_thres, hard_shrink=self.hard_shrink_opt) if self.mem_usage[3] else None
        # self.memory = Memory_up(memory_size=10, feature_dim=256, key_dim=512, temp_update=0.1, temp_gather=0.1)

        self.up_3 = up(features_root * 8, features_root * 4, op=self.skip_ops[-1])
        self.up_2 = up(features_root * 4, features_root * 2, op=self.skip_ops[-2])
        self.up_1 = up(features_root * 2, features_root, op=self.skip_ops[-3])
        self.out_conv = outconv(features_root, num_in_ch)

    def forward(self, x, mem=True, mem_ano=False):
    # def forward(self, x,  train=True):
        """
        :param x: size [bs,C*seq_len,H,W]
        :return:
        """
        x0 = self.in_conv(x)     # x0: 2, 32, 16, 256, 256
        x1 = self.down_1(x0)     # x1: 2, 64, 8,  128, 128
        x2 = self.down_2(x1)     # x2: 2, 128,4,  64,  64
        x3 = self.down_3(x2)     # x3: 2, 256,2,  32,  32


        """if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                x3, keys, train)
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(x3, keys, train)"""

        if self.mem_usage[3] and mem:
        # if self.mem_usage[3] :
            # flatten [bs,C,H,W] --> [bs,C*H*W]
            bs, C, D, H, W = x3.shape
            x3 = x3.permute(0, 2, 3, 4, 1)
            x3 = x3.contiguous().view(-1, C)
            # x3 = x3.view(bs * C, -1)
            mem3_out = self.mem3(x3)
            x3 = mem3_out["out"]
            # attention weight size [bs,N], N is num_slots
            att_weight3 = mem3_out["att_weight"]
            # unflatten
            x3 = x3.view(bs, D, H, W, C)
            x3 = x3.permute(0, 4, 1, 2, 3)

        if self.mem_usage[3] and mem_ano:
        # if self.mem_usage[3] :
            # flatten [bs,C,H,W] --> [bs,C*H*W]
            bs, C, D, H, W = x3.shape
            x3 = x3.permute(0, 2, 3, 4, 1)
            x3 = x3.contiguous().view(-1, C)
            # x3 = x3.view(bs * C, -1)
            mem3_out_ano = self.mem3_ano(x3)
            x3 = mem3_out_ano["out"]
            # attention weight size [bs,N], N is num_slots
            att_weight3_ano = mem3_out_ano["att_weight"]
            # unflatten
            x3 = x3.view(bs, D, H, W, C)
            x3 = x3.permute(0, 4, 1, 2, 3)

        recon = self.up_3(x3, x2 if self.skip_ops[-1] != "none" else None)

        if self.mem_usage[2]:
            # pass through memory again
            bs, C, D, H, W = recon.shape
            recon = recon.permute(0, 2, 3, 4, 1)
            recon = recon.contiguous().view(-1, C)
            # recon = recon.view(bs, -1)
            mem2_out = self.mem2(recon)
            recon = mem2_out["out"]
            att_weight2 = mem2_out["att_weight"]
            # recon = recon.view(bs, C, D, H, W)
            recon = recon.view(bs, D, H, W, C)
            recon = recon.permute(0, 4, 1, 2, 3)

        recon = self.up_2(recon, x1 if self.skip_ops[-2] != "none" else None)

        if self.mem_usage[1]:
            # pass through memory again
            # recon = recon.cpu()
            # self.mem1.cpu()
            bs, C, D, H, W = recon.shape
            # recon = recon.view(bs, -1)
            recon = recon.permute(0, 2, 3, 4, 1)
            recon = recon.contiguous().view(-1, C)
            mem1_out = self.mem1(recon)
            recon = mem1_out["out"]
            att_weight1 = mem1_out["att_weight"]
            # recon = recon.view(bs, C, D, H, W)
            recon = recon.view(bs, D, H, W, C)
            recon = recon.permute(0, 4, 1, 2, 3)
            # recon = recon.cuda()


        recon = self.up_1(recon, x0 if self.skip_ops[-3] != "none" else None)
        recon = self.out_conv(recon)  # Conv3d



        """if self.num_mem == 3:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=att_weight2, att_weight1=att_weight1)
        if self.num_mem == 2:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=att_weight2,
                        att_weight1=torch.zeros_like(att_weight3))"""  # dummy attention weights
        if self.num_mem == 1 and mem_ano:
            outs = dict(recon=recon, att_weight3=torch.zeros(1, 1), att_weight3_ano=att_weight3_ano,
                        mem=torch.zeros(1, 1), mem_ano=mem3_out_ano['mem'])  # dummy attention weights
        if self.num_mem == 1 and mem:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight3_ano=torch.zeros(1, 1),
                        mem=mem3_out['mem'], mem_ano=torch.zeros(1, 1))  # dummy attention weights

        if mem == False and mem_ano == False:
            outs = dict(recon=recon, att_weight3=torch.zeros(1, 1), att_weight3_ano=torch.zeros(1, 1),
                        mem=torch.zeros(1, 1), mem_ano=torch.zeros(1, 1))  # dummy attention weights


        if self.num_mem == 0:
            outs = dict(recon=recon, att_weight3=torch.zeros(1, 1),
                        att_weight2=torch.zeros(1, 1),
                        att_weight1=torch.zeros(1, 1))
        return outs



        """if train:
            return recon, x3, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss
        else:
            return recon, x3, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss"""



    """if __name__ == '__main__':
        model = ML_MemAE_SC(num_in_ch=2, seq_len=1, features_root=32, num_slots=2000, shrink_thres=1 / 2000,
                            mem_usage=[False, True, True, True], skip_ops=["none", "concat", "concat"])
        dummy_x = torch.rand(4, 2, 32, 32)  # 均匀分布
        dummy_out = model(dummy_x)
        # print(dummy_x.shape)
        print(-1)"""



class d_net(nn.Module):
    def __init__(self, channel):
        super(d_net, self).__init__()
        """num_in_ch = channel
        features_root = 32
        self.in_conv = inconv(num_in_ch, features_root)  # double_conv
        self.down_1 = down(features_root, features_root * 2)  # Conv3d  double_conv
        self.down_2 = down(features_root * 2, features_root * 4)
        self.down_3 = down(features_root * 4, features_root * 8)"""


        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.layer1 = nn.Sequential(

            nn.Conv3d(channel, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)

            # nn.ReLU(True),
            # nn.Linear(32768, 8192),
            # nn.ReLU(True),
            # nn.Linear(8192, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 512),
            # nn.ReLU(True),
            # nn.Linear(512, 1)
            # nn.ReLU(True)
            # nn.Linear(128, 1)
            # nn.ReLU(True),

        )
        self.layer2 = nn.Sequential(nn.Flatten(),
            nn.Linear(131072, 1),
            # nn.Linear(524288, 1),                      # 2, 256,2,  32,  32
            nn.Sigmoid())

    def forward(self, x):
        # x0 = self.in_conv(x)  # x0: 2, 32, 16, 256, 256
        # x1 = self.down_1(x0)  # x1: 2, 64, 8,  128, 128
        # x2 = self.down_2(x1)  # x2: 2, 128,4,  64,  64
        # x3 = self.down_3(x2)  # x3: 2, 256,2,  32,  32
        feature = self.layer1(x)
        output = self.layer2(feature)
        # x = torch.abs(x)
        return output
        # return out = dict(feature=feature, output=output)

class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, hard_shrink_opt=True, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.hard_shrink_opt = hard_shrink_opt
        self.memory = Memory(mem_dim=self.mem_dim, fea_dim=self.fea_dim,
                             hard_shrink=self.hard_shrink_opt)

    def forward(self, input):
        s = input.data.shape  #计算维数
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])   #转为一维
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att}
