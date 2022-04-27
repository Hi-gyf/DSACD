import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F


class SACD(nn.Module):
    def __init__(self, n_stu, n_pro, n_know, hid1, hid2, step):
        super(SACD, self).__init__()
        self.skill = nn.Parameter(torch.rand(n_stu, n_know))
        self.ability = nn.Parameter(torch.rand(n_stu, n_know))
        self.speed = nn.Parameter(torch.rand(n_stu, step, n_know))
        self.difficulty = nn.Parameter(torch.rand(n_pro, n_know))
        self.workload = nn.Parameter(torch.rand(n_pro, n_know))
        self.disc = nn.Parameter(torch.rand(n_pro, 1))

        self.speed_modeling = SpeedModeling(n_know, step)
        self.factor_fusion = FactorFusion(n_know)

        self.predictor = nn.Sequential(
            nn.Linear(3 * n_know, hid1),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(hid1, hid2),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(hid2, 1),
            nn.Sigmoid()
        )

        for name, param in self.named_parameters():
            if 'weight' in name or len(param.size()) >= 2:
                nn.init.xavier_normal_(param)

    def forward(self, sid, pid, Q, rt, device, test=False):
        skill = torch.sigmoid(self.skill[sid])
        ability = torch.sigmoid(self.ability[sid])
        speed = torch.sigmoid(self.speed[sid])
        diff = torch.sigmoid(self.difficulty[pid])
        workload = torch.sigmoid(self.workload[pid])
        disc = torch.sigmoid(self.disc[pid])

        skill_factor = skill * Q
        ability_factor = disc * (ability - diff)
        speed_mat, speed_factor = self.speed_modeling(speed, workload.unsqueeze(dim=1), rt.unsqueeze(dim=1), device)

        x = self.factor_fusion(speed_mat, speed_factor, skill_factor, ability_factor, test)
        output = self.predictor(x)
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        for module in self.predictor:
            module.apply(clipper)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class FactorFusion(nn.Module):
    def __init__(self, n_know):
        super(FactorFusion, self).__init__()
        self.ability_layer = nn.Linear(n_know, n_know)
        self.skill_layer = nn.Linear(n_know, n_know)
        self.speed_layer = nn.Linear(n_know, n_know)

        self.as_att = DotAttention(n_know)
        self.ss_att = DotAttention(n_know)

    def forward(self, speed_mat, speed, skill, ability, test):
        speed = speed.float().unsqueeze(dim=1)
        skill = skill.float().unsqueeze(dim=1)
        ability = ability.float().unsqueeze(dim=1)

        skill_l = self.skill_layer(skill)
        ability_l = self.ability_layer(ability)
        speed_l = speed

        s_s, skill_f = self.as_att(skill_l, speed_mat, speed_mat)
        a_s, ability_f = self.ss_att(ability_l, speed_mat, speed_mat)

        x = torch.cat((skill_f + skill_l, ability_f + ability_l, speed_l), 2).float().squeeze(dim=1)
        return x


class DotAttention(nn.Module):
    def __init__(self, scale):
        super(DotAttention, self).__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))
        u = u / self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        output = torch.bmm(attn, v)
        return attn, output


class SpeedModeling(nn.Module):
    def __init__(self, n_know, step):
        super(SpeedModeling, self).__init__()
        self.n_know = n_know
        self.speed_matching = SpeedMatching(n_know, step)
        self.lstm = nn.LSTM(n_know, n_know, 1, batch_first=True)

    def forward(self, speed, workload, rt, device):
        _, sm = self.speed_matching(workload, speed, speed)
        bs = speed.size()[0]

        h0 = torch.randn(1, bs, self.n_know).to(device)
        c0 = torch.randn(1, bs, self.n_know).to(device)
        lm, (hn, cn) = self.lstm(sm, (h0, c0))

        st = hn.squeeze(dim=0) + torch.log(rt)
        return lm, st - workload.squeeze(dim=1)


class SpeedMatching(nn.Module):
    def __init__(self, scale, step):
        super(SpeedMatching, self).__init__()
        self.scale = scale
        self.step = step
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))
        u = u / self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        output = attn.view(-1, self.step, 1) * v
        return attn, output