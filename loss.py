import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from torch import linalg as LA

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def SM(Xs,Xt,Ys,Yt,Cs_memory,Ct_memory,Wt=None,decay=0.3):
    Cs=Cs_memory.clone()
    Ct=Ct_memory.clone()
    K=Cs.size(0)
    for k in range(K):
        dec_s = True
        dec_t = True
        Xs_k=Xs[Ys==k]
        Xt_k=Xt[Yt==k]
        if len(Xs_k)==0:
            Cs_k=0.0
            dec_s = False
        else:
            Cs_k=torch.mean(Xs_k,dim=0)

        if len(Xt_k)==0:
            Ct_k=0.0
            dec_t = False
        else:
            Ct_k=torch.mean(Xt_k,dim=0)
        if(dec_s):
            Cs[k,:]=(1-decay)*Cs_memory[k,:]+decay*Cs_k
        if(dec_t):
            Ct[k,:]=(1-decay)*Ct_memory[k,:]+decay*Ct_k
    inter_dist=0
    intra_dist_ss = 0
    intra_dist_st = 0
    intra_dist_tt = 0
    Cs_norm = torch.nn.functional.normalize(Cs)
    Ct_norm = torch.nn.functional.normalize(Ct)
    for k_i in range(K):
        for k_j in range(K):
            if k_i != k_j:
                intra_dist_st += torch.dot(Cs_norm[k_i], Ct_norm[k_j])
                intra_dist_ss += torch.dot(Cs_norm[k_i], Cs_norm[k_j])
                intra_dist_tt += torch.dot(Ct_norm[k_i], Ct_norm[k_j])
            else:
                inter_dist += torch.dot(Cs[k_i]-Ct[k_j], Cs[k_i]-Ct[k_j])
    intra_dist_st = intra_dist_st / ((K - 1) * (K - 1))
    intra_dist_ss = intra_dist_ss / ((K - 1) * (K - 1))
    intra_dist_tt = intra_dist_tt / ((K - 1) * (K - 1))
    inter_dist = inter_dist / K
    return inter_dist + intra_dist_st + intra_dist_tt, intra_dist_ss, Cs, Ct