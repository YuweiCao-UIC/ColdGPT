import torch
import torch.nn as nn
import torch.nn.functional as F

def InfoNCE_loss(view1, view2, temperature):
    out_1 = F.normalize(view1)
    out_2 = F.normalize(view2)

    out = torch.cat([out_1, out_2], dim=0)
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    batch_size = out_1.size(0)
    neg_sim = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(neg_sim) - torch.eye(2 * batch_size, device=neg_sim.device)).bool()
            # [2*B, 2*B-1]
    neg_sim = neg_sim.masked_select(mask).view(2 * batch_size, -1).sum(1)
    pos_loss = -torch.log(pos_sim)
    neg_loss = torch.log(neg_sim)
    return (pos_loss + neg_loss).mean()
    #return pos_loss.mean()