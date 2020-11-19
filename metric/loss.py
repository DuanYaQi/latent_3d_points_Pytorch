from metric.chamfer_distance import chamfer_distance
import torch.nn as nn
import torch

# -----------------------------------------------------------------------------------------
class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, gt, pred):
        cost_for, cost_bac = chamfer_distance(gt, pred)
        loss = (torch.mean(cost_for)) + (torch.mean(cost_bac))
        return loss

    def forward1(self, pred, gt, pcd_radius=1.0):
        cost_for, cost_bac = chamfer_distance(gt, pred)
        cost = 0.5 * cost_for + 0.5 * cost_bac
        cost /= pcd_radius.view(-1, 1)
        cost = torch.mean(cost)
        return cost

    def forward2(self, x, y):
        """
        :param x: (bs, np, 3)
        :param y: (bs, np, 3)
        :return: loss
        """
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        dist = torch.sqrt(1e-6 + torch.sum(torch.pow(x - y, 2), 3))  # bs, ny, nx
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)
        return min1.mean(dim=-1) + min2.mean(dim=-1)