import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from models.fpn import FPN50


url_resnet_50 = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


class DepthGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthGCNLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer_self = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, bias=True)
        self.layer_others = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, att):
        """
        x: [B, J, C]
        att: [B, J, J]
        """
        n_joints = x.size(1)
        x = x.transpose(1, 2)  # [B, C, J]

        feat_self = self.layer_self(x)  # [B, C, J]
        feat_self = feat_self.transpose(1, 2)  # [B, J, C]

        feat_others = self.layer_others(x)  # [B, C, J]
        feat_others = feat_others.transpose(1, 2)  # [B, J, C]

        diag = torch.eye(n_joints, dtype=torch.float).to(x.device)
        out = torch.matmul(att * diag, feat_self) + torch.matmul(att * (1 - diag), feat_others)  # [B, J, C]

        out = out.transpose(1, 2)  # [B, C, J]
        out = self.bn(out)
        out = self.relu(out)
        out = out.transpose(1, 2)  # [B, J, C]

        return out


class DepthGCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, pool=None):
        super(DepthGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.pool = pool

        layers = []
        in_ch = self.in_channels
        for l in range(self.n_layers):
            layers.append(DepthGCNLayer(in_ch, self.out_channels))
            in_ch = self.out_channels
        self.gcn_layers = nn.ModuleList(layers)

    def forward(self, x, att):
        """
        x: [B, J, C]
        att: [B, J, J]
        """
        out = x

        for l in range(self.n_layers):
            out = self.gcn_layers[l](out, att)

        if self.pool is None:
            return out
        elif self.pool == "max":
            return torch.max(out, dim=1)
        elif self.pool == "avg":
            return torch.mean(out, dim=1)
        else:
            raise NotImplementedError


class RootNet(nn.Module):

    def __init__(self, n_joints, n_bins, adj):
        super(RootNet, self).__init__()

        self.n_joints = n_joints
        self.n_bins = n_bins
        self.adj = torch.as_tensor(adj).float()  # [J, J]

        self.fpn = FPN50()

        self.layers_hm_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_hm_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_hm_4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_hm_5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_dm_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_dm_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_dm_4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_dm_5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.out_heatmap = nn.Conv2d(256, self.n_joints, kernel_size=1, bias=True)

        self.depth_gcn = DepthGCN(in_channels=256, out_channels=256, n_layers=2, pool="avg")

        self.out_bins = nn.Linear(256, self.n_bins, bias=True)

        # initialization
        self.fpn.resnet.load_state_dict(model_zoo.load_url(url_resnet_50), strict=False)

    def freeze_resnet(self):
        self.fpn.freeze_resnet()

    def forward(self, x, coord_map, bbox_masks, vis, epoch, target=None):
        """
        x: [B, 3, H_in, W_in]. Optical axis centered input image.
        coord_map: [B, 2, H_out, W_out]. Normalized image coordinates with focal length fx, fy divided from original image coordinates
        bbox_masks: [B, H_out, W_out]. {0, 1} mask for each person (up to MAXP) in the image.
        vis: [B, J]. {0, 1} indicator of each person's each joint's visibility.
        """
        batch_size = x.size(0)

        fp2, fp3, fp4, fp5 = self.fpn(x)

        # ====================
        hm_branch_2 = self.layers_hm_2(fp2)
        hm_branch_3 = self.layers_hm_3(fp3)
        hm_branch_4 = self.layers_hm_4(fp4)
        hm_branch_5 = self.layers_hm_5(fp5)

        dm_branch_2 = self.layers_dm_2(fp2)
        dm_branch_3 = self.layers_dm_3(fp3)
        dm_branch_4 = self.layers_dm_4(fp4)
        dm_branch_5 = self.layers_dm_5(fp5)

        hm_branch = torch.cat([hm_branch_2, hm_branch_3, hm_branch_4, hm_branch_5], dim=1)  # [B, 4C, H, W]
        dm_branch = torch.cat([dm_branch_2, dm_branch_3, dm_branch_4, dm_branch_5], dim=1)  # [B, 4C, H, W]

        # ====================
        # heatmap regression
        hm = self.out_heatmap(hm_branch)  # [B, J, H, W]
        _, _, H, W = hm.size()

        repeated_bbox_masks = bbox_masks.view(batch_size, 1, H, W).repeat(1, self.n_joints, 1, 1)  # [B, J, H, W]
        masked_hm = repeated_bbox_masks * hm + (1 - repeated_bbox_masks) * -1e15  # [B, J, H, W]

        hm_flatten = masked_hm.view(batch_size, self.n_joints, -1)  # [B, J, HW]
        normalized_hm = F.softmax(hm_flatten, dim=-1)
        normalized_hm = normalized_hm.view(batch_size, self.n_joints, H, W)  # [B, J, H, W]

        # pose pooling
        repeated_hm_2 = torch.repeat_interleave(normalized_hm, 2, dim=1)  # [B, 1122...JJ, H, W]
        repeated_coord_map = coord_map.repeat(1, self.n_joints, 1, 1)  # [B, xyxy...xy, H, W]
        pred_pose = torch.sum(repeated_hm_2 * repeated_coord_map, dim=[-2, -1])  # [B, xyxy...xy]

        # depth feature pooling
        repeated_hm_C = torch.repeat_interleave(normalized_hm, dm_branch.size(1), dim=1)  # [B, 11111...JJJJJ, H, W]
        repeated_dm = dm_branch.repeat(1, self.n_joints, 1, 1)  # [B, dmdm...dm, H, W]
        depth_feat = torch.sum(repeated_hm_C * repeated_dm, dim=[-2, -1])  # [B, dmdm...dm]
        depth_feat = depth_feat.view(batch_size, self.n_joints, -1)  # [B, J, 4C+3]

        # graph convolution
        adj = self.adj.view(1, self.n_joints, self.n_joints).repeat(batch_size, 1, 1)  # [B, J, J]
        adj = adj.to(x.device)
        att = adj
        att = att / torch.sum(att, dim=-1, keepdim=True)  # L1 normalization on last dimension

        depth_feat = self.depth_gcn(depth_feat, att)  # [B, C]

        bins_pred = self.out_bins(depth_feat)  # [B, K+1]
        bins_pred = F.softmax(bins_pred, dim=-1)

        # bins to bin_idx
        bin_idx_pred = torch.sum(bins_pred * torch.arange(self.n_bins).float().to(bins_pred.device), dim=-1)  # [B]

        out = {
            "heatmap": normalized_hm,  # [B, J, H, W]
            "pose": pred_pose,  # [B, J*2]
            "bins": bins_pred,  # [B, K+1]
            "bin_idx": bin_idx_pred,  # [B]
        }

        if target is None:
            return out, None
        else:
            """
            heatmap_gt: [B, J, H, W]
            pose_gt: [B, xyxy...xy]
            bins_gt: [B, K+1]
            bin_idx_gt: [B]
            """
            heatmap_gt, pose_gt, bins_gt, bin_idx_gt = target

            # compute loss
            # 1. heatmap
            # 2. 2d pose
            # 3. bins
            # 4. bin_idx
            loss_weights = {
                "heatmap": 10.0 * (0.5 ** (epoch / 10)),
                "pose": 1.0,
                "bins": 1e-3 * (0.5 ** (epoch / 10)),
                "bin_idx": 1.0,
            }

            loss_heatmap = get_heatmap_loss(heatmap_gt, normalized_hm, vis, loss_type="L2")
            loss_pose = get_pose_loss(pose_gt, pred_pose, vis, loss_type="L1")
            loss_bins = get_bins_loss(bins_gt, bins_pred, loss_type="CE")
            loss_bin_idx = get_bin_idx_loss(bin_idx_gt / (self.n_bins - 1), bin_idx_pred / (self.n_bins - 1), loss_type="L1")

            loss = loss_heatmap * loss_weights["heatmap"] + \
                loss_pose * loss_weights["pose"] + \
                loss_bins * loss_weights["bins"] + \
                loss_bin_idx * loss_weights["bin_idx"]

            return out, {
                "tot_loss": loss,
                "heatmap": loss_heatmap,
                "pose": loss_pose,
                "bins": loss_bins,
                "bin_idx": loss_bin_idx,
                "weights": loss_weights,
            }


def get_heatmap_loss(gt_hm, pred_hm, vis, loss_type):
    """
    gt_hm: [B, J, H, W]
    pred_hm: [B, J, H, W]
    vis: [B, J]

    Loss: L2 (MSE), or L1
    """
    if loss_type == "L2":
        loss = torch.mean((gt_hm - pred_hm) ** 2, dim=(-2, -1))  # [B, J]
        loss = torch.sum(loss * vis, dim=-1) / torch.sum(vis, dim=-1)  # [B]
    elif loss_type == "L1":
        loss = torch.mean(torch.abs(gt_hm - pred_hm), dim=(-2, -1))  # [B, J]
        loss = torch.sum(loss * vis, dim=-1) / torch.sum(vis, dim=-1)  # [B]
    else:
        assert False, "Unknown loss type"

    return loss


def get_pose_loss(gt_pose, pred_pose, vis, loss_type):
    """
    gt_pose: [B, J*2]
    pred_pose: [B, J*2]
    vis: [B, J]

    Loss: L2 (MSE), or L1
    """
    vis = torch.repeat_interleave(vis, 2, dim=-1)  # [B, 1122...JJ]

    if loss_type == "L2":
        loss = torch.sum((gt_pose - pred_pose) ** 2 * vis, dim=-1) / torch.sum(vis, dim=-1)  # [B]
    elif loss_type == "L1":
        loss = torch.sum(torch.abs(gt_pose - pred_pose) * vis, dim=-1) / torch.sum(vis, dim=-1)  # [B]
    else:
        assert False, "Unknown loss type"

    return loss


def get_bins_loss(gt_bins, pred_bins, loss_type):
    """
    gt_bins: [B, K+1]
    pred_bins: [B, K+1]

    compute cross entropy loss between gt and pred bins
    Loss: Cross entropy (CE), or L1, or L2
    """
    if loss_type == "CE":
        loss = -1 * torch.sum(gt_bins * torch.log(pred_bins + 1e-8), dim=-1)  # [B]
    elif loss_type == "L1":
        loss = torch.mean(torch.abs(gt_bins - pred_bins), dim=-1)  # [B]
    elif loss_type == "L2":
        loss = torch.mean((gt_bins - pred_bins) ** 2, dim=-1)  # [B]
    else:
        assert False, "Unknown loss type"

    return loss


def get_bin_idx_loss(gt_bin_idx, pred_bin_idx, loss_type):
    """
    gt_bin_idx: [B]
    pred_bin_idx: [B]

    Loss: L2 (MSE), or L1
    """
    if loss_type == "L2":
        loss = (gt_bin_idx - pred_bin_idx) ** 2  # [B]
    elif loss_type == "L1":
        loss = torch.abs(gt_bin_idx - pred_bin_idx)  # [B]
    else:
        assert False, "Unknown loss type"

    return loss


def get_model(n_joints, n_bins, adj):
    return RootNet(n_joints, n_bins, adj)
