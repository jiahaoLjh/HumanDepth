import logging
import copy

import cv2
import numpy as np
import torch
import torch.utils.data


from config import cfg


class Dataset(torch.utils.data.Dataset):

    def __init__(self, db):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.db = db
        self.mode = db.mode
        self.n_joints = db.n_joints
        self.joint_name = db.joint_name

        self.bin_start = db.bin_start
        self.bin_end = db.bin_end
        self.n_bins = db.n_bins

        self.adj = db.adj

        self.data = db.get_data()

    def evaluate(self, preds):
        return self.db.evaluate(preds)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])
        img_path = data["image_path"]
        f = data["f"]
        c = data["c"]
        joints_2d = data["joints_2d"]  # [J, 2]
        bins = data["bins"]            # [K]
        bin_idx = data["bin_idx"]      # []

        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read {}".format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channels = img.shape

        scale, rot, do_flip = 1.0, 0.0, False

        # transform image
        cx, cy, pw, ph = c[0], c[1], data["img_effective_width"], data["img_effective_height"]

        mat_input = get_trans_from_patch((cx, cy), (pw, ph), cfg.input_size, do_flip, scale, rot)
        trans_img = cv2.warpAffine(img, mat_input, cfg.input_size, flags=cv2.INTER_LINEAR)

        mat_output = get_trans_from_patch((cx, cy), (pw, ph), cfg.output_size, do_flip, scale, rot)

        # transform coord_map
        coord_map = np.stack(np.meshgrid(np.arange(pw), np.arange(ph)), axis=-1).astype(np.float32)  # [H, W, 2]
        coord_map = (coord_map - np.array([pw / 2., ph / 2.])) / f
        mat_coord = get_trans_from_patch((pw / 2., ph / 2.), (pw, ph), cfg.output_size, do_flip, scale, rot)
        trans_coord_map = cv2.warpAffine(coord_map, mat_coord, cfg.output_size, flags=cv2.INTER_LINEAR)

        # transform 2d joints to output space
        trans_joints_2d = np.zeros_like(joints_2d)
        for j in range(self.n_joints):
            trans_joints_2d[j] = trans_point2d(joints_2d[j], mat_output)

        # generate heatmap
        trans_heatmap = get_heatmap(trans_joints_2d, cfg.output_size, sigma=cfg.hm_sigma)  # [H, W, J]

        # bbox mask
        bbox_mask = np.ones([cfg.output_size[0], cfg.output_size[1]])

        # visibility mask
        joints_vis = np.ones([self.n_joints])

        # transform 2d pose
        trans_pose = (joints_2d - c) / f

        out_img = resnet_normalize(trans_img)
        out_img = torch.as_tensor(out_img.transpose((2, 0, 1))).float()
        out_coord_map = torch.as_tensor(trans_coord_map.transpose((2, 0, 1))).float()
        out_heatmap = torch.as_tensor(trans_heatmap.transpose((2, 0, 1))).float()
        out_pose = torch.as_tensor(trans_pose.reshape([-1])).float()
        out_bins = torch.as_tensor(bins).float()
        out_bin_idx = torch.as_tensor(bin_idx).float()
        out_bbox_mask = torch.as_tensor(bbox_mask).float()
        out_vis_mask = torch.as_tensor(joints_vis).float()

        return {
            "img": out_img,  # [3, H, W]
            "coord_map": out_coord_map,  # [2, H, W]
            "heatmap": out_heatmap,  # [J, H, W]
            "pose": out_pose,  # [J*2]
            "bins": out_bins,  # [K+1]
            "bin_idx": out_bin_idx,  # []
            "bbox_mask": out_bbox_mask,  # [H, W]
            "vis_mask": out_vis_mask,  # [J]
        }


def get_heatmap(joints_2d, output_size, sigma):
    """
    joints_2d: [J, 2] in output space
    output_size: (H, W) of output space
    """
    H, W = output_size
    n_joints, _ = joints_2d.shape

    hm = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(np.float32)  # [H, W, 2]
    hm = hm.reshape([H, W, 1, 2]).repeat(n_joints, axis=2)  # [H, W, J, 2]

    j2d = joints_2d.reshape([1, 1, n_joints, 2])  # [1, 1, J, 2]
    j2d = j2d.repeat(H, axis=0).repeat(W, axis=1)  # [H, W, J, 2]

    # prob = exp(-(dx * dx + dy * dy) / 2)
    hm = np.exp(-1 * np.sum((hm - j2d) ** 2, axis=-1) / (2 * sigma ** 2))  # [H, W, J]
    hm = hm / (np.sum(hm, axis=(0, 1)) + 1e-10)

    return hm


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def get_trans_from_patch(center, input_size, output_size, do_flip, scale, rot, inv=False):
    c_x, c_y = center
    src_width, src_height = input_size
    dst_width, dst_height = output_size

    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    if do_flip:
        src_rightdir = rotate_2d(np.array([src_w * -0.5, 0], dtype=np.float32), rot_rad)
    else:
        src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def resnet_normalize(image):
    image = image.astype(np.float32) / 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()
    for i in range(3):
        preprocessed_img[:, :, i] = (preprocessed_img[:, :, i] - mean[i]) / std[i]

    return preprocessed_img
