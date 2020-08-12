import os.path as osp
import h5py
import logging

import tqdm
import numpy as np


class Human36M:

    def __init__(self, mode):
        self.logger = logging.getLogger(self.__class__.__name__)

        assert mode in ["train", "test"], "Invalid mode {}".format(mode)
        self.mode = mode

        self.data_path = osp.join("data", "Human36M")
        self.annot_path = osp.join(self.data_path, "annotations")
        self.root_image_path = osp.join(self.data_path, "images", self.mode)

        self.n_joints = 17
        self.joint_name = [
            "Hip",
            "RHip",
            "RKnee",
            "RFoot",
            "LHip",
            "LKnee",
            "LFoot",
            "Spine",
            "Thorax",
            "Neck/Nose",
            "Head",
            "LShoulder",
            "LElbow",
            "LWrist",
            "RShoulder",
            "RElbow",
            "RWrist",
        ]
        self.action_name = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Posing", "Purchases", "Sitting", "SittingDown", "Smoking", "Photo", "Waiting", "Walking", "WalkDog", "WalkTogether"]

        self.bin_start = 2.0
        self.bin_end = 8.0
        self.n_bins = 61
        self.img_effective_width = 1000
        self.img_effective_height = 1000

        self.adj = np.eye(self.n_joints)
        self.kinematic_links = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
        for j1, j2 in self.kinematic_links:
            self.adj[j1, j2] = 1
            self.adj[j2, j1] = 1

        self.data = self._load_h5py_data()

    def get_data(self):
        return self.data

    def _get_subsampling_rate(self):
        if self.mode == "train":
            return 5
        elif self.mode == "test":
            return 64
        else:
            raise Exception("Invalid mode")

    def _get_subject(self):
        if self.mode == "train":
            return [1, 5, 6, 7, 8]
        elif self.mode == "test":
            return [9, 11]
        else:
            raise Exception("Invalid mode")

    def _get_name(self, subject, action, subaction, camera_id):
        name = "s_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}".format(subject, action, subaction, camera_id)
        return name

    def _load_h5py_data(self):
        self.logger.info("Loading Human3.6m data...")

        subjects = self._get_subject()
        subsampling_rate = self._get_subsampling_rate()

        seqs = dict()

        for sub in subjects:
            self.logger.info("Loading data for subject {}...".format(sub))
            with h5py.File(osp.join(self.annot_path, "subject_{}.h5".format(sub)), "r") as f:
                self.logger.info("{} sequences loaded".format(len(f)))
                for sid in range(len(f)):
                    subject = f["{}".format(sid)].attrs["subject"]
                    action = f["{}".format(sid)].attrs["action"]
                    subaction = f["{}".format(sid)].attrs["subaction"]
                    action_name = f["{}".format(sid)].attrs["action_name"]
                    camera_id = f["{}".format(sid)].attrs["camera_id"]
                    seqs[subject, action, subaction, camera_id] = {
                        "f": f["{}/f".format(sid)][:],
                        "c": f["{}/c".format(sid)][:],
                        "joints_2d": f["{}/joints_2d".format(sid)][:],
                        "joints_3d": f["{}/joints_3d".format(sid)][:],
                        "meta": {
                            "subject": subject,
                            "action": action,
                            "subaction": subaction,
                            "action_name": action_name,
                            "camera_id": camera_id,
                        },
                    }
            with h5py.File(osp.join(self.annot_path, "subject_{}_cpn.h5".format(sub)), "r") as f:
                self.logger.info("{} CPN sequences loaded".format(len(f)))
                for sid in range(len(seqs)):
                    if str(sid) not in f:
                        continue
                    subject = f["{}".format(sid)].attrs["subject"]
                    action = f["{}".format(sid)].attrs["action"]
                    subaction = f["{}".format(sid)].attrs["subaction"]
                    camera_id = f["{}".format(sid)].attrs["camera_id"]
                    assert (subject, action, subaction, camera_id) in seqs
                    seqs[subject, action, subaction, camera_id]["joints_2d_cpn"] = f["{}/joints_2d_cpn".format(sid)][:]

        data = []
        target_bin_idx = []
        for seq_info in tqdm.tqdm(seqs, total=len(seqs), ncols=100):
            subject, action, subaction, camera_id = seq_info
            seq = seqs[seq_info]

            # one video is missing
            # if subject == 11 and action == 2 and subaction == 2 and camera_id == 1:
            if subject == 11 and action == 2 and subaction == 2:
                continue

            assert seq["f"].shape[0] == seq["c"].shape[0] == seq["joints_2d"].shape[0] == seq["joints_3d"].shape[0], seq["meta"]
            n_frames = seq["joints_2d"].shape[0]
            f = seq["f"][0].reshape([2])  # [2]
            c = seq["c"][0].reshape([2])  # [2]
            joints_2d = seq["joints_2d"].reshape([n_frames, self.n_joints, 2])  # [F, J, 2]
            joints_3d = seq["joints_3d"].reshape([n_frames, self.n_joints, 3])  # [F, J, 3]
            joints_2d_cpn = seq["joints_2d_cpn"][:n_frames, :, :]

            for frame_id in range(0, n_frames, subsampling_rate):
                j2d = joints_2d[frame_id]  # [J, 2]
                j3d = joints_3d[frame_id]  # [J, 3]
                j2d_cpn = joints_2d_cpn[frame_id]  # [J, 2]

                depth_root = j3d[0, 2] / np.sqrt(f[0] * f[1])
                bin_idx = (np.log(depth_root) - np.log(self.bin_start)) / (np.log(self.bin_end) - np.log(self.bin_start)) * (self.n_bins - 1)
                bins = 1.0 - np.clip(np.abs(np.arange(self.n_bins) - bin_idx), 0, 1)

                name = self._get_name(subject, action, subaction, camera_id)
                data.append({
                    "meta": {
                        "subject": subject,
                        "action": action,
                        "subaction": subaction,
                        "action_name": action_name,
                        "camera_id": camera_id,
                        "frame_id": frame_id + 1,
                    },
                    "image_path": osp.join(self.root_image_path, name, "{}_{:06d}.jpg".format(name, frame_id + 1)),
                    "f": f,
                    "c": c,
                    "img_effective_width": self.img_effective_width,
                    "img_effective_height": self.img_effective_height,
                    "joints_2d": j2d,
                    "joints_3d": j3d,
                    "joints_2d_cpn": j2d_cpn,
                    "bins": bins,
                    "bin_idx": bin_idx,
                })

                target_bin_idx.append(bin_idx)

        self.logger.info("{} images loaded".format(len(data)))

        targets = np.array(target_bin_idx)
        self.logger.info("Bin idx max = {}, bin idx min = {}".format(np.max(targets), np.min(targets)))

        return data

    def evaluate(self, preds):
        self.logger.info("Evaluating...")

        gts = self.data
        preds_pose, preds_bin_idx = preds
        n_frames = len(gts)
        assert len(preds_pose) == n_frames
        assert len(preds_bin_idx) == n_frames

        error_pose = np.zeros([n_frames, self.n_joints, 2])  # 2d pose error
        error_mrpe = np.zeros([n_frames, 3])  # MRPE

        for i in range(n_frames):
            gt = gts[i]
            f = gt["f"]  # [2]
            c = gt["c"]  # [2]
            j2d = gt["joints_2d"]  # [J, 2]
            j3d = gt["joints_3d"]  # [J, 3]
            j2d_cpn = gt["joints_2d_cpn"]  # [J, 2]

            # 2d pose error
            pred_pose = preds_pose[i]  # [2J]
            pred_pose = pred_pose.reshape([self.n_joints, 2])  # [J, 2]
            pred_pose = pred_pose * f + c

            err_pose = np.abs(j2d - pred_pose)  # [J, 2]
            error_pose[i] = err_pose

            # mrpe
            pred_bin_idx = preds_bin_idx[i]  # []
            pred_depth = pred_bin_idx / (self.n_bins - 1) * (np.log(self.bin_end) - np.log(self.bin_start)) + np.log(self.bin_start)
            pred_depth = np.exp(pred_depth) * np.sqrt(f[0] * f[1])

            # cpn root
            x, y = (j2d_cpn[0] - c) / f * pred_depth
            error_mrpe[i] = np.abs(j3d[0] - np.array([x, y, pred_depth]))  # [3]

        avg_error_pose = np.mean(np.sqrt(np.sum(error_pose ** 2, axis=-1)))
        avg_error_pose_u = np.mean(error_pose[:, :, 0])
        avg_error_pose_v = np.mean(error_pose[:, :, 1])
        self.logger.info("Pose error: {:.2f}. dU = {:.2f}, dV = {:.2f}".format(avg_error_pose, avg_error_pose_u, avg_error_pose_v))
        avg_error_pose_root = np.mean(np.sqrt(np.sum(error_pose[:, 0, :] ** 2, axis=-1)))
        avg_error_pose_u_root = np.mean(error_pose[:, 0, 0])
        avg_error_pose_v_root = np.mean(error_pose[:, 0, 1])
        self.logger.info("Root error: {:.2f}. dU = {:.2f}, dV = {:.2f}".format(avg_error_pose_root, avg_error_pose_u_root, avg_error_pose_v_root))

        avg_error_mrpe = np.mean(np.sqrt(np.sum(error_mrpe ** 2, axis=-1)))
        avg_error_mrpe_x = np.mean(error_mrpe[:, 0])
        avg_error_mrpe_y = np.mean(error_mrpe[:, 1])
        avg_error_mrpe_z = np.mean(error_mrpe[:, 2])

        self.logger.info("MRPE = {:.2f}".format(avg_error_mrpe))
        self.logger.info("MRPE x = {:.2f}".format(avg_error_mrpe_x))
        self.logger.info("MRPE y = {:.2f}".format(avg_error_mrpe_y))
        self.logger.info("MRPE z = {:.2f}".format(avg_error_mrpe_z))

        return avg_error_pose, avg_error_pose_root, avg_error_mrpe_x, avg_error_mrpe_y, avg_error_mrpe_z, avg_error_mrpe
