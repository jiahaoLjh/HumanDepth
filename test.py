import os.path as osp
import sys
import tqdm
import argparse
import logging
import logging.config

import coloredlogs
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import cfg
from model import get_model
from dataset import Dataset


# create logger and direct log to file
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
coloredlogs.install(level=logging.INFO, logger=logger)
fh = logging.FileHandler(osp.join(cfg.log_dir, "log.txt"))
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
fh.setFormatter(fmt)
logger.addHandler(fh)

# dynamically import dataset
for i in range(len(cfg.trainset)):
    exec("from {} import {}".format(cfg.trainset[i], cfg.trainset[i]))
if cfg.testset not in cfg.trainset:
    exec("from {} import {}".format(cfg.testset, cfg.testset))


class Tester:
    def __init__(self, load_tag):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.dataset, self.data_loader, self.n_steps = self.prepare_data()
        self.model = self.create_model(load_tag)

    def create_model(self, load_tag):
        self.logger.info("Creating model...")
        model = get_model(self.dataset.n_joints, self.dataset.n_bins, self.dataset.adj)
        model = nn.DataParallel(model).cuda()

        self.load_model(model, load_tag)

        return model

    def load_model(self, model, load_tag):
        load_path = osp.join(cfg.exp_root, load_tag, "saved_models", "ckpt.pth.tar")
        assert osp.isfile(load_path), "Pre-trained model {} not exist".format(load_path)

        ckpt = torch.load(load_path)
        self.logger.info("Loading model from {} epoch {}".format(load_path, ckpt["epoch"]))

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])

    def prepare_data(self):
        self.logger.info("Creating test dataset...")

        dataset = Dataset(eval(cfg.testset)("test"))
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.num_gpus * cfg.batch_size_per_gpu,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True)

        n_steps = len(data_loader)

        return dataset, data_loader, n_steps

    def test_an_epoch(self):
        self.model.eval()

        preds_pose = []
        preds_bin_idx = []

        tbar = tqdm.tqdm(total=self.n_steps, ncols=100)

        with torch.no_grad():
            for step, batch_data in enumerate(self.data_loader):
                img = batch_data["img"]
                coord_map = batch_data["coord_map"]
                heatmap = batch_data["heatmap"]
                pose = batch_data["pose"]
                bins = batch_data["bins"]
                bin_idx = batch_data["bin_idx"]
                bbox_mask = batch_data["bbox_mask"]
                vis_mask = batch_data["vis_mask"]

                out, _ = self.model(img, coord_map, bbox_mask, vis_mask, epoch=0, target=(heatmap, pose, bins, bin_idx))

                out_pose = out["pose"]  # [B, J*2]
                out_bin_idx = out["bin_idx"]  # [B]

                preds_pose.append(out_pose.cpu().data.numpy())
                preds_bin_idx.append(out_bin_idx.cpu().data.numpy())

                tbar.update(1)

        tbar.close()

        preds_pose = np.concatenate(preds_pose, axis=0)  # [N, J*2]
        preds_bin_idx = np.concatenate(preds_bin_idx, axis=0)  # [N]

        err_pose, err_pose_root, error_x, error_y, error_z, mrpe = self.dataset.evaluate((preds_pose, preds_bin_idx))


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="gpu(s) to use")
    parser.add_argument("--bs", type=int, help="batch size per gpu")
    parser.add_argument("--tag", type=str, required=True, help="experiment to load for evaluation")
    args, _ = parser.parse_known_args()

    return args


def main():
    args = parse_command_line()
    cfg.update_args(vars(args))

    logger.debug("COMMAND LINE: {}".format(str(sys.argv)))
    logger.info("CONFIG:")
    for k in sorted(vars(cfg)):
        logger.info("\t{}: {}".format(k, vars(cfg)[k]))

    cudnn.enabled = True
    cudnn.benchmark = True

    tester = Tester(cfg.tag)

    logger.info("Evaluating exp {} on gpu {}".format(cfg.tag, cfg.gpus))
    tester.test_an_epoch()


if __name__ == "__main__":
    main()
