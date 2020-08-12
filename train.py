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


class Trainer:
    def __init__(self, load_tag):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.datasets, self.data_loaders, self.iterators, self.n_steps = self.prepare_data()
        self.model, self.start_epoch, self.end_epoch, self.tot_step = self.create_model(load_tag)
        self.optimizer = self.create_optimizer(self.model)

    def save_model(self, epoch, best_model):
        save_path = osp.join(cfg.model_dir, "ckpt.pth.tar")
        torch.save({
            "model": self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            "epoch": epoch,
        }, save_path)

        self.logger.info("Save model to {}".format(save_path))

        if best_model:
            save_path = osp.join(cfg.model_dir, "ckpt_best.pth.tar")
            torch.save({
                "model": self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                "epoch": epoch,
            }, save_path)

            self.logger.info("Save model to {}".format(save_path))

    def set_lr(self, epoch):
        cur_lr = cfg.lr * cfg.lr_decay_factor ** np.sum(np.array(cfg.lr_decay_epochs) <= epoch)
        self.optimizer.param_groups[0]["lr"] = cur_lr

        return cur_lr

    def prepare_data(self):
        self.logger.info("Creating training datasets...")

        datasets = []
        data_loaders = []
        iterators = []

        for i in range(len(cfg.trainset)):
            ds = Dataset(eval(cfg.trainset[i])("train"))
            dl = torch.utils.data.DataLoader(
                dataset=ds,
                batch_size=cfg.num_gpus * cfg.batch_size_per_gpu // len(cfg.trainset),
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            datasets.append(ds)
            data_loaders.append(dl)
            iterators.append(iter(dl))

        n_steps = len(data_loaders[0])

        return datasets, data_loaders, iterators, n_steps

    def load_model(self, model, load_tag):
        load_path = osp.join(cfg.exp_root, load_tag, "saved_models", "ckpt.pth.tar")

        ckpt = torch.load(load_path)
        self.logger.info("Loading model from {} epoch {}".format(load_path, ckpt["epoch"]))

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])

        return ckpt["epoch"]

    def create_model(self, load_tag):
        self.logger.info("Creating model...")
        model = get_model(self.datasets[0].n_joints, self.datasets[0].n_bins, self.datasets[0].adj)
        model = nn.DataParallel(model).cuda()

        # load model to continue training
        if load_tag is not None:
            start_epoch = self.load_model(model, load_tag) + 1
        else:
            start_epoch = 0
        end_epoch = cfg.end_epoch
        tot_step = 0

        return model, start_epoch, end_epoch, tot_step

    def create_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        return optimizer

    def train_an_epoch(self, epoch):
        self.model.train()

        self.set_lr(epoch)

        for i in range(len(cfg.trainset)):
            self.iterators[i] = iter(self.data_loaders[i])

        n_steps_per_epoch = min(self.n_steps, cfg.steps_per_epoch)
        tbar = tqdm.tqdm(total=n_steps_per_epoch, ncols=100)

        for step in range(n_steps_per_epoch):

            data = {
                "img": [],
                "coord_map": [],
                "heatmap": [],
                "pose": [],
                "bins": [],
                "bin_idx": [],
                "bbox_mask": [],
                "vis_mask": [],
            }

            for i in range(len(cfg.trainset)):
                try:
                    batch_data = next(self.iterators[i])
                except StopIteration:
                    self.iterators[i] = iter(self.data_loaders[i])
                    batch_data = next(self.iterators[i])

                for k in data:
                    data[k].append(batch_data[k])

            for k in data:
                data[k] = torch.cat(data[k], dim=0)

            out, loss = self.model(data["img"], data["coord_map"], data["bbox_mask"], data["vis_mask"], epoch=epoch, target=(data["heatmap"], data["pose"], data["bins"], data["bin_idx"]))
            tot_loss = loss["tot_loss"].mean()

            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()

            self.tot_step += 1

            tbar.set_description("Loss = {:.4f}".format(tot_loss.item()))
            tbar.update(1)

        tbar.close()

        return self.tot_step


class Tester:
    def __init__(self, model):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.dataset, self.data_loader, self.n_steps = self.prepare_data()

        self.best_result = None

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

    def test_an_epoch(self, epoch, global_step):
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

                out, loss = self.model(img, coord_map, bbox_mask, vis_mask, epoch=epoch, target=(heatmap, pose, bins, bin_idx))

                out_pose = out["pose"]  # [B, J*2]
                out_bin_idx = out["bin_idx"]  # [B]

                preds_pose.append(out_pose.cpu().data.numpy())
                preds_bin_idx.append(out_bin_idx.cpu().data.numpy())

                tbar.update(1)

        tbar.close()

        preds_pose = np.concatenate(preds_pose, axis=0)  # [N, J*2]
        preds_bin_idx = np.concatenate(preds_bin_idx, axis=0)  # [N]

        best_model = False
        err_pose, err_pose_root, error_x, error_y, error_z, mrpe = self.dataset.evaluate((preds_pose, preds_bin_idx))

        if self.best_result is None:
            self.best_result = mrpe
        else:
            best_mrpe = self.best_result

            if mrpe < best_mrpe:
                best_model = True
                best_mrpe = mrpe

            self.best_result = best_mrpe

        return best_model


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="gpu(s) to use")
    parser.add_argument("--bs", type=int, help="batch size per gpu")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--tag", type=str, help="experiment to load to continue training")
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

    trainer = Trainer(cfg.tag)
    tester = Tester(trainer.model)

    # train
    for epoch in range(trainer.start_epoch, trainer.end_epoch):
        logger.info("Epoch {} of exp {} on gpu {}".format(epoch, cfg.exp_tag, cfg.gpus))

        step = trainer.train_an_epoch(epoch)
        best_model = tester.test_an_epoch(epoch, step)

        trainer.save_model(epoch, best_model)


if __name__ == "__main__":
    main()
