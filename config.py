import sys
import datetime
import os
import os.path as osp


class Config:

    def __init__(self):
        if sys.argv[0].startswith("train"):
            self.exp_tag = datetime.datetime.now().strftime("%m%d_%H%M%S") + "_train"
        elif sys.argv[0].startswith("test"):
            self.exp_tag = datetime.datetime.now().strftime("%m%d_%H%M%S") + "_test"

        # directories
        self.root_dir = osp.dirname(osp.abspath(__file__))
        self.data_dir = osp.join(self.root_dir, "data")
        self.exp_root = osp.join(self.root_dir, "exp")

        self.exp_dir = osp.join(self.exp_root, self.exp_tag)

        self.model_dir = osp.join(self.exp_dir, "saved_models")
        self.log_dir = osp.join(self.exp_dir, "log")

        # datasets
        self.trainset = ["Human36M"]
        self.testset = "Human36M"

        # model
        self.input_size = (256, 256)
        self.output_size = (64, 64)
        self.hm_sigma = 0.75

        # training
        self.lr = 1e-4
        self.lr_decay_factor = 0.8
        self.lr_decay_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.end_epoch = 100
        self.steps_per_epoch = 2000

        # others
        self.batch_size_per_gpu = 16
        self.num_workers = 8

        self.gpus = "0"
        self.num_gpus = 1

    def update_args(self, args):
        assert type(args) == dict, type(args)

        # update gpus
        if "gpu" in args and args["gpu"] is not None:
            for g in args["gpu"].split(","):
                assert g in ["0", "1", "2"], args["gpu"]
            self.gpus = args["gpu"]
            self.num_gpus = len(args["gpu"].split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        print("Set GPU to {}".format(self.gpus))

        if "lr" in args and args["lr"] is not None:
            self.lr = args["lr"]
            print("Set learning rate to {}".format(self.lr))

        if "bs" in args and args["bs"] is not None:
            self.batch_size_per_gpu = args["bs"]
            print("Set batch size to {}".format(self.batch_size_per_gpu))

        if "tag" in args and args["tag"] is not None:
            self.tag = args["tag"]
        else:
            self.tag = None


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, "common"))
from utils.dir_utils import add_pypath, make_folder

add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))

make_folder(cfg.exp_dir)
make_folder(cfg.model_dir)
make_folder(cfg.log_dir)
