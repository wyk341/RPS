import os
import sys

from tools.all_tools import attach_meta, parse_args
from tools.train_semi import train_start_semi
from tools.train_semi1 import train_start_semi1
from tools.train_semi2 import train_start_semi2
from tools.train_semi3 import train_start_semi3
from tools.train_semi4 import train_start_semi4
from tools.train_sup import train_start_supervised
from mmocr.core.optimizer.builder import build_optimizer
# Ensure project root is on sys.path when running this file directly (python ./tools/train.py)
# So that imports like `from tools.simple_train import ...` work.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#!/usr/bin/env python
import json
import torch
import numpy as np
import os, os.path as osp, pickle, time
from torch.utils.data import Dataset
import numpy as np
from mmcv import Config
from mmcv.runner import  set_random_seed
from mmdet.datasets import (build_dataloader)
from mmocr.models import build_detector
from torch.utils.data import Subset

torch.multiprocessing.set_start_method('fork')

# ================== 默认训练配置 ==================
DEFAULT_CONFIG_PATH = '/workspace/MASA/config/config_fintabnet.py'
DEFAULT_WORK_DIR = './output_fintabnet/' #结果输出路径
# DEFAULT_SEED = 123              # 或者 None
# =================================================

_BASE_DS_CACHE = None  # 进程内复用

# ---- 把 flag / CLASSES / PALETTE 等元信息转接到 Subset 上 ----

class RepeatDataset(Dataset):
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = int(times)
        self._len = len(dataset) * self.times

        # 处理 flag（给 GroupSampler 用）
        if hasattr(dataset, 'flag'):
            f = np.asarray(dataset.flag)
            self.flag = np.tile(f, self.times)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

def run_training_by_cfg(model, data_loaders_labeled, data_loaders_unlabeled, data_loaders_val, cfg):
    # 每次训练都重新 build optimizer（强烈建议）
    optimizer = build_optimizer(model, cfg.optimizer)

    mode = str(getattr(cfg, "train_mode", "supervised")).lower()
    print(f"[RUN] train_mode = {mode}")

    if mode in ("supervised", "sup", "s"):
        train_start_supervised(
            model,
            data_loaders_labeled,
            data_loaders_val,
            optimizer,
            cfg
        )

    elif mode in ("semi", "semisupervised", "semi_supervised", "ssl"):
        train_start_semi4(
            model,
            data_loaders_labeled,
            data_loaders_unlabeled,
            data_loaders_val,
            optimizer,
            cfg
        )

    else:
        raise ValueError(f"Unknown cfg.train_mode={mode}. Use 'supervised' or 'semi'.")

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.mc_config:
        mc = Config.fromfile(args.mc_config)
        if isinstance(cfg.data.train, list):
            for i in range(len(cfg.data.train)):
                cfg.data.train[i].pipeline[0].update(
                    file_client_args=mc['mc_file_client_args'])
        else:
            cfg.data.train.pipeline[0].update(
                file_client_args=mc['mc_file_client_args'])

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    #print("import modules from string list.")

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    distributed = False
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    # ----------------------加载模型 -------------------------
    model = build_detector(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total: {total_params / 1e6:.2f} M')

    #---------------------- 加载预训练模型参数 ------------------
    pretrained = getattr(cfg, "pretrained", None)
    resume_from = getattr(cfg, "resume_from", None)
    load_from = getattr(cfg, "load_from", None)

    if resume_from:
        print(f"[CFG] resume_from = {resume_from}")
    elif load_from:
        print(f"[CFG] load_from = {load_from}")
    elif pretrained:
        print(f"[CFG] pretrained = {pretrained}")
    else:
        print('ℹ No pretrained specified or file not found, training from scratch.')
    # ------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total: {total_params / 1e6:.2f} M')

    # ======================================================================
    # =========================== 数据集 & 划分加载 ===========================
    # ======================================================================

    print("=== Start building datasets with base_dataset.pkl + stats.json ===")
    global_start = time.time()

    # ---------------- 1) 准备 base_dataset（只构造 / 反序列化一次） ----------------
    # 1) 训练集路径以及划分准备
    base_pkl_path_train = getattr(cfg, 'base_pkl_path_train')  # 你之前保存的路径
    split_dir_train = getattr(cfg, 'split_dir_train')  # 注意和你磁盘上的目录一致

    if osp.exists(base_pkl_path_train):
        print(f"load base_dataset from {base_pkl_path_train}")
        with open(base_pkl_path_train, 'rb') as f:
            base_dataset_train = pickle.load(f)
    else:
        msg = f"[ERROR] base_dataset pkl not exist：{base_pkl_path_train}\n"
        print(msg)
        raise FileNotFoundError(msg)

    # 2) 验证集路径以及划分准备
    base_pkl_path_val = getattr(cfg, 'base_pkl_path_val')  # 你之前保存的路径
    split_dir_val = getattr(cfg, 'split_dir_val')  # 注意和你磁盘上的目录一致

    if osp.exists(base_pkl_path_val):
        print(f"load base_dataset from {base_pkl_path_val}")
        with open(base_pkl_path_val, 'rb') as f:
            base_dataset_val = pickle.load(f)
    else:
        msg = f"[ERROR] base_dataset pkl not exist：{base_pkl_path_val}\n"
        print(msg)
        raise FileNotFoundError(msg)


    print("start base dataset")
    start_time = time.time()

    base_len_train = len(base_dataset_train)
    base_len_val = len(base_dataset_val)
    end_time = time.time()
    print(f"base_dataset_train length: {base_len_train}")
    print(f"base_dataset_val length: {base_len_val}")
    print(f"base dataset 准备耗时: {end_time - start_time:.2f} 秒")

    # ---------------- 2) 读取划分信息 stats.json ----------------
    # 从 cfg 中读取 split_dir（stats.json 所在目录），否则使用原默认
    os.makedirs(split_dir_train, exist_ok=True)
    stats_path_train = osp.join(split_dir_train, 'stats.json')

    os.makedirs(split_dir_val, exist_ok=True)
    stats_path_val = osp.join(split_dir_val, 'val_fast.json')

    with open(stats_path_train, 'r') as f:
        stats_train = json.load(f)
    with open(stats_path_val, 'r') as f:
        stats_val = json.load(f)

    labeled_indices = stats_train['labeled_indices']
    unlabeled_indices = stats_train['unlabeled_indices']
    val_indices = stats_val['val_fast_indices']

    print(f"labeled_indices   数量: {len(labeled_indices)}")
    print(f"unlabeled_indices 数量: {len(unlabeled_indices)}")
    print(f"val_indices      数量: {len(val_indices)}")

    # ---------------- 3) 用 Subset 构造三个子集（极轻量） ----------------
    print("build Subset datasets ...")
    start_time = time.time()

    dataset_labeled = Subset(base_dataset_train, labeled_indices)
    dataset_unlabeled = Subset(base_dataset_train, unlabeled_indices)
    dataset_val = Subset(base_dataset_val, val_indices)

    attach_meta(dataset_labeled, labeled_indices, base_dataset_train)
    attach_meta(dataset_unlabeled, unlabeled_indices, base_dataset_train)
    attach_meta(dataset_val, val_indices, base_dataset_val)

    end_time = time.time()
    print(f"build Subset 耗时: {end_time - start_time:.2f} 秒")

    # ======================================================================
    # =========================== Dataloader 构造 ===========================
    # ======================================================================
    # 1) 有标签数据
    print("Preparing labeled dataloader......")
    start_time = time.time()
    data_loader_labeled = build_dataloader(
        dataset_labeled,
        cfg.sup_bs,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed
    )
    data_loaders_labeled = [data_loader_labeled]
    end_time = time.time()
    print(f"labeled dataloader 耗时: {end_time - start_time:.2f} 秒")

    # 2) 无标签数据
    print("Preparing unlabeled dataloader......")
    start_time = time.time()
    data_loader_unlabeled = build_dataloader(
        dataset_unlabeled,
        cfg.unsup_bs,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed
    )
    data_loaders_unlabeled = [data_loader_unlabeled]
    end_time = time.time()
    print(f"unlabeled dataloader 耗时: {end_time - start_time:.2f} 秒")

    # 3) 验证集数据
    print("Preparing val dataloader......")
    start_time = time.time()
    data_loader_val = build_dataloader(
        dataset_val,
        cfg.val_bs,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed
    )

    data_loaders_val = [data_loader_val]
    end_time = time.time()
    print(f"val dataloader 耗时: {end_time - start_time:.2f} 秒")

    global_end = time.time()
    print(f"=== 数据集+loader 总耗时: {global_end - global_start:.2f} 秒 ===")

    # ======================================================================
    # ============================== 调用训练 ===============================
    # ======================================================================

    del base_dataset_train
    del base_dataset_val
    del stats_train, stats_val
    del labeled_indices, unlabeled_indices, val_indices

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    run_training_by_cfg(
        model,
        data_loaders_labeled,
        data_loaders_unlabeled,
        data_loaders_val,
        cfg
    )










    # # ======================================================================
    # # =========================== 有标签数据准备 ============================
    # # ======================================================================
    # print("strat labeled dataset")
    # start_time = time.time()
    # os.makedirs('cache', exist_ok=True)
    # if osp.exists('cache/spilt_test/ratio_10_90/labeled_with_flag.pkl'):
    #     with open('cache/spilt_test/ratio_10_90/labeled_with_flag.pkl', 'rb') as f:
    #         dataset = pickle.load(f)
    # else:
    #     dataset = build_dataset(cfg.data.train)
    #     with open('cache/spilt_test/ratio_10_90/labeled_with_flag.pkl', 'wb') as f:
    #         pickle.dump(dataset, f)
    # datasets_labeled = [dataset]
    # print("Preparing labeled dataloader......")
    # data_loaders_labeled = [
    #     build_dataloader(
    #         ds,
    #         cfg.data.samples_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         len(cfg.gpu_ids),
    #         dist=distributed,
    #         seed=cfg.seed) for ds in datasets_labeled
    # ]
    # end_time = time.time()
    # print(f"labeled dataset 耗时: {end_time - start_time:.2f} 秒")
    # #
    #
    # # ======================================================================
    # # =========================== 无标签数据准备 ============================
    # # ======================================================================
    # print("strat unlabeled dataset")
    # start_time = time.time()
    # os.makedirs('cache', exist_ok=True)
    # if osp.exists('cache/spilt1000/ratio_10_80_10/unlabeled_with_flag.pkl'):
    #     with open('cache/spilt1000/ratio_10_80_10/unlabeled_with_flag.pkl', 'rb') as f:
    #         dataset = pickle.load(f)
    # else:
    #     dataset = build_dataset(cfg.data.train)
    #     with open('cache/spilt1000/ratio_10_80_10/unlabeled_with_flag.pkl', 'wb') as f:
    #         pickle.dump(dataset, f)
    # datasets_unlabeled = [dataset]
    # print("Preparing unlabeled dataloader......")
    # data_loaders_unlabeled = [
    #     build_dataloader(
    #         ds,
    #         cfg.data.samples_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         len(cfg.gpu_ids),
    #         dist=distributed,
    #         seed=cfg.seed) for ds in datasets_unlabeled
    # ]
    # end_time = time.time()
    # print(f"unlabeled dataset 耗时: {end_time - start_time:.2f} 秒")
    #
    #
    # # ======================================================================
    # # =========================== 预测集数据准备 ============================
    # # ======================================================================
    # print("strat val dataset")
    # start_time = time.time()
    # os.makedirs('cache', exist_ok=True)
    # if osp.exists('cache/spilt_test/ratio_10_90/labeled_with_flag.pkl'):
    #     with open('cache/spilt_test/ratio_10_90/labeled_with_flag.pkl', 'rb') as f:
    #         dataset = pickle.load(f)
    # else:
    #     dataset = build_dataset(cfg.data.train)
    #     with open('cache/spilt_test/ratio_10_90/labeled_with_flag.pkl', 'wb') as f:
    #         pickle.dump(dataset, f)
    # datasets_val = [dataset]
    # print("Preparing val dataloader......")
    # data_loaders_val= [
    #     build_dataloader(
    #         ds,
    #         cfg.data.samples_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         len(cfg.gpu_ids),
    #         dist=distributed,
    #         seed=cfg.seed) for ds in datasets_val
    # ]
    # end_time = time.time()
    # print(f"val dataset 耗时: {end_time - start_time:.2f} 秒")
    #
    #
    # # ======================================================================
    # # ============================== 调用训练 ===============================
    # # ======================================================================
    # optimizer = build_optimizer(model, cfg.optimizer)
    # #
    # del datasets_unlabeled,datasets_labeled,datasets_val
    # train_start(model,data_loaders_labeled,data_loaders_unlabeled,data_loaders_val,optimizer,cfg)
    #
    # del datasets_labeled, datasets_val
    # train_overfit_one_image(model,data_loaders_val,data_loaders_val,optimizer,cfg)

if __name__ == '__main__':
    main()