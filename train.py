#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
train.py
Usage:
    python train.py --data_path ./qm9.parquet --target_idx u0 \
                    --epochs 300 --batch_size 128 --device cuda
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset

from dataset import QM9Dataset
from schnet import SchNet


def get_args():
    parser = argparse.ArgumentParser(description="SchNet training on QM9")
    parser.add_argument("--data_path", type=str,
                        help="Path to QM9 .parquet file",
                        default="./qm9/data/train-00000-of-00001-baa918c342229731.parquet")
    parser.add_argument("--target_idx", type=str,
                        choices=["mu", "alpha", "homo", "lumo", "gap", "r2",
                                 "zpve", "u0", "u", "h", "g", "cv"],
                        help="0QM9 property to regress", default="u0")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./ckpts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm", type=bool, default=True)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, loss_fun, device, epoch,
                mean=None, std=None):
    # 训练模式
    model.train()
    total_loss, total_num, i = 0.0, 0, 0

    for z, pos, y, batch_idx in train_loader:
        # 模型 输入 都要放一个设备上
        z, pos, y, batch_idx = z.to(device), pos.to(device), y.to(device), batch_idx.to(device)

        # 归一化
        if mean and std:
            y = (y - mean) / std
        else:
            y = y

        # 预测结果 计算 loss
        pred = model(z, pos, batch_idx).squeeze(-1)
        loss = loss_fun(pred, y)

        # 三部曲
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 总损失值
        total_loss += loss.item() * y.size(0)
        total_num += y.size(0)

        if i % 100 == 0:
            print(f"epoch: {epoch} step {i} loss: {total_loss / total_num}")
        i += 1
    return total_loss / total_num


@torch.no_grad()
def validate(model, loader, loss_fun, device, mean, std):
    model.eval()
    total_loss, total_mae, total_num = 0.0, 0.0, 0
    for z, pos, y, batch_idx in loader:
        z, pos, y, batch_idx = z.to(device), pos.to(device), y.to(device), batch_idx.to(device)

        # 传入归一化参数
        if mean and std:
            y = (y - mean) / std
        else:
            y = y

        pred = model(z, pos, batch_idx).squeeze(-1)
        loss = loss_fun(pred, y)
        mae = (pred - y).abs().mean()

        total_loss += loss.item() * y.size(0)
        total_mae += mae
        total_num += y.size(0)
    return total_loss / total_num, total_mae / total_num


def my_collate(batch):
    # 将一个 batch 的原子直接拼接成一个大图
    z_list, pos_list, y_list = zip(*batch)
    z_cat = torch.cat(z_list, dim=0)
    pos_cat = torch.cat(pos_list, dim=0)
    y_stack = torch.stack(y_list, dim=0)

    # 构造 batch_index：0 0 ... 0 | 1 1 ... 1 | ...
    batch_idx = torch.arange(len(batch)).repeat_interleave(
        torch.tensor([len(z) for z in z_list])
    )

    return z_cat, pos_cat, y_stack, batch_idx


def get_loader(args):
    dataset = QM9Dataset(args.data_path, args.target_idx)
    print(f"总数据集大小: {len(dataset)}")

    if os.path.exists("./ckpts/train_indices.pt") and os.path.exists("./ckpts/test_indices.pt") and \
        os.path.exists("./ckpts/val_indices.pt"):
        print(">> 读取已有数据集")
        train_indices = torch.load('./ckpts/train_indices.pt')
        test_indices = torch.load('./ckpts/test_indices.pt')
        val_indices = torch.load('./ckpts/test_indices.pt')

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        val_dataset = Subset(dataset, val_indices)
        print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}, 验证集大小: {len(val_dataset)}")

    else:
        # 读取划分训练集
        print(">> 正在划分数据集")
        train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1])
        print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}, 验证集大小: {len(val_dataset)}")
        # 保存索引
        torch.save(train_dataset.indices, './ckpts/train_indices.pt')
        torch.save(test_dataset.indices, './ckpts/test_indices.pt')
        torch.save(val_dataset.indices, './ckpts/val_indices.pt')

    # 数据加载 pytorch 的 collate_fn 会自动把一个 batch 的数据 stack 起来
    # 但 Z 原子个数是变动的 导致堆叠维度不一致

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, collate_fn=my_collate)

    return train_loader, val_loader, test_loader


def main():
    # 参数读取 设置随机数
    args = get_args()
    set_seed(args.seed)

    # loader
    train_loader, val_loader, _ = get_loader(args=args)

    # 续算逻辑
    ckpt_dir = Path(args.save_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    best_path = ckpt_dir / "best.pt"
    norm_path = ckpt_dir / "norm.pt"

    start_epoch = 1
    if best_path.exists():  # 检查点存在
        print(">> 发现检查点，自动续算...")
        ckpt = torch.load(best_path, map_location=args.device)
        start_epoch = ckpt["epoch"] + 1
        args_saved = ckpt["args"]
        if args.norm != args_saved.norm:
            raise ValueError("归一化配置与检查点不一致，请手动删除 ckpt 或对齐参数")
    else:
        print("× checkpoint 重新开始训练")

    # 归一化
    if args.norm:
        if norm_path.exists() and best_path.exists():
            stats = torch.load(norm_path)
            mean, std = stats["mean"], stats["std"]
        else:
            print(">> 开启归一化训练 >> 计算训练集归一化参数...")
            y_all = torch.cat([y.float() for _, _, y, _ in train_loader])
            mean, std = y_all.mean().item(), y_all.std().item()
            torch.save({"mean": mean, "std": std}, norm_path)
    else:
        mean = std = None

    # 模型初始化 训练不传 mean std 推理传入
    model = SchNet().to(args.device)

    # 如果续算，加载权重
    if best_path.exists():
        model.load_state_dict(ckpt["state_dict"])

    # 损失函数
    loss_fun = nn.L1Loss()  # MAE

    # 优化器 weight_decay L2 正则化
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # 动态学习率 *factor patience epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)

    best_val_mae = ckpt.get("val_mae", float("inf")) if best_path.exists() else float("inf")

    # 模型训练逻辑
    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.time()
        # 传入 模型 训练集 优化器
        train_loss = train_epoch(model, train_loader, optimizer, loss_fun, args.device,
                                 epoch, mean, std)

        # 模型验证
        val_loss, val_mae = validate(model, val_loader, loss_fun, args.device,
                                     mean, std)
        scheduler.step(val_mae)

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            torch.save({"state_dict": model.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "val_mae": val_mae}, best_path)

        print(f"Epoch {epoch:03d} | "
              f"train MAE {train_loss:.4f} | "
              f"val MAE {val_mae:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"time {time.time()-tic:.1f}s")


if __name__ == "__main__":
    main()
