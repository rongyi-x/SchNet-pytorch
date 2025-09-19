#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
validate.py
Usage:
    python validate.py --data_path ./qm9.parquet --target_idx u0 --device cuda
"""
import os.path
from pathlib import Path

import torch
from torch import nn
from train import get_loader, get_args

from schnet import SchNet


# -------------------- 工具函数 --------------------
@torch.no_grad()
def evaluate(model, loader, loss_fun, device):
    model.eval()
    total_loss, total_mae, total_num = 0.0, 0.0, 0
    for z, pos, y, batch_idx in loader:
        z, pos, y, batch_idx = z.to(device), pos.to(device), y.to(device), batch_idx.to(device)

        # 归一化
        pred = model(z, pos, batch_idx).squeeze(-1)
        print(pred, y)
        loss = loss_fun(pred, y)
        mae = (pred - y).abs().mean()

        total_loss += loss.item() * y.size(0)
        total_mae += mae.item() * y.size(0)
        total_num += y.size(0)

    return total_loss / total_num, total_mae / total_num

# -------------------- 主流程 --------------------
def main():
    args = get_args()
    device = torch.device(args.device)
    # 测试集加载
    _, _, test_loader = get_loader(args)

    # 2. 模型
    model = SchNet().to(device)
    ckpt_path = Path("./ckpts/best.pt")
    assert ckpt_path.exists(), f"找不到模型权重 {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # 3. 归一化参数
    norm_path = Path("./ckpts/norm.pt")
    if norm_path.exists():
        stats = torch.load(norm_path)
        mean, std = stats["mean"], stats["std"]
    else:
        mean = std = None

    # 推理传入参数
    model.std, model.mean = std, mean

    # 4. 计算指标
    loss_fun = nn.L1Loss()
    test_loss, test_mae = evaluate(model, test_loader, loss_fun, device)
    print(f"Test MAE: {test_mae:.4f}  (normalized loss: {test_loss:.4f})")


if __name__ == "__main__":
    main()

