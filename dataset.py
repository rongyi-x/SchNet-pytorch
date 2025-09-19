#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Schnet
@File ：dataset.py
@Author ：RongYi
@Date ：2025/8/29 15:45
@E-mail ：2071914258@qq.com
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

element_dict = {
    "0": 0,   # padding / unknown
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118
}


class QM9Dataset(Dataset):
    def __init__(self, dataset_path: str, target_idx: str):
        self.df = pd.read_parquet(dataset_path)
        self.target_idx = target_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 获取 z 原子序数的映射
        z = torch.tensor([element_dict[i] for i in row['atomic_symbols']], dtype=torch.long)

        # 获取元素位置
        pos = torch.tensor(np.vstack(row['pos']), dtype=torch.float32)

        # 获取目标属性
        if self.target_idx is None:
            print("""
        # |  字段名   |       含义        |    单位      |
        # | -------- | ---------------   | ----------- |
        # | `mu`     | 偶极矩             | Debye       |
        # | `alpha`  | 各向同性极化率      | Bohr³       |
        # | `homo`   | HOMO 能量          | eV          |
        # | `lumo`   | LUMO 能量          | eV          |
        # | `gap`    | HOMO-LUMO 能隙     | eV          |
        # | `r2`     | 电子空间范围         | Bohr²       |
        # | `zpve`   | 零点振动能          | eV          |
        # | `u0`     | 内能 @ 0 K         | eV          |
        # | `u`      | 内能 @ 298.15 K     | eV          |
        # | `h`      | 焓 @ 298.15 K      | eV          |
        # | `g`      | 自由能 @ 298.15 K   | eV          |
        # | `cv`     | 等容比热容 @ 298.15 K | cal/(mol·K) |
                    """)
            raise ValueError("请输入正确的字段")

        else:
            y = torch.tensor(row[self.target_idx], dtype=torch.float32)

        return z, pos, y

#
# if __name__ == '__main__':
#     dataset = QM9Dataset("./qm9/data/train-00000-of-00001-baa918c342229731.parquet",
#                          target_idx="u0")
#     z, pos, U = dataset.__getitem__(5)
#     print(z, pos, U)

