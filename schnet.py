import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Dict, Optional, Tuple, Any, Set, List


class SchNet(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        readout: str = 'add',
        dipole: bool = False,
        atomref: Optional[Tensor] = None,

    ):
        super().__init__()
        self.max_num_neighbors = max_num_neighbors
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.scale = None
        self.atomref = atomref
        self.mean = mean
        self.std = std

        # 偶极矩需要计算质心
        if self.dipole:
            import ase
            atomic_mass = torch.from_numpy(ase.atoms.atomic_masses)
            self.register_buffer('atomic_mass', atomic_mass)

        # embedding 层 (100, 128)
        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0)

        # 构造原子无向图
        self.interaction_graph = RadiusInteractionGraph(self.cutoff, self.max_num_neighbors)
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # interaction 层
        self.interactions = nn.ModuleList()
        for _ in range(self.num_interactions):
            self.interactions.append(
                InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            )

        # atom_wise 层
        self.atom_wise1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.atom_wise2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, z: Tensor, pos: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        # 并行计算 会把 N 个分子拼接成一个 batch
        batch = torch.zeros_like(z) if batch is None else batch

        # embedding 层
        h = self.embedding(z)
        # 构造图 获得(2, E), (E)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        # (E) -标量维度扩展> (E, num_gaussians) 边特征
        edge_attr = self.distance_expansion(edge_weight)

        # interaction 层
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_weight, edge_attr)

        # atom-wise 线性层
        h = self.atom_wise1(h)
        h = self.act(h)
        h = self.atom_wise2(h)

        # out = self.readout(h, batch, dim=0)
        # 假设 batch 从 0 开始连续编号，batch_size = batch.max() + 1
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, *h.shape[1:], device=h.device, dtype=h.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)

        # 将原子特征输出为分子特征: sum pooling 层
        # 偶极矩任务
        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            out = out * (pos - c.index_select(0, batch))

        # 训练时做了标准化 推理要反标准化
        if not self.dipole and self.mean is not None and self.std is not None:
            out = out * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            out = out + self.atomref(z)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        nn.init.xavier_uniform_(self.atom_wise1.weight)
        self.atom_wise1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_wise2.weight)
        self.atom_wise2.bias.data.fill_(0)


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians: int, num_filters: int, cutoff: float):
        super().__init__()
        self.atom_wise1 = nn.Linear(hidden_channels, hidden_channels)
        self.conv = CFConv(hidden_channels, num_gaussians, num_filters, cutoff)
        self.act = ShiftedSoftplus()
        self.atom_wise2 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor):
        x1 = self.atom_wise1(x)
        x1 = self.conv(x1, edge_index, edge_weight, edge_attr)
        x1 = self.act(x1)
        x1 = self.atom_wise2(x1)
        return x + x1

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.atom_wise1.weight)
        self.atom_wise1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_wise2.weight)
        self.atom_wise2.bias.data.fill_(0)
        self.conv.reset_parameters()


class RadiusInteractionGraph(nn.Module):
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    @staticmethod
    def _split_by_batch(pos: Tensor, batch: Tensor) -> Tuple[list[Tensor], list[Tensor]]:
        # 拿到分子id
        molecules = torch.unique(batch, sorted=True)

        pos_list, idx_list = [], []
        for mol in molecules:
            mask = batch == mol
            pos_list.append(pos[mask])
            # 获得该分子中的原子在 batch 中的索引
            idx_list.append(torch.nonzero(mask).squeeze(-1))
        return pos_list, idx_list

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        device = pos.device
        pos_list, idx_list = self._split_by_batch(pos, batch)

        src_all, dst_all, dist_all = [], [], []

        for pos_mol, idx_mol in zip(pos_list, idx_list):
            n = pos_mol.size(0)
            if n == 1:  # 单原子分子没有邻居
                continue

            # 计算欧式距离矩阵 pos: (N, 3) -> dist: (N, N)
            dist_mat = torch.cdist(pos_mol, pos_mol)

            # 条件掩码: 距离
            mask = (dist_mat > 0) & (dist_mat < self.cutoff)

            # 将 mask 中 False 的 dist 值置为 inf 无穷大
            inf = torch.tensor(float('inf'), device=device)
            masked_dist = dist_mat.masked_fill(~mask, inf)

            # max_num_neighbors 每个分子内部决定 k
            k = min(self.max_num_neighbors, n - 1)
            # largest=False： 沿着 dim=1 找 k 个最小值 neighbor_idx: (N, k)
            _, neighbor_idx = torch.topk(masked_dist, k=k, dim=1, largest=False)

            # 构建 COO 边列表 (N) -> (N, 1) -> (N, k) -> (N * k)
            # expand: 第一维不变 把长度为 1 的第二维 复制 k 次
            src_local = neighbor_idx.reshape(-1)
            dst_local = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).reshape(-1)

            # 单分子原子索引映射到全局 idx_mol
            src_all.append(idx_mol[src_local])
            dst_all.append(idx_mol[dst_local])

            # dist_mat 就是当前分子的距离 所以无需映射
            dist_all.append(dist_mat[src_local, dst_local])

        # 拼接成整张图
        if not src_all:  # 极端情况：没有任何边
            empty = torch.empty((2, 0), dtype=torch.long, device=device)
            return empty, torch.empty(0, device=device)

        # 源节点是消息的发送方 目标节点是消息的接受方
        edge_index = torch.stack([torch.cat(src_all), torch.cat(dst_all)], dim=0)
        edge_weight = torch.cat(dist_all)

        return edge_index, edge_weight


class GaussianSmearing(nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 10.0,
        num_gaussians: int = 50,
    ):  
        super().__init__()

        # uk
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        # 广播机制  -> (N, 50)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class MessagePassing(nn.Module):
    # 内部参数
    special_args = {
        'edge_index', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'index', 'dim_size'
    }

    def __init__(self, aggr: str = "add", size: List = None) -> None:
        # 使用 sum 聚合；flow 默认为 source_to_target
        super().__init__()
        self.aggr = aggr  # 支持 add, mean, max …
        self.size = size

    # 参数扫描
    def _scan_args_(self, func) -> List:
        # 取出参数 去掉self
        names = list(func.__code__.co_varnames)[1:]
        return [n for n in names if n not in self.special_args]

    # 参数挑选
    def _collect_(self, user_args, edge_index, kwargs):
        i, j = 1, 0  # source to target
        out = {}
        # 用户参数处理逻辑
        for arg in user_args:
            if arg.endswith('_j'):
                key = arg[:-2]
                out[arg] = kwargs[key].index_select(0, edge_index[j])
            elif arg.endswith('_i'):
                key = arg[:-2]
                out[arg] = kwargs[key].index_select(0, edge_index[i])
            else:
                # 取出/占位 防止报错 inputs
                out[arg] = kwargs.get(arg, None)

        # 添加常用参数
        out['index'] = edge_index[i]  # 聚合维度
        out['dim_size'] = self.size[0] if self.size is not None else None
        return out

    # 入口函数
    def propagate(self, edge_index: Tensor, **kwargs) -> Tensor:
        # 获取 size
        size = int(edge_index.max()) + 1

        # 参数扫描
        msg_args = self._scan_args_(self.message)
        aggr_args = self._scan_args_(self.aggregate)
        upd_args = self._scan_args_(self.update)

        # 1. 处理传入参数
        coll_dict = self._collect_(msg_args + aggr_args + upd_args, edge_index, kwargs)

        # 2. 消息
        msg = self.message(**{k: coll_dict[k] for k in msg_args})

        # 3. 聚合
        aggr = self.aggregate(msg, coll_dict['index'], dim_size=size)

        # 4. 更新
        return self.update(aggr)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor, dim_size: Optional[int]) -> Tensor:
        if self.aggr == "add":
            # 创建一个 维度 (node_dim, inputs.dim[1:]) 除了节点维度 inputs 维度不变 适应输出
            # index_add 按索引累加 在第 0 维节点维度累加 index 聚合维度 i 也就是 target
            # inputs 是产生的消息 (N, F)
            return torch.zeros(dim_size, * inputs.shape[1:],
                               device=inputs.device).index_add(0, index, inputs)
        if self.aggr == 'mean':
            # torch.bincount 统计 target 节点周围邻居的个数 minlength 设置输出最小为 dim_size
            # clamp 避免除 0  view.(-1, 1) 广播机制 (N) -> (N, 1)
            return torch.zeros(dim_size, *inputs.shape[1:],
                               device=inputs.device).index_add(0, index, inputs)\
        / torch.bincount(index, minlength=dim_size).clamp(min=1).view(-1, 1)

        raise ValueError(f"Unsupported aggregation {self.aggr}")

    def update(self, inputs: Tensor) -> Tensor:
        return inputs


class CFConv(MessagePassing):
    def __init__(self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float):
        super().__init__()
        # rbf -> num_gaussians 维度映射
        self.dense1 = nn.Linear(num_gaussians, num_filters)
        self.act = ShiftedSoftplus()
        self.dense2 = nn.Linear(num_filters, num_filters)

        self.linear1 = nn.Linear(hidden_channels, num_filters, bias=False)
        self.linear2 = nn.Linear(num_filters, hidden_channels)
        self.cutoff = cutoff
        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        # 连续卷积核生成函数
        C = 0.5 * (torch.cos(edge_weight * torch.pi / self.cutoff) + 1.0)

        # 权重矩阵
        W = self.dense1(edge_attr)
        W = self.act(W)
        W = self.dense2(W)
        W *= C.view(-1, 1)

        # 先映射到 num_filters 维度
        # 进行消息传递
        # 然后回到 hidden_channels
        x = self.linear1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.linear2(x)

        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return W * x_j

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.dense1.weight)
        self.dense1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dense2.weight)
        self.dense2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.linear1.weight)  # 无偏置
        nn.init.xavier_uniform_(self.linear2.weight)
        self.dense2.bias.data.fill_(0)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift.to(x.device)


if __name__ == '__main__':
    schnet = SchNet()




