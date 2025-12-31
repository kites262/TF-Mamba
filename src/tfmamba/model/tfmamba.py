import unittest

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BasicMambaBlock(nn.Module):
    def __init__(self, hidden_dim, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(hidden_dim)

        self.mamba = Mamba(
            d_model=hidden_dim,  # 模型维度
            d_state=16,  # SSM 状态维度
            d_conv=4,  # 局部卷积宽度
            expand=2,  # 扩展因子
        )

        if self.bidirectional:
            self.mamba_rev = Mamba(
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, D_Model]
        residual = x
        x = self.norm(x)

        if self.bidirectional:
            # 正向流
            out_fwd = self.mamba(x)
            # 反向流: 翻转序列 -> Mamba -> 翻转回来
            out_rev = self.mamba_rev(x.flip(dims=[1])).flip(dims=[1])
            out = out_fwd + out_rev
        else:
            out = self.mamba(x)

        return residual + out


class TMambaBlock(nn.Module):
    """
    时域 Mamba 模块: 处理原始振动信号
    """

    def __init__(self, hidden_dim, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([BasicMambaBlock(hidden_dim, bidirectional=False) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FMambaBlock(nn.Module):
    """
    频域 Mamba 模块: 处理 FFT 后的频谱
    特点: 使用双向 Mamba (Bi-Mamba)，因为频域特征无因果顺序
    """

    def __init__(self, d_model, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([BasicMambaBlock(d_model, bidirectional=True) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TFMambaModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 1. Embedding Layers (Conv1d + GN + SiLU)
        # 这里的 stride 设置用于下采样，减少序列长度，节省显存
        self.t_embed = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=16, stride=4, padding=6),
            nn.GroupNorm(8, hidden_dim),  # GroupNorm 对 Batch Size 不敏感，适合工业数据
            nn.SiLU(),
        )

        self.f_embed = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=16, stride=4, padding=6), nn.GroupNorm(8, hidden_dim), nn.SiLU()
        )

        # 2. Mamba Backbones
        self.t_mamba = TMambaBlock(hidden_dim)
        self.f_mamba = FMambaBlock(hidden_dim)

        # 3. Classifying Head (Conv1d based)
        # 输入维度是 2 * d_model (因为融合了两路特征)
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # x input: [Batch, Seq_Len, 1] -> [32, 2048, 1]

        # --- A. 数据准备 ---
        # 1. 时域数据: 调整维度为 [B, C, L] 以适配 Conv1d
        x_t = x.permute(0, 2, 1)  # [B, 1, 2048]

        # 2. 频域数据: FFT 变换
        # rfft 输出复数，取模得到幅值谱
        x_f_raw = torch.fft.rfft(x.squeeze(-1), dim=-1).abs()  # [B, 1025]
        x_f = x_f_raw.unsqueeze(1)  # [B, 1, 1025]
        # 简单的截断或填充以匹配 Conv1d 的处理习惯(可选，此处直接输入)

        # --- B. Embedding ---
        x_t = self.t_embed(x_t)  # [B, D, L_t]
        x_f = self.f_embed(x_f)  # [B, D, L_f]

        # --- C. Mamba Layers ---
        # Mamba 需要 [B, L, D] 格式，所以需要 permute
        x_t = x_t.permute(0, 2, 1)
        x_t = self.t_mamba(x_t)  # [B, L_t, D]

        x_f = x_f.permute(0, 2, 1)
        x_f = self.f_mamba(x_f)  # [B, L_f, D]

        # --- D. 融合与池化 ---
        # Global Average Pool: 将变长序列压缩为特征向量
        # mean(dim=1) 消除序列长度维度
        feat_t = x_t.mean(dim=1)  # [B, D]
        feat_f = x_f.mean(dim=1)  # [B, D]

        # 融合: 拼接
        feat_fusion = torch.cat([feat_t, feat_f], dim=1)  # [B, 2*D]

        # --- E. 分类头 ---
        # 为了满足使用 Conv1d 做分类头的要求，我们将向量 reshape 为 [B, 2*D, 1]
        feat_fusion = feat_fusion.unsqueeze(-1)  # [B, 2*D, 1]

        logits = self.head(feat_fusion)  # [B, Num_Classes, 1]

        return logits.squeeze(-1)  # [B, Num_Classes]


def show_model_params():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TFMambaModel(
        num_classes=10,
        hidden_dim=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")


class TestTFMambaModel(unittest.TestCase):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def setUp(self):
        self.model = TFMambaModel(num_classes=10, hidden_dim=64).to(self.device)

    def test_forward_shape(self):
        input_tensor = torch.randn(32, 2048, 1).to(self.device)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (32, 10))


if __name__ == "__main__":
    show_model_params()
    unittest.main()
