# TF-Mamba

## 任务介绍

> 以轴承振动信号为基础, 研究基于机器学习的轴承故障检测与诊断方法, 通过对信号进行预处理和特征提取, 结合多特征融合策略, 设计分类器对故障类型进行判别, 最终完成对故障的诊断

在本任务中, 我们需要针对序列数据, 设计一个分类模型, 从而实现对轴承是否故障以及故障类型的判别

针对序列问题, 我们设计了 TF-Mamba 模型, 该模型基于 Mamba 状态空间模型, 能够在线性复杂度下捕捉类似 Transformer 的长距离依赖关系, 同时具备 RNN 的高效序列建模能力

为了提高模型的识别能力, 我们引入多特征融合策略, 通过对原始信号进行多种特征提取, 并将其融合为多通道输入, 提供更丰富的信息供模型学习

在实验中, 我们将 TF-Mamba 模型与非常简单的 1d-CNN 模型进行了对比, 从而验证了 TF-Mamba 在序列任务中的**高效性**, 表明了**状态空间模型**在处理序列数据时的**契合度**以及**融合多特征策略的有效性**

## 代码结构

```plaintext
.
├── conf                        # 训练配置
│   ├── config                  # 通用配置
│   │   └── base.yaml
│   ├── config.yaml             # 主配置文件
│   ├── console                 # 控制台相关配置
│   │   └── swanlab.yaml
│   └── model                   # 不同模型配置
│       ├── cnn.yaml
│       └── tfmamba.yaml
├── data/                       # 数据集存放路径
├── outputs/                    # 训练输出路径
├── pyproject.toml              # 项目依赖
├── README.MD                   # 项目说明文件
├── scripts                     # 数据下载和训练脚本
│   ├── data.sh
│   ├── env.sh
│   └── train
│       ├── cnn_by_file.sh
│       ├── cnn_by_window.sh
│       ├── tfmamba_by_file.sh
│       └── tfmamba_by_window.sh
├── src                         # 源代码
│   └── tfmamba
│       ├── helper              # 辅助模块
│       │   ├── dataset.py
│       │   └── pipeline.py
│       ├── model               # 模型定义
│       │   ├── cnn.py
│       │   └── tfmamba.py
│       ├── protocol            # 协议层
│       │   ├── dataset.py
│       │   ├── experiment.py
│       │   ├── logger.py
│       │   └── metrics.py
│       └── utils               # 工具模块
│           ├── dataloader.py
│           ├── logger.py
│           └── metrics.py
└── train.py
```

## 快速开始

在项目文件夹, 通过以下命令创建并激活虚拟环境:

```bash
conda create -n pro@TF-Mamba python=3.11 -y
conda activate pro@TF-Mamba

bash scripts/env.sh
```

下载数据集:

```bash
bash scripts/data.sh
```

开始训练:

```bash
bash scripts/train/tfmamba_by_window.sh
bash scripts/train/tfmamba_by_file.sh
bash scripts/train/cnn_by_window.sh
bash scripts/train/cnn_by_file.sh
```

## 结果

File-Sized Split Training Results:

| Model    | Overall Accuracy | Recall  | Precision | F1-Score |
| -------- | ---------------- | ------- | --------- | -------- |
| TF-Mamba | 10.00%           | 100.00% | 100.00%   | 100.00%  |
| CNN      | 72.63%           | 64.39%  | 64.22%    | 56.79%   |

在线查看训练结果: [SwanLab TF-Mamba Results](https://swanlab.cn/@kites/TF-Mamba/runs)
