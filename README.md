<<<<<<< HEAD
# Domain-invariant Clinical Representation Learning for Emerging Disease Prediction

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-orange.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code for the paper: **"Domain-invariant Clinical Representation Learning by Bridging Data Distribution Shift across EMR Datasets"**.

## 摘要 (Abstract)

新兴疾病由于信息有限，在症状识别和及时临床干预方面带来了巨大挑战。本文提出了一种领域不变的表示学习方法，旨在解决新兴疾病场景下临床数据稀缺和跨数据集特征不一致的问题。我们通过构建一个由教师模型指导的过渡模型，并结合领域对抗训练，来学习一种领域不变的特征表示。该框架能够有效捕捉不同医疗领域间的共性知识，并通过一个基于动态时间规整（DTW）的迁移机制来处理私有特征，最终在多个新兴疾病预测任务上取得了优越的性能。

## 框架概述 (Framework Overview)

我们的模型采用了一个三阶段的迁移学习框架，以实现从数据丰富的源域到数据稀缺的目标域的知识迁移。

![模型整体框架图](https://i.imgur.com/K1LdC7I.png)
*图：新兴疾病预测模型的整体框架。左: 在源数据集上训练教师模型; 中: 训练领域不变的过渡模型; 右: 将知识迁移到目标域模型并进行微调。*

1.  **阶段一：预训练教师模型 (Pretrain Teacher Model)**：在大规模的源域数据集（如PhysioNet）上训练一个强大的多通道GRU模型，作为知识的来源。
2.  **阶段二：训练过渡模型 (Train Transition Model)**：该模型是框架的核心。它同时接收源域和目标域的数据，通过**知识蒸馏**模仿教师模型的表示，并通过**领域对抗训练**学习领域不变的特征。
3.  **阶段三：迁移与微调 (Transfer and Fine-tune)**：将过渡模型中训练好的GRU参数迁移到最终的目标模型。其中，共享特征直接迁移；私有特征通过DTW距离匹配最相似的源域特征进行迁移。最后在目标域数据上进行微调。

## 环境要求 (Requirements)

建议使用conda创建独立的虚拟环境。
```bash
# 创建conda环境
conda create -n your_env_name python=3.7
conda activate your_env_name

# 安装核心依赖
pip install torch==1.12.1 torchvision==0.13.1
pip install numpy pandas scikit-learn matplotlib

# 安装DTW库
pip install dtw-python
```

## 数据集准备 (Dataset Preparation)

本研究使用了以下数据集：
* **源域数据集**: [PhysioNet Challenge 2019 Sepsis Dataset](https://physionet.org/content/challenge-2019/1.0.0/)
* **目标域数据集**:
    * TJH COVID-19 Dataset
    * HMH COVID-19 Dataset
    * PUTH ESRD Dataset

您需要下载这些数据集，并进行详细的预处理，包括数据清洗、缺失值填充、标准化等。最终，请将处理好的数据保存为代码预期的格式（通常是`.pkl`或`.dat`文件）。**请参考每个Notebook中数据加载部分的代码，以了解确切的文件名和路径结构。**

## 如何运行 (Usage)

本项目的实验是通过一系列独立的Jupyter Notebook (`.ipynb`) 文件来运行的。每个文件对应一项特定的实验（如一个特定的基线模型或消融实验）。

### 1. 运行核心模型 (Ours / `distcare_adversal`)

要复现本文提出的核心模型的结果，您需要按顺序完成三个阶段的训练（尽管它们可能被整合在同一个Notebook中）。

1.  **第一阶段：训练教师模型**
    * 找到并运行用于训练教师模型的脚本（如 `teacher_model_train.ipynb`）。
    * 该过程会在源域数据上进行训练，并保存一个最佳的教师模型检查点（例如，到 `./model/` 目录下）。

2.  **第二阶段：训练过渡模型**
    * 找到并运行核心的过渡模型训练脚本（例如，`distcare_adversal_tj.ipynb`）。
    * 该脚本会加载第一阶段保存的教师模型，并同时使用源域和目标域数据进行知识蒸馏和领域对抗训练。
    * 训练完成后，会保存一个最佳的过渡模型检查点。

3.  **第三阶段：微调与评估**
    * 找到并运行最终的K折交叉验证脚本。
    * 该脚本会：
        * 创建最终的目标模型 (`distcare_target`)。
        * 加载第二阶段保存的过渡模型，并调用`transfer_gru_dict`等函数执行知识迁移。
        * 在目标域数据上进行K折交叉验证的微调和评估。
        * 在运行结束后，**在屏幕和日志文件的末尾打印出最终的平均性能指标**。

### 2. 运行基线模型 (Baselines)

要复现论文中表格里的基线模型（如GRU, Transformer, DANN, TimeNet等）的结果，请运行对应的Notebook文件。
* 例如，要运行DANN模型在TJH数据集上的实验，请找到并运行 `dann_tj.ipynb`。
* 这些脚本通常是自包含的，包含了该模型从训练到评估的全部流程。

### 3. 绘制收敛曲线图

在运行完所有需要的实验并生成日志文件后，可以使用 `plot_convergence.py` 脚本来绘制模型的收敛速度对比图。
1.  打开 `plot_convergence.py` 文件。
2.  在 `MODELS_TO_PLOT` 字典中，配置您想要绘制的模型的日志文件路径 (`log_dir`) 和确切的文件名 (`filename`)。
3.  在终端运行 `python plot_convergence.py`。
4.  生成的图像将保存为 `model_convergence_comparison.png`。

## 引用我们的工作 (Citing Our Work)

如果您在您的研究中使用了我们的工作或代码，请引用我们的论文。


