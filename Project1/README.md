# 实验一：基于 TF-IDF 的新闻文本分类

本项目比较朴素贝叶斯、逻辑回归、线性与 RBF SVM、多层感知机，以及使用 SGD 训练的逻辑回归和线性 SVM。实验依次考察模型超参数、文本处理、特征预算、参数平均、随机划分稳定性和逐批训练损失，最后使用各模型的验证最优配置导出无标签测试集预测。

## 1. 环境安装

实验使用 Python 3.10，在 CPU 上运行，不需要 CUDA。推荐使用独立的 Conda 环境：

```bash
conda create -n project1 python=3.10 -y
conda activate project1
python -m pip install numpy==2.2.5 pandas==2.3.3 scipy==1.15.3 scikit-learn==1.7.2 nltk==3.10.3 matplotlib==3.10.9
```

已有环境时直接执行 `conda activate project1`。PorterStemmer 不需要下载 NLTK 语料。绘图使用 Matplotlib 的 `Agg` 后端，无需图形界面。

为减少不同线程设置造成的浮点差异，可在运行前设置：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

MLP 的逐批损失通过覆写 `MLPClassifier._backprop` 记录，实际计算仍由父类完成。该方法属于 scikit-learn 内部接口，因此复现实验时应使用上述版本。

## 2. 项目文件与代码结构

```text
Project1/
├── draft_main.py            # 公共数据、模型、评估、结果保存与基础参数扫描
├── explore.py               # 文本处理、SGD、特征预算、稳定性、绘图和最终预测
├── train_data.csv           # 标注数据：text、target
├── test_data_unlabeled.csv  # 无标签测试数据：text
├── sweep_results.csv        # 完整精度的配置与验证指标
├── results.md               # 按实验组展示的结果表
├── report.md                # 实验报告
├── predictions.csv         # 六模型的测试集预测
├── figures/
│   ├── loss_comparison.png  # 一行三图，400 dpi
│   ├── loss_comparison.pdf  # 矢量损失图
│   └── loss_curves.csv      # 逐 mini-batch 损失记录
├── TUNING.md                # 课程提供的参数说明
└── 实验一要求.pdf           # 实验要求与评分标准
```

两份 Python 文件的职责如下：

| 文件 | 主要职责 |
| --- | --- |
| `draft_main.py` | 加载数据；分层划分；训练集拟合 TF-IDF；构建 NB、LR、SVC、MLP；计算验证与训练指标；保存实验结果和预测；运行基础模型参数扫描 |
| `explore.py` | 定义文本处理；执行 SGD 训练循环；记录 MLP 批次损失；运行特征和随机划分对照；选择六模型最优配置并全量重训；提供完整实验入口 |

两份文件均通过 `if __name__ == '__main__'` 启动命令行任务，导入模块时不训练模型。数据和输出路径按脚本所在目录定位，不依赖启动命令的工作目录。

## 3. 数据划分与评估规则

- 标注样本 7368 条，类别标签为 0–9；测试样本 2457 条。
- 默认 `random_state=42`，按类别分层作 80%/20% 划分，得到训练集 5894 条、验证集 1474 条。
- 先划分再拟合 TF-IDF；验证文本只调用 `transform`。每个随机划分各自拟合词表和 IDF。
- TF-IDF 默认使用单词、最多 5000 个特征、L2 归一化。所有文本均转小写。
- 以验证集 Macro-F1 选择配置，同时记录 Accuracy、Macro-P、Macro-R、Train F1 与 F1 gap。
- 无标签测试集不参与调参。最终预测时，才使用全部标注样本拟合对应 TF-IDF 和模型。

`sweep_results.csv` 的字段为：

| 字段 | 含义 |
| --- | --- |
| `model` | `nb`、`lr`、`svm`、`mlp`、`lr_sgd` 或 `svm_sgd` |
| `params` | 模型参数字典；稳定性实验另记录 `split_seed` |
| `accuracy`、`precision`、`recall`、`f1` | 验证集 Accuracy 与宏平均 P、R、F1 |
| `train_f1` | 训练集 Macro-F1 |
| `f1_gap` | Train F1 − 验证集 Macro-F1 |
| `tfidf` | 文本处理与特征预算标识 |
| `exp` | `tuning`、`tfidf`、`header_control`、`feature_budget`、`seed_check` 或 `loss_figure`；可选基础基线为 `baseline` |

每次运行都会训练所选配置并写入本次指标。相同的实验、文本表示、模型和参数对应同一条记录，重复运行会更新该记录，不重复追加。写入使用临时文件与原子替换，避免留下写入不完整的 CSV。

## 4. 完整实验运行

在 `Project1` 目录下执行：

```bash
conda activate project1
python explore.py all
```

该命令按下表顺序运行全部默认实验，随后按六模型的最优配置全量训练并生成 `predictions.csv`：

| 阶段 | 默认配置数量 | 对应 results.md |
| --- | ---: | --- |
| LR、线性 SVM、RBF SVM、MLP 参数扫描 | 49 | 第 1–4 节 |
| 四个基础模型 × 五种文本处理 | 20 | 第 5–9 节 |
| SGD 的正则化、参数平均和文本表示对照 | 60 | 第 10–17 节 |
| RBF SVM 的三种 20000 维文本表示 | 3 | 第 18 节 |
| 五个划分 × SVM、MLP | 10 | 第 19 节 |
| LR-SGD、SVM-SGD、MLP 的逐批损失 | 3 | 第 20 节 |
| 六模型最优配置全量训练与测试集预测 | 不计入验证结果 | `predictions.csv` |

默认实验共 145 条验证记录。损失图的三条记录用于关联曲线与评估指标，不应作为独立的新配置证据。完整运行涉及多组 SVC 和 MLP 训练，耗时取决于 CPU 与线程设置。

## 5. 分别运行各实验

### 5.1 基础模型与参数扫描

```bash
python draft_main.py tune
python draft_main.py tune --model lr
python draft_main.py tune --model svm
python draft_main.py tune --model mlp
```

默认参数网格：

| 模型 | 参数范围 | 组数 |
| --- | --- | ---: |
| LR | C∈{0.1,1,10,100}；max_iter∈{1000,2000} | 8 |
| 线性 SVC | C∈{0.1,1,10,100,1000} | 5 |
| RBF SVC | 同上 5 个 C；gamma∈{scale,auto,0.1,1} | 20 |
| MLP | 隐藏层∈{(50,),(100,),(200,),(100,50)}；α∈{0.0001,0.001,0.01,0.1} | 16 |

MLP 固定 ReLU、Adam、max_iter=300。基础扫描会打印各模型最优结果、分类报告和混淆矩阵。

如需单独运行指导书的四模型基线：

```bash
python draft_main.py baseline
python draft_main.py baseline --model nb
```

基线使用 NB 默认参数、LR 默认 C 且 max_iter=1000、线性 SVC 的 C=1、MLP 的隐藏层 (100,)；其 `baseline` 记录不计入上面的 145 条默认实验。

### 5.2 文本处理对照

```bash
python explore.py cleaning
python explore.py cleaning --model svm
```

固定模型参数：NB 默认；LR 的 C=100、max_iter=1000；RBF SVM 的 C=1000、gamma=scale；MLP 的隐藏层 (100,)、α=0.001、max_iter=300。文本处理如下：

| 标识 | 处理方式 |
| --- | --- |
| `base` | 原始文本的单词 TF-IDF |
| `ngram12` | 单词与二元词组，ngram_range=(1,2) |
| `stop` | 删除英文停用词 |
| `stem` | Porter 词干化 |
| `nohdr_fixed` | 删除第一个空行之前的邮件头，并显式转小写 |

上述表示均最多 5000 维。自定义 `preprocessor` 会替代默认小写转换，因此邮件头处理函数需显式调用 `.lower()`；词干化 tokenizer 则接收已转小写的文本。

### 5.3 SGD 优化与文本表示对照

```bash
python explore.py sgd-sweep
python explore.py sgd-sweep --tfidf base
python explore.py sgd-sweep --tfidf stop --model lr_sgd
python explore.py sgd-sweep --tfidf nohdr_fixed
python explore.py sgd-sweep --tfidf stem
```

默认依次运行 `base`、`ngram12`、`stop`、`nohdr_fixed`、`stem`：

- 原始文本：LR-SGD 比较 α∈{1e−6,1e−5}，SVM-SGD 比较 α∈{1e−5,1e−4}，分别搭配 average=False/True，共 8 组。
- 二元词组与去停用词：两个模型分别比较 α∈{1e−6,1e−5,1e−4,1e−3,1e−2} 及参数平均开关，每种表示 20 组。
- 去邮件头与词干化：分别选择原始文本下 LR-SGD、SVM-SGD 验证 Macro-F1 前三的参数，每种表示 6 组。单独运行这两项前，须已完成 `sgd-sweep --tfidf base`。

`--top-k-base 3` 可显式指定迁移前三组；`--focused` 使用各模型两个 α 与平均开关的组合。

SGD 使用默认 L2 正则化和 `optimal` 学习率；α 同时参与正则化与学习率计算。训练循环为 100 epoch，每轮固定种子打乱，按 batch=128 调用 `partial_fit`，丢弃不足 128 条的尾部，保持各配置训练规则一致。`average` 是模型参数平均，会影响预测结果，与损失图的移动平均线不同。

### 5.4 特征预算与稳定性

```bash
python explore.py feature-budget
python explore.py seed-check
```

`feature-budget` 固定 RBF SVM 的 C=1000、gamma=scale，比较单词、单词＋二元词组、词干化单词的 20000 维表示。

`seed-check` 使用分层划分 seed 0–4。每个划分中，SVM 与 MLP 共享同一套 5000 维词干化 TF-IDF；模型超参数固定为基础参数扫描选出的组合，模型随机种子与划分 seed 相同。输出五次验证 F1 的均值和样本标准差。该实验比较固定配置的稳定性，并未在每个划分上重新调参。

### 5.5 损失曲线

训练并记录三个模型的逐批损失：

```bash
python explore.py loss-figure
```

仅使用已有 `figures/loss_curves.csv` 重绘，不重新训练：

```bash
python explore.py loss-figure --plot-only
```

指定图片路径：

```bash
python explore.py loss-figure --output figures/loss_comparison.png
```

绘图统一使用原始文本、5000 维 TF-IDF、默认分层划分。LR-SGD 的 α=1e−5，SVM-SGD 的 α=1e−4，均 average=True；MLP 使用 (100,)、α=0.001 和 Adam。前两者为 100 epoch、每 epoch 46 批，共各 4600 点；当前 MLP 在 165 epoch 停止，每 epoch 30 批，共 4950 点。

SGD 在每批更新后计算当前批次的 one-vs-rest 损失与 L2 项；MLP 记录反向传播时计算的批次损失。图中保留原始波动，并以一个 epoch 的批次数作为移动平均窗口。不同模型的损失定义、记录时点和批次大小不同，应比较各自收敛趋势，不按损失绝对值评价模型优劣。

曲线 CSV 包括 `model`、`tfidf`、`params`、`step`、`epoch`、`train_loss`、`loss_scope`。`step` 是批次更新计数，`epoch` 可为小数。PNG 为一行三图、400 dpi，同时输出矢量 PDF。

## 6. 最优配置与最终预测

```bash
python explore.py predict-final
```

默认处理全部六个模型：从 `sweep_results.csv` 中按验证集 Macro-F1 选择配置，排除 `seed_check` 与 `loss_figure`，避免用其他划分的结果选模或重复计入绘图评估。随后按各自最优文本处理和参数，在全部 7368 条标注样本上重新拟合 TF-IDF 和分类器，再预测测试集。

当前选定配置为：

| 预测列 | 文本表示 | 参数 | 选择依据：验证 Macro-F1 |
| --- | --- | --- | ---: |
| `nb` | stem，5000 维 | MultinomialNB 默认 | 0.89654 |
| `lr` | stop，5000 维 | C=100，max_iter=1000 | 0.91618 |
| `svm` | stem，20000 维 | RBF，C=1000，gamma=scale | **0.92478** |
| `mlp` | base，5000 维 | (100,)，α=0.001，max_iter=300，Adam | 0.91897 |
| `lr_sgd` | stop，5000 维 | log_loss，α=1e−5，average=True | 0.91957 |
| `svm_sgd` | base，5000 维 | hinge，α=1e−4，average=True | 0.91794 |

所有最终模型的随机种子为 42。这里的“最优”指已完成实验中的验证最优，并不表示全部可能参数空间的全局最优。

也可仅训练并更新指定模型列：

```bash
python explore.py predict-final --model svm
```

`predictions.csv` 不含索引，列顺序为 `nb,lr,svm,mlp,lr_sgd,svm_sgd`，共 2457 行，顺序与 `test_data_unlabeled.csv` 一致，值为 0–9 的类别标签。命令会检查输出行数与标签范围；已有预测文件时，还会打印各列与本次预测相比变化的条数。

损失图使用统一输入做收敛比较，而最终预测使用各模型分别选定的最佳输入，两者的用途不同。

## 7. 结果与复现注意事项

- `sweep_results.csv` 保留完整精度；`results.md` 展示五位小数，按表格各指标的最优值加粗，并列最优一起标出；F1 gap 按最小值标注，但不能单独用它选模。
- 训练过程中产生的结果由命令写入 CSV；Markdown 报告和结果表是实验结果的展示文档。
- 同一数据、参数、随机种子与依赖版本下应得到相同结果。不同数值库或线程设置可能使浮点计算、收敛停止点出现细微差异。
- 运行最优配置迁移与最终预测前，需要先获得相应验证记录；`python explore.py all` 按依赖关系依次执行。
- 若 Matplotlib 提示默认缓存目录不可写，可设置 `export MPLCONFIGDIR=/tmp/project1-matplotlib` 后运行绘图。
