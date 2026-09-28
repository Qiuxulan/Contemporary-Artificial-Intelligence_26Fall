# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import pandas as pd

# ---- 日志：把输出同时写进 logs/ 下的日志文件 ----
# 放在最前面，这样连下面的"数据加载成功"等信息也会被记录
import os
import sys
from datetime import datetime

LOG_TO_CONSOLE = True          # True=终端和日志都输出；False=只写日志，不动终端

class Tee:
    """把 stdout/stderr 同时写到终端和日志文件"""
    def __init__(self, path, console=True):
        self.console = console
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout            # 先保存原始终端输出，避免递归
    def write(self, text):
        self.file.write(text)
        if self.console:
            self.stdout.write(text)
    def flush(self):
        self.file.flush()
        if self.console:
            self.stdout.flush()

os.makedirs('logs', exist_ok=True)
LOG_PATH = os.path.join('logs', f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
_log = Tee(LOG_PATH, console=LOG_TO_CONSOLE)
sys.stdout = _log
sys.stderr = _log               # sklearn 的 ConvergenceWarning 等也会进日志

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

# 2. 调用函数来获取数据
#    这个函数会自动读取 .csv 文件并返回你需要的所有内容
X_train, y_train, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# --- 在这里开始你的实验！ ---
# 现在，你可以使用 X_train, y_train, 和 X_test_unlabeled 这三个变量
# 来进行后续的特征提取、模型训练和预测了。

# 举例：（下面这段是原文件的示例代码，保留在此但已注释掉：
#       它只用单个模型，且最后会用单列结果覆盖 predictions.csv，
#       与"四个模型各占一列、互不覆盖"的要求冲突。实际流程见文件末尾。）
# # 1. 创建TF-IDF向量化器
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
#
# # 2. 训练一个模型...
# from sklearn.svm import SVC
# svm_model = SVC()
# svm_model.fit(X_train_tfidf, y_train)
#
# # 3. 对测试集进行预测...
# X_test_tfidf = vectorizer.transform(X_test_unlabeled)
# predictions = svm_model.predict(X_test_tfidf)
#
# # 4. 保存你的预测结果...
# import pandas as pd
# pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)

# =====================================================================
# 实验一：文本分类
#   四个模型：朴素贝叶斯 / 逻辑回归 / 支持向量机 / 多层感知机
#   流程：划分验证集 -> TF-IDF 特征 -> 训练 -> 验证集评估 -> 预测测试集
#   说明：sklearn 只支持 CPU，本数据集仅 7k 余条，CPU 上即可快速跑完。
# =====================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

# ---- 全局配置（调参 / 快速迭代只需改这里）----
SEED     = 42        # 固定随机种子，保证结果可复现
VAL_SIZE = 0.2       # 验证集比例
OUT_CSV  = 'predictions.csv'
SWEEP_CSV = 'sweep_results.csv'   # 每组参数的验证集指标，存下来方便画调参曲线

# 只跑这里列出的模型：做参数搜索时只留一个，其余跳过，省时间
# 全跑就是 ['nb', 'lr', 'svm', 'mlp']
RUN_MODELS = ['nb', 'lr', 'svm', 'mlp', 'lr_sgd', 'svm_sgd']

# 这两个模型改用「逐轮训练」的方式，顺便记录训练损失曲线
# （sklearn 1.7 已移除 SGDClassifier.loss_curve_，只能自己按 epoch 记录）
LOSS_CURVE_MODELS = ['lr_sgd', 'svm_sgd']
SGD_EPOCHS = 100          # 逐轮训练的轮数；看曲线尾部是否走平，没走平就加大
SGD_BATCH  = 128          # 小批量大小：越小，损失曲线的抖动越明显（记录点也越多）

# ---- TF-IDF 配置（本轮实验的自变量）----
# 每个变体相对 base 只改一个参数，符合"一次只改变一个因素"的实验规范
# 其他可探索的 TfidfVectorizer 参数（想试就在下面加一条，注意仍是"只改一个"）：
#   'sublinear_tf': True      用 1+log(tf) 代替原始 tf，抑制高频词
#   'max_df': 0.5             丢掉出现在超过 50% 文档里的词（数据驱动的"停用词"）
#   'min_df': 2               丢掉只出现在 1 篇文档里的词
#   'strip_accents': 'unicode' 去掉重音符号
def strip_headers(text):
    """只保留正文：丢掉第一个空行之前的邮件头（From/Subject/Organization 等）。
       用 [-1] 是为了兼容没有空行的帖子 —— 那样会原样返回，不会报错。"""
    return text.split('\n\n', 1)[-1]

import re
from nltk.stem import PorterStemmer

_token_re = re.compile(r"(?u)\b\w\w+\b")     # 与 TfidfVectorizer 默认分词规则保持一致
_stemmer  = PorterStemmer()

def stem_tokenizer(text):
    """先按默认规则分词，再对每个词做 Porter 词干化（running/runs -> run）"""
    return [_stemmer.stem(t) for t in _token_re.findall(text)]

TFIDF_CONFIGS = {
    'base':    {'max_features': 5000},                          # 现有基线，用于对照
    'ngram12': {'max_features': 5000, 'ngram_range': (1, 2)},   # 加入二元词组
    'stop':    {'max_features': 5000, 'stop_words': 'english'}, # 去除英文停用词
    'nohdr':   {'max_features': 5000, 'preprocessor': strip_headers},  # 去掉邮件头
    'stem':    {'max_features': 5000, 'tokenizer': stem_tokenizer},    # Porter 词干化
}
# 本轮比对三种清洗：base（原始）、nohdr（去邮件头）、stem（词干化）
TFIDF_TO_USE = ['base', 'nohdr', 'stem']

# 本次实验的阶段标签：写进 sweep_results.csv，用来区分"调参阶段"和"换表示阶段"的记录，
# 避免这次的结果把之前的调参历史（C/gamma/alpha 曲线）覆盖掉
EXP_TAG = 'tfidf'

# 把配置写进日志，方便复现（实验规范要求记录所有实验配置）
print(f"[配置] SEED={SEED}  VAL_SIZE={VAL_SIZE}  OUT_CSV={OUT_CSV}\n"
      f"[配置] RUN_MODELS={RUN_MODELS}\n"
      f"[配置] TFIDF_TO_USE={TFIDF_TO_USE}\n")

# ---- 1. 划分训练集 / 验证集（分层抽样，保持 10 类比例不变）----
# 注意顺序：先划分，再用「仅训练集」拟合 TF-IDF，避免验证集信息泄露
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED)
print(f"训练集: {len(X_tr)}  验证集: {len(X_val)}  测试集: {len(X_test_unlabeled)}\n")

# ---- 2. 模型类与参数网格 ----
MODEL_CLASSES = {
    'nb':  MultinomialNB,
    'lr':  LogisticRegression,
    'svm': SVC,
    'mlp': MLPClassifier,
    # 下面两个都是 SGDClassifier，只是损失函数不同：
    #   log_loss = 逻辑回归(SGD 版)   hinge = 线性 SVM(SGD 版)
    'lr_sgd':  SGDClassifier,
    'svm_sgd': SGDClassifier,
}

# ---- LR-SGD / SVM-SGD 调参范围 ----
# 注意 alpha 与 SVC 的 C 方向相反：alpha 越大正则越强（SVC 是 C 越大正则越弱）
# 所以 SVC 那边最优是 C=1000（弱正则），这里对应的 alpha 应该在很小的一端
SGD_ALPHA   = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
SGD_AVERAGE = [False, True]        # average=True 用平均权重，通常更稳

LR_SGD_GRID  = [{'loss': 'log_loss', 'alpha': a, 'average': av, 'random_state': SEED}
                for a in SGD_ALPHA for av in SGD_AVERAGE]
SVM_SGD_GRID = [{'loss': 'hinge', 'alpha': a, 'average': av, 'random_state': SEED}
                for a in SGD_ALPHA for av in SGD_AVERAGE]

# ---- SVM 调参范围（参考 TUNING.md）----
# C 对两种核都有效；gamma 只对 rbf 等非线性核有效，linear 核下会被静默忽略，
# 所以两种核分开构造，避免用 linear 把每个 gamma 重复跑一遍
SVM_KERNELS = ['linear', 'rbf']          # 要扫哪些核
SVM_C       = [0.1, 1, 10, 100, 1000]    # 按数量级取；1000 用来确认 C 是否已到平台
SVM_GAMMA   = ['scale', 'auto', 0.1, 1]  # 仅 rbf 有效

SVM_GRID = []
if 'linear' in SVM_KERNELS:
    SVM_GRID += [{'kernel': 'linear', 'C': c, 'random_state': SEED} for c in SVM_C]
if 'rbf' in SVM_KERNELS:
    SVM_GRID += [{'kernel': 'rbf', 'C': c, 'gamma': g, 'random_state': SEED}
                 for c in SVM_C for g in SVM_GAMMA]

# ---- MLP 调参范围（参考 TUNING.md）----
# TUNING.md 给了两个可调轴：网络结构、L2 正则 alpha；
# activation 按它的建议保持默认 relu，max_iter 用实验指导书的 300
MLP_HIDDEN = [(50,), (100,), (200,), (100, 50)]   # 从简单 -> 更宽 / 更深
MLP_ALPHA  = [0.0001, 0.001, 0.01, 0.1]           # TUNING.md 建议的四个数量级

# 结构 × alpha 的交叉网格：能看出「网络越宽越深是否越需要强正则」这一交互
MLP_GRID = [{'hidden_layer_sizes': h, 'alpha': a,
             'max_iter': 300, 'random_state': SEED}
            for h in MLP_HIDDEN for a in MLP_ALPHA]

# ---- LR 调参范围（参考 TUNING.md）----
# TUNING.md 说 C 是逻辑回归最重要的参数（正则化强度的"倒数"，越小正则越强），
# 建议按数量级扫；max_iter 它说默认够用、遇到 ConvergenceWarning 再调大
LR_C        = [0.1, 1, 10, 100]
LR_MAX_ITER = [1000, 2000]

# 两档 max_iter 同时跑还有个体检作用：同一个 C 下若两档结果一致，
# 说明 1000 次迭代已经收敛，TUNING.md 说的 ConvergenceWarning 不会出现
LR_GRID = [{'C': c, 'max_iter': mi, 'random_state': SEED}
           for c in LR_C for mi in LR_MAX_ITER]

# 本轮实验比较的是 TF-IDF 配置，所以每个模型只跑【之前调优得到的最优参数】这一组，
# 让模型侧保持不变、只有特征表示在变。想回去扫参数就把值换回 LR_GRID / SVM_GRID / MLP_GRID
PARAM_GRIDS = {
    'nb':  [{}],
    # LR 最优：C=100，验证集 macro-F1 0.9149（max_iter 1000/2000 结果相同，取快的）
    'lr':  [{'C': 100, 'max_iter': 1000, 'random_state': SEED}],
    # SVM 最优：rbf C=1000 gamma='scale'，验证集 macro-F1 0.9154
    'svm': [{'kernel': 'rbf', 'C': 1000, 'gamma': 'scale', 'random_state': SEED}],
    # MLP 最优：(100,) alpha=0.001，验证集 macro-F1 0.9190
    'mlp': [{'hidden_layer_sizes': (100,), 'alpha': 0.001, 'max_iter': 300, 'random_state': SEED}],
    # SGD 两个模型：用上轮扫出来的最优（想重新扫就把值换回 LR_SGD_GRID / SVM_SGD_GRID）
    'lr_sgd':  [{'loss': 'log_loss', 'alpha': 1e-6, 'average': False, 'random_state': SEED}],
    'svm_sgd': [{'loss': 'hinge', 'alpha': 1e-4, 'average': False, 'random_state': SEED}],
}

# ---- SGD 类的损失函数（sklearn 已移除 loss_curve_，按官方目标函数手工计算）----
def sgd_objective(clf, X, y):
    """sklearn SGD 的目标函数：(1/n)Σ loss + alpha * 0.5 * ||w||^2"""
    scores = clf.decision_function(X)                    # (n, n_classes)
    Y = np.zeros_like(scores)
    Y[np.arange(len(y)), np.searchsorted(clf.classes_, y)] = 1
    if clf.loss == 'hinge':
        per = np.maximum(0, 1 - (2 * Y - 1) * scores)    # 多分类按 one-vs-rest 求和
    else:                                                # log_loss
        p = 1.0 / (1.0 + np.exp(-scores))
        per = -(Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12))
    return per.sum(axis=1).mean() + clf.alpha * 0.5 * (clf.coef_ ** 2).sum()


def fit_with_loss_curve(model, X, y, epochs, batch_size):
    """小批量逐批训练，每批更新后记一次损失 —— 这样才保留 SGD 的抖动细节。
       注意 partial_fit 本身不会打乱数据，所以这里每个 epoch 手动 shuffle。"""
    classes, losses = np.unique(y), []
    n, rng = X.shape[0], np.random.RandomState(SEED)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n - batch_size + 1, batch_size):
            idx = order[s:s + batch_size]
            model.partial_fit(X[idx], y[idx], classes=classes)
            # 只算当前小批的损失，才看得到微小波动（算全量会被平均掉）
            losses.append(sgd_objective(model, X[idx], y[idx]))
    return losses

# ---- 3. 遍历 TF-IDF 配置 -> 模型 -> 参数组 ----
# 指标一律在验证集上算；训练集指标只用于诊断过拟合
results, test_preds = {}, {}
best_curves = {}         # 模型 -> 最优配置的训练损失曲线
sweep_records = []       # 每个 (tfidf, 模型, 参数) 组合的结果，最后存 CSV
for tf_name in TFIDF_TO_USE:
    tf_kwargs = TFIDF_CONFIGS[tf_name]
    print(f"{'#'*60}\n# TF-IDF 配置: {tf_name}   {tf_kwargs}\n{'#'*60}")

    # 每个配置都重新拟合 vectorizer，且仍然只在训练集上 fit（避免信息泄露）
    vec      = TfidfVectorizer(**tf_kwargs)
    X_tr_t   = vec.fit_transform(X_tr)
    X_val_t  = vec.transform(X_val)
    X_test_t = vec.transform(X_test_unlabeled)
    print(f"实际特征维度: {X_tr_t.shape[1]}\n")

    for name in RUN_MODELS:
        grids = PARAM_GRIDS[name]
        for i, params in enumerate(grids, 1):
            model = MODEL_CLASSES[name](**params)
            if name in LOSS_CURVE_MODELS:
                # SGD 类模型：逐轮训练，顺带拿到损失曲线
                losses = fit_with_loss_curve(model, X_tr_t, y_tr, SGD_EPOCHS, SGD_BATCH)
            else:
                model.fit(X_tr_t, y_tr)
                losses = None

            # 验证集指标：模型的真实表现，用于比较和选优
            y_val_pred = model.predict(X_val_t)
            acc = accuracy_score(y_val, y_val_pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_val, y_val_pred, average='macro', zero_division=0)

            # 训练集指标：只用来诊断过拟合，不能当作模型性能汇报
            tr_f1 = precision_recall_fscore_support(
                y_tr, model.predict(X_tr_t), average='macro', zero_division=0)[2]

            print(f"[{tf_name} | {name} {i}/{len(grids)}] {params}")
            print(f"    验证集 accuracy={acc:.4f} macro-P={p:.4f} macro-R={r:.4f} macro-F1={f1:.4f}")
            print(f"    训练集 macro-F1={tr_f1:.4f}   差距={tr_f1 - f1:+.4f}")
            sweep_records.append({'exp': EXP_TAG, 'tfidf': tf_name, 'model': name,
                                  'params': str(params), 'accuracy': acc,
                                  'precision': p, 'recall': r, 'f1': f1,
                                  'train_f1': tr_f1, 'f1_gap': tr_f1 - f1})

            # 记录该模型跨所有 TF-IDF 配置的全局最优，并顺手预测测试集
            if name not in results or f1 > results[name]['f1']:
                results[name] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1,
                                 'tfidf': tf_name, 'params': str(params),
                                 'y_val_pred': y_val_pred}
                test_preds[name] = model.predict(X_test_t)
                if losses is not None:
                    best_curves[name] = losses      # 只保留最优配置的损失曲线
                print(f"    ↑ {name} 新的全局最优（TF-IDF={tf_name}）")
    print()

# ---- 4. 结果汇总 ----
# 保存所有组合的明细：合并进已有文件。按 (阶段, 模型, TF-IDF) 三元组去重，
# 所以之前调参阶段的记录会原样保留，不会被这次实验覆盖
sweep_new = pd.DataFrame(sweep_records)
if os.path.exists(SWEEP_CSV):
    try:
        old_sweep = pd.read_csv(SWEEP_CSV)
        # 兼容早期没有这两列的记录：早期都是 max_features=5000，属 base 配置、调参阶段
        if 'tfidf' not in old_sweep.columns:
            old_sweep['tfidf'] = 'base'
        old_sweep['tfidf'] = old_sweep['tfidf'].fillna('base')
        if 'exp' not in old_sweep.columns:
            old_sweep['exp'] = 'tuning'
        old_sweep['exp'] = old_sweep['exp'].fillna('tuning')

        # 只丢掉与本次完全同 (阶段, 模型, TF-IDF) 的旧记录，其余历史全部保留
        keys_new = set(zip(sweep_new['exp'], sweep_new['model'], sweep_new['tfidf']))
        keep = [(e, m, t) not in keys_new for e, m, t in
                zip(old_sweep['exp'], old_sweep['model'], old_sweep['tfidf'])]
        old_sweep = old_sweep[keep]
        sweep_new = pd.concat([old_sweep, sweep_new], ignore_index=True)
    except Exception as e:
        print(f"[警告] 读取 {SWEEP_CSV} 失败({e})，改为整体重写")

sweep_new = sweep_new.sort_values(['exp', 'model', 'tfidf', 'f1'],
                                  ascending=[True, True, True, False])
sweep_new.to_csv(SWEEP_CSV, index=False)
print(f"参数扫描明细已写入 {SWEEP_CSV}，共 {len(sweep_new)} 行"
      f"（含历史记录，未覆盖）\n")

# 核心对比表：TF-IDF 配置 × 模型 -> 最优验证集 macro-F1（只看本次实验）
cur = sweep_new[sweep_new['exp'] == EXP_TAG]
pivot = cur.pivot_table(index='tfidf', columns='model', values='f1', aggfunc='max')
print("--- 验证集 macro-F1：TF-IDF 配置 × 模型 ---")
print(pivot.round(4))
print()

print("--- 各模型全局最优（跨所有 TF-IDF 配置）---")
for name in RUN_MODELS:
    r = results[name]
    print(f"  {name:4s} macro-F1={r['f1']:.4f}  accuracy={r['accuracy']:.4f}  "
          f"(TF-IDF={r['tfidf']}, params={r['params']})")
print()

# 各模型全局最优的详细报告（均为验证集结果）
for name in RUN_MODELS:
    r = results[name]
    print(f"===== {name} 全局最优的验证集详细结果（TF-IDF={r['tfidf']}）=====")
    print(classification_report(y_val, r['y_val_pred'], digits=4, zero_division=0))
    print("混淆矩阵:")
    print(confusion_matrix(y_val, r['y_val_pred']))
    print()

# 保存 SGD 类模型的训练损失曲线
if best_curves:
    import matplotlib
    matplotlib.use('Agg')            # 无显示环境，直接存成图片
    import matplotlib.pyplot as plt
    for name, curve in best_curves.items():
        curve = np.asarray(curve)
        plt.figure()
        # 原始逐批损失：抖动全部保留
        # 图例用英文：matplotlib 默认字体没有中文，写中文会变成方块
        plt.plot(range(1, len(curve) + 1), curve, lw=0.6, alpha=0.55,
                 label='per-batch loss')
        # 叠加滑动平均，方便看整体趋势（只是画图时算，不影响原始数据）
        w = max(1, len(curve) // 100)
        if w > 1:
            trend = np.convolve(curve, np.ones(w) / w, mode='valid')
            plt.plot(range(w, len(curve) + 1), trend, lw=1.8,
                     label=f'moving average (window {w})')
        plt.xlabel('mini-batch update'); plt.ylabel('objective')
        plt.title(f'{name} training loss (TF-IDF={results[name]["tfidf"]}, '
                  f'batch={SGD_BATCH}, epochs={SGD_EPOCHS})')
        plt.legend()
        plt.savefig(f'loss_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"损失曲线已保存: loss_{name}.png  （{len(curve)} 个点）")
    print()

# ---- 5. 保存预测结果：合并进已有文件，本次没跑的模型列保持不变 ----
new_df = pd.DataFrame(test_preds)       # 列名即模型名
if os.path.exists(OUT_CSV):
    try:
        old_df = pd.read_csv(OUT_CSV)
        if len(old_df) == len(new_df):
            # 行数一致才合并：只覆盖本次跑过的列，其余原样保留
            for col in new_df.columns:
                old_df[col] = new_df[col]
            new_df = old_df
        else:
            print(f"[警告] {OUT_CSV} 有 {len(old_df)} 行，本次预测 {len(new_df)} 行，"
                  f"行数不一致，改为整体重写")
    except Exception as e:
        print(f"[警告] 读取 {OUT_CSV} 失败({e})，改为整体重写")

new_df.to_csv(OUT_CSV, index=False)
print(f"预测结果已写入 {OUT_CSV}，列: {list(new_df.columns)}，共 {len(new_df)} 行")

# ---- 6. 收尾：恢复终端输出，关闭日志 ----
sys.stdout, sys.stderr = _log.stdout, _log.stdout
_log.file.close()
print(f"日志已保存到 {LOG_PATH}")