"""文本分类的公共训练组件、四模型基线与超参数扫描。

示例：python draft_main.py tune --model svm
"""

import argparse
import ast
import csv
import os
from pathlib import Path
import tempfile

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
SWEEP_CSV = ROOT / 'sweep_results.csv'
PREDICTIONS_CSV = ROOT / 'predictions.csv'
SEED = 42
VAL_SIZE = 0.2
MAX_FEATURES = 5000
MODELS = ('nb', 'lr', 'svm', 'mlp')
RESULT_COLUMNS = ('model', 'params', 'accuracy', 'precision', 'recall',
                  'f1', 'train_f1', 'f1_gap', 'tfidf', 'exp')


def load_data():
    """返回标注文本、标签和按文件顺序排列的无标签测试文本。"""
    train_df = pd.read_csv(ROOT / 'train_data.csv')
    test_df = pd.read_csv(ROOT / 'test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled


def split_data(seed=SEED):
    X_train, y_train, _ = load_data()
    return train_test_split(X_train, y_train, test_size=VAL_SIZE,
                            stratify=y_train, random_state=seed)


def vectorize(X_tr, X_val, **kwargs):
    """先划分，再仅用训练子集拟合词表和 IDF。"""
    vec = TfidfVectorizer(**kwargs)
    X_tr_t = vec.fit_transform(X_tr)
    return vec, X_tr_t, vec.transform(X_val)


MODEL_CLASSES = {
    'nb': MultinomialNB,
    'lr': LogisticRegression,
    'svm': SVC,
    'mlp': MLPClassifier,
}


def baseline_params(seed=SEED):
    """PDF 给出的四个基础模型。"""
    return {
        'nb': {},
        'lr': {'max_iter': 1000, 'random_state': seed},
        'svm': {'kernel': 'linear', 'C': 1.0, 'random_state': seed},
        'mlp': {'hidden_layer_sizes': (100,), 'max_iter': 300,
                'random_state': seed},
    }


def selected_params(seed=SEED):
    """基础扫描选出的固定参数，用于控制文本表示比较中的模型条件。"""
    return {
        'nb': {},
        'lr': {'C': 100, 'max_iter': 1000, 'random_state': seed},
        'svm': {'kernel': 'rbf', 'C': 1000, 'gamma': 'scale',
                'random_state': seed},
        'mlp': {'hidden_layer_sizes': (100,), 'alpha': 0.001,
                'max_iter': 300, 'random_state': seed},
    }


def tuning_grids(seed=SEED):
    """LR 8 组、SVM 25 组、MLP 16 组超参数组合。"""
    lr = [{'C': c, 'max_iter': steps, 'random_state': seed}
          for c in [0.1, 1, 10, 100] for steps in [1000, 2000]]
    svm = [{'kernel': 'linear', 'C': c, 'random_state': seed}
           for c in [0.1, 1, 10, 100, 1000]]
    svm += [{'kernel': 'rbf', 'C': c, 'gamma': gamma,
             'random_state': seed}
            for c in [0.1, 1, 10, 100, 1000]
            for gamma in ['scale', 'auto', 0.1, 1]]
    mlp = [{'hidden_layer_sizes': hidden, 'alpha': alpha,
            'max_iter': 300, 'random_state': seed}
           for hidden in [(50,), (100,), (200,), (100, 50)]
           for alpha in [0.0001, 0.001, 0.01, 0.1]]
    return {'lr': lr, 'svm': svm, 'mlp': mlp}


def make_model(name, params):
    return MODEL_CLASSES[name](**params)


def result_row(exp, tfidf, name, params, model, X_tr_t, y_tr, X_val_t, y_val):
    """统一记录验证指标；训练 F1 只用于分析拟合差距。"""
    pred = model.predict(X_val_t)
    p, r, f1, _ = precision_recall_fscore_support(
        y_val, pred, average='macro', zero_division=0)
    train_f1 = precision_recall_fscore_support(
        y_tr, model.predict(X_tr_t), average='macro', zero_division=0)[2]
    row = {'model': name, 'params': str(params),
           'accuracy': accuracy_score(y_val, pred), 'precision': p,
           'recall': r, 'f1': f1, 'train_f1': train_f1,
           'f1_gap': train_f1 - f1, 'tfidf': tfidf, 'exp': exp}
    return row, pred


def record_key(row):
    """用完整配置去重，保留同一模型的不同参数扫描记录。"""
    params = ast.literal_eval(row['params'])
    return (row['exp'], row['tfidf'], row['model'],
            repr(sorted(params.items())))


def save_result(row):
    """按完整配置写入结果；重复运行同一配置时更新其指标。"""
    key = record_key(row)
    rows = []
    if SWEEP_CSV.exists():
        with SWEEP_CSV.open(newline='', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
    for index, previous in enumerate(rows):
        if record_key(previous) == key:
            rows[index] = row
            break
    else:
        rows.append(row)
    fd, path = tempfile.mkstemp(prefix='.results_', suffix='.csv', dir=ROOT)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS,
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(path, SWEEP_CSV)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def save_predictions(new_columns):
    """原子保存预测，按传入的模型列更新。"""
    new = pd.DataFrame(new_columns)
    if PREDICTIONS_CSV.exists():
        old = pd.read_csv(PREDICTIONS_CSV)
        if len(old) != len(new):
            raise ValueError('predictions.csv 行数与测试集不一致，已停止写入')
        for name in new:
            old[name] = new[name]
        new = old
    fd, path = tempfile.mkstemp(prefix='.predictions_', suffix='.csv', dir=ROOT)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as file:
            new.to_csv(file, index=False)
        os.replace(path, PREDICTIONS_CSV)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def run(args):
    grids = ({name: [params] for name, params in baseline_params().items()}
             if args.stage == 'baseline' else tuning_grids())
    exp = 'baseline' if args.stage == 'baseline' else 'tuning'
    names = tuple(grids) if args.model == 'all' else (args.model,)
    if any(name not in grids for name in names):
        raise ValueError('NB 使用默认参数，请运行 baseline --model nb 或 explore.py cleaning')
    X_tr, X_val, y_tr, y_val = split_data(SEED)
    _, X_tr_t, X_val_t = vectorize(X_tr, X_val, max_features=MAX_FEATURES)
    print(f'训练 {len(X_tr)}，验证 {len(X_val)}；TF-IDF {X_tr_t.shape[1]} 维')
    best = {}
    for name in names:
        for i, params in enumerate(grids[name], 1):
            model = make_model(name, params)
            model.fit(X_tr_t, y_tr)
            row, pred = result_row(exp, 'base', name, params, model,
                                   X_tr_t, y_tr, X_val_t, y_val)
            save_result(row)
            print(f'{name} {i}/{len(grids[name])} {params}  '
                  f'accuracy={row["accuracy"]:.4f} macro-F1={row["f1"]:.4f}')
            if name not in best or row['f1'] > best[name][0]['f1']:
                best[name] = (row, pred)
    for name, (row, pred) in best.items():
        print(f'\n{name} 验证最优：{row["params"]}, macro-F1={row["f1"]:.4f}')
        print(classification_report(y_val, pred, digits=4, zero_division=0))
        print('混淆矩阵：\n', confusion_matrix(y_val, pred))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['baseline', 'tune'])
    parser.add_argument('--model', choices=(*MODELS, 'all'), default='all')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
