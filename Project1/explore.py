"""扩展实验：文本处理、SGD 优化、稳定性检查和损失曲线。

示例：python explore.py all
"""

import argparse
import ast
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from draft_main import (MAX_FEATURES, ROOT, SEED, MODELS, PREDICTIONS_CSV,
                        load_data, make_model, record_key, result_row,
                        run as run_basic, save_predictions, save_result,
                        selected_params, split_data, vectorize)


SGD_EPOCHS = 100
SGD_BATCH = 128
SGD_MODELS = ('lr_sgd', 'svm_sgd')
TOKEN_RE = re.compile(r'(?u)\b\w\w+\b')


def strip_headers_lower(text):
    """自定义预处理会绕过默认 lowercase，因此在这里显式转小写。"""
    return text.split('\n\n', 1)[-1].lower()


@lru_cache(maxsize=1)
def porter_stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def stem_tokenizer(text):
    """对默认规则识别出的单词进行 Porter 词干化。"""
    stemmer = porter_stemmer()
    return [stemmer.stem(token) for token in TOKEN_RE.findall(text)]


TFIDF_CONFIGS = {
    'base': {'max_features': MAX_FEATURES},
    'ngram12': {'max_features': MAX_FEATURES, 'ngram_range': (1, 2)},
    'stop': {'max_features': MAX_FEATURES, 'stop_words': 'english'},
    'stem': {'max_features': MAX_FEATURES, 'tokenizer': stem_tokenizer,
             'token_pattern': None},
    'nohdr_fixed': {'max_features': MAX_FEATURES,
                    'preprocessor': strip_headers_lower},
}


def sgd_params(name, alpha, average, seed=SEED):
    return {'loss': 'log_loss' if name == 'lr_sgd' else 'hinge',
            'alpha': alpha, 'average': average, 'random_state': seed}


def selected_sgd_params(seed=SEED):
    """原始 TF-IDF 参数比较选出的两组 SGD 配置。"""
    return {
        'lr_sgd': sgd_params('lr_sgd', 1e-5, True, seed),
        'svm_sgd': sgd_params('svm_sgd', 1e-4, True, seed),
    }


def sgd_objective(model, X, y):
    """多分类 one-vs-rest 的训练目标；用于同一模型内观察收敛。"""
    scores = model.decision_function(X)
    positive = (y[:, None] == model.classes_[None, :])
    signed = np.where(positive, 1.0, -1.0) * scores
    if model.loss == 'log_loss':
        per = np.logaddexp(0.0, -signed)
    else:
        per = np.maximum(0.0, 1.0 - signed)
    return per.sum(axis=1).mean() + 0.5 * model.alpha * np.square(model.coef_).sum()


def train_sgd(model, X, y, seed=SEED, collect_loss=False):
    """训练 100 epoch；每轮打乱，使用 128 条完整批次并丢弃尾部。"""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    curve = []
    batches_per_epoch = len(y) // SGD_BATCH
    for epoch in range(SGD_EPOCHS):
        order = rng.permutation(len(y))
        for start in range(0, len(y) - SGD_BATCH + 1, SGD_BATCH):
            idx = order[start:start + SGD_BATCH]
            model.partial_fit(X[idx], y[idx], classes=classes)
            if collect_loss:
                position = epoch + (start // SGD_BATCH + 1) / batches_per_epoch
                curve.append((position, sgd_objective(model, X[idx], y[idx])))
    return curve


class BatchLossMLPClassifier(MLPClassifier):
    """记录 Adam 每个批次反向传播时的真实损失，保持 fit 的更新与停止规则。"""

    def _backprop(self, *args, **kwargs):
        result = super()._backprop(*args, **kwargs)
        self.batch_losses_.append(float(result[0]))
        return result

    def fit(self, X, y, **kwargs):
        self.batch_losses_ = []
        return super().fit(X, y, **kwargs)


def tfidf_config(name):
    """返回文本表示配置，_20k 表示最多保留 20000 个特征。"""
    if name.endswith('_20k'):
        return dict(TFIDF_CONFIGS[name[:-4]], max_features=20000)
    return dict(TFIDF_CONFIGS[name])


def run_cleaning(args):
    """固定模型参数，比较五种文本处理。"""
    selected = selected_params()
    names = MODELS if args.model == 'all' else (args.model,)
    X_tr, X_val, y_tr, y_val = split_data()
    for tfidf, kwargs in TFIDF_CONFIGS.items():
        exp = 'header_control' if tfidf == 'nohdr_fixed' else 'tfidf'
        _, X_tr_t, X_val_t = vectorize(X_tr, X_val, **kwargs)
        for name in names:
            params = selected[name]
            model = make_model(name, params)
            model.fit(X_tr_t, y_tr)
            row, _ = result_row(exp, tfidf, name, params, model,
                                X_tr_t, y_tr, X_val_t, y_val)
            save_result(row)
            print(f'{tfidf} {name}: accuracy={row["accuracy"]:.5f}, '
                  f'macro-F1={row["f1"]:.5f}', flush=True)


def sgd_configs(tfidf, names=SGD_MODELS, top_k=None, focused=False):
    """默认：原始文本 8 组，词组和停用词各 20 组，邮件头和词干化各 6 组。"""
    if top_k is None and tfidf in ('nohdr_fixed', 'stem'):
        top_k = 3
    if top_k is not None:
        rows = pd.read_csv(ROOT / 'sweep_results.csv')
        configs = []
        for name in names:
            candidates = rows[(rows['exp'] == 'tfidf') &
                              (rows['tfidf'] == 'base') &
                              (rows['model'] == name)]
            candidates = candidates.sort_values('f1', ascending=False,
                                                 kind='stable').drop_duplicates('params')
            if len(candidates) < top_k:
                raise ValueError(f'{name}: 请先运行 sgd-sweep --tfidf base')
            configs.extend((name, ast.literal_eval(row.params))
                           for row in candidates.head(top_k).itertuples())
        return configs
    alpha_grid = ({'lr_sgd': [1e-6, 1e-5], 'svm_sgd': [1e-5, 1e-4]}
                  if focused or tfidf == 'base' else
                  {name: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] for name in SGD_MODELS})
    return [(name, sgd_params(name, alpha, average))
            for name in names for alpha in alpha_grid[name]
            for average in [False, True]]


def run_sgd_sweep(args):
    names = SGD_MODELS if args.model == 'all' else (args.model,)
    representations = (('base', 'ngram12', 'stop', 'nohdr_fixed', 'stem')
                       if args.tfidf == 'all' else (args.tfidf,))
    for tfidf in representations:
        configs = sgd_configs(tfidf, names, args.top_k_base, args.focused)
        run_sgd_group(tfidf, configs)


def run_sgd_group(tfidf, configs):
    X_tr, X_val, y_tr, y_val = split_data()
    _, X_tr_t, X_val_t = vectorize(X_tr, X_val, **tfidf_config(tfidf))
    for name, params in configs:
        model = SGDClassifier(**params)
        train_sgd(model, X_tr_t, y_tr)
        row, _ = result_row('tfidf', tfidf, name, params, model,
                            X_tr_t, y_tr, X_val_t, y_val)
        save_result(row)
        print(f'{tfidf} {name} {params}: macro-F1={row["f1"]:.5f}', flush=True)


def run_feature_budget(args):
    """在相同模型参数下比较三种 20000 维文本表示。"""
    X_tr, X_val, y_tr, y_val = split_data()
    params = selected_params()['svm']
    for tfidf in ('base_20k', 'ngram12_20k', 'stem_20k'):
        _, X_tr_t, X_val_t = vectorize(X_tr, X_val, **tfidf_config(tfidf))
        model = make_model('svm', params)
        model.fit(X_tr_t, y_tr)
        row, _ = result_row('feature_budget', tfidf, 'svm', params,
                            model, X_tr_t, y_tr, X_val_t, y_val)
        save_result(row)
        print(f'{tfidf}: {X_tr_t.shape[1]} 维，macro-F1={row["f1"]:.5f}', flush=True)


def run_seed_check(args):
    """两个模型共享词干化 TF-IDF，在五个相同划分上配对比较。"""
    scores = {name: [] for name in ('svm', 'mlp')}
    for seed in (0, 1, 2, 3, 4):
        X_tr, X_val, y_tr, y_val = split_data(seed)
        _, X_tr_t, X_val_t = vectorize(X_tr, X_val, **tfidf_config('stem'))
        for name in ('svm', 'mlp'):
            params = selected_params(seed)[name]
            record_params = dict(params, split_seed=seed)
            model = make_model(name, params)
            model.fit(X_tr_t, y_tr)
            row, _ = result_row('seed_check', 'stem', name, record_params,
                                model, X_tr_t, y_tr, X_val_t, y_val)
            save_result(row)
            scores[name].append(row['f1'])
            print(f'seed={seed} {name}: macro-F1={row["f1"]:.5f}', flush=True)
    for name, values in scores.items():
        print(f'{name}: {np.mean(values):.5f} ± {np.std(values, ddof=1):.5f} '
              f'(n={len(values)})')


def run_loss_figure(args):
    """绘制 LR-SGD、线性 SVM-SGD、MLP 的训练损失曲线。"""

    output = Path(args.output).resolve() if args.output else ROOT / 'figures' / 'loss_comparison.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        save_loss_plot(load_loss_curves(output.with_name('loss_curves.csv')), output)
        return
    X_tr, X_val, y_tr, y_val = split_data()
    _, X_tr_t, X_val_t = vectorize(X_tr, X_val, **tfidf_config('base'))
    curves = []
    records = []
    for name, tfidf in [('lr_sgd', 'base'), ('svm_sgd', 'base'), ('mlp', 'base')]:
        if name in SGD_MODELS:
            params = selected_sgd_params()[name]
            model = SGDClassifier(**params)
            loss = train_sgd(model, X_tr_t, y_tr, collect_loss=True)
        else:
            params = selected_params()[name]
            model = BatchLossMLPClassifier(**params)
            model.fit(X_tr_t, y_tr)
            batch_size = min(200, len(y_tr)) if model.batch_size == 'auto' else model.batch_size
            batches_per_epoch = int(np.ceil(len(y_tr) / batch_size))
            loss = [((step + 1) / batches_per_epoch, value)
                    for step, value in enumerate(model.batch_losses_)]
        row, _ = result_row('loss_figure', tfidf, name, params, model,
                            X_tr_t, y_tr, X_val_t, y_val)
        save_result(row)
        curves.append((name, tfidf, params, loss, row['f1']))
        records.extend({'model': name, 'tfidf': tfidf, 'params': str(params),
                        'step': step, 'epoch': epoch, 'train_loss': value,
                        'loss_scope': 'mini_batch'}
                       for step, (epoch, value) in enumerate(loss, 1))
        print(f'{name} ({tfidf}): {len(loss)} batch losses / {loss[-1][0]:g} epochs, '
              f'验证 macro-F1={row["f1"]:.4f}', flush=True)

    save_loss_plot(curves, output)
    pd.DataFrame(records).to_csv(output.with_name('loss_curves.csv'), index=False)
    print(f'已保存 {output}、{output.with_suffix(".pdf")} 和 loss_curves.csv')
    print('三个模型的损失定义不同，应比较各自收敛趋势，不比较纵轴绝对值。')


def save_loss_plot(curves, output):
    """将三条损失曲线横向排版，分别使用各自的纵轴范围。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.65), constrained_layout=True)
    colors = ('#9792CB', '#E2C99E', '#852736')
    average_color = '#414B9E'
    titles = ('LR-SGD', 'Linear SVM-SGD', 'MLP')
    for ax, color, title, (name, tfidf, params, loss, f1) in zip(
            axes, colors, titles, curves):
        representation = 'Base TF-IDF' if tfidf == 'base' else tfidf
        optimizer = ('averaged' if params['average'] else 'unaveraged') if name in SGD_MODELS else 'Adam'
        detail = f'{representation} · {optimizer} · α={params["alpha"]:g}'
        positions, values = zip(*loss)
        positions = np.asarray(positions)
        values = np.asarray(values)
        window = int(round(1 / (positions[1] - positions[0])))
        average = np.convolve(values, np.ones(window) / window, mode='valid')
        average_positions = positions[window - 1:]
        ax.plot(positions, values, color=color, linewidth=0.55,
                label='Per-batch loss')
        ax.plot(average_positions, average, color=average_color, linewidth=1.0,
                label=f'Moving average (window {window})')
        ax.set_title(title, fontsize=12, fontweight='semibold', pad=20)
        ax.text(0.5, 1.02, detail, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=8.5, color='#555555')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Mini-batch training loss', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle='--', linewidth=0.65, alpha=0.38)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0.97, 0.94, f'Validation macro-F1: {f1:.4f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
        ax.legend(loc='upper right', bbox_to_anchor=(1, 0.87), fontsize=7.5,
                  framealpha=0.95, handlelength=2)
        if name == 'mlp':
            zoom = ax.inset_axes([0.44, 0.30, 0.52, 0.29])
            section = [(epoch, value) for epoch, value in loss if 5 <= epoch <= 15]
            zoom_x, zoom_y = zip(*section)
            zoom.plot(zoom_x, zoom_y, color=color, linewidth=0.65)
            mask = (average_positions >= 5) & (average_positions <= 15)
            zoom.plot(average_positions[mask], average[mask],
                      color=average_color, linewidth=0.9)
            zoom.set_title('Mini-batch detail: epochs 5–15', fontsize=8)
            zoom.tick_params(labelsize=7)
            zoom.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    fig.savefig(output, dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_loss_curves(path):
    history = pd.read_csv(path)
    results = pd.read_csv(ROOT / 'sweep_results.csv')
    curves = []
    for name in (*SGD_MODELS, 'mlp'):
        data = history[history['model'] == name].sort_values('step')
        if data.empty or data['tfidf'].nunique() != 1 or data['params'].nunique() != 1:
            raise ValueError(f'{name}: 曲线数据缺失或包含多个配置')
        tfidf, params = data.iloc[0][['tfidf', 'params']]
        key = record_key({'exp': 'loss_figure', 'tfidf': tfidf,
                          'model': name, 'params': params})
        matching = [row for row in results.to_dict('records')
                    if record_key(row) == key]
        if len(matching) != 1:
            raise ValueError(f'{name}: 找不到唯一的损失图评估结果')
        curves.append((name, tfidf, ast.literal_eval(params),
                       list(zip(data['epoch'], data['train_loss'])), matching[0]['f1']))
    return curves


def best_configurations(names):
    """按默认划分的验证 Macro-F1 选模，排除多 seed 检验和绘图重复评估。"""
    rows = pd.read_csv(ROOT / 'sweep_results.csv')
    candidates = rows[~rows['exp'].isin(['seed_check', 'loss_figure'])]
    best = {}
    for name in names:
        options = candidates[candidates['model'] == name].dropna(subset=['f1'])
        if options.empty:
            raise ValueError(f'{name}: 没有验证结果，请先运行实验')
        best[name] = options.sort_values('f1', ascending=False, kind='stable').iloc[0]
    return best


def run_predict_final(args):
    """使用各模型验证最优配置，在全部标注数据上重训并预测测试集。"""
    names = (*MODELS, *SGD_MODELS) if args.model == 'all' else (args.model,)
    selected = best_configurations(names)
    X_train, y_train, X_test = load_data()
    previous = pd.read_csv(PREDICTIONS_CSV) if PREDICTIONS_CSV.exists() else None
    if previous is not None and len(previous) != len(X_test):
        raise ValueError('predictions.csv 行数与测试集不一致')
    features = {}
    predictions = {}
    for name, row in selected.items():
        tfidf = row['tfidf']
        params = ast.literal_eval(row['params'])
        if tfidf not in features:
            _, X_train_t, X_test_t = vectorize(X_train, X_test, **tfidf_config(tfidf))
            features[tfidf] = (X_train_t, X_test_t)
        X_train_t, X_test_t = features[tfidf]
        print(f'{name}: TF-IDF={tfidf}, params={params}, '
              f'验证 Macro-F1={row["f1"]:.5f}', flush=True)
        model = SGDClassifier(**params) if name in SGD_MODELS else make_model(name, params)
        if name in SGD_MODELS:
            train_sgd(model, X_train_t, y_train, seed=params['random_state'])
        else:
            model.fit(X_train_t, y_train)
        values = model.predict(X_test_t)
        if len(values) != len(X_test) or not np.isin(values, np.unique(y_train)).all():
            raise ValueError(f'{name}: 预测行数或标签无效')
        predictions[name] = values
        if previous is not None and name in previous:
            changed = np.count_nonzero(previous[name].to_numpy() != values)
            print(f'{name}: 完成 {len(values)} 条预测，与现有文件相比变化 {changed} 条', flush=True)
    save_predictions(predictions)
    print(f'已保存 {PREDICTIONS_CSV}，更新列：{list(predictions)}')


def run_all(args):
    """依次运行参数扫描、特征对照、优化、稳定性、损失曲线与最终预测。"""
    run_basic(argparse.Namespace(stage='tune', model='all'))
    run_cleaning(argparse.Namespace(model='all'))
    run_sgd_sweep(argparse.Namespace(model='all', tfidf='all',
                                    top_k_base=None, focused=False))
    run_feature_budget(args)
    run_seed_check(args)
    run_loss_figure(argparse.Namespace(output=None, plot_only=False))
    run_predict_final(argparse.Namespace(model='all'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)
    sub.add_parser('all', help='运行完整实验流程并导出六模型测试集预测')
    cleaning = sub.add_parser('cleaning', help='四模型的文本处理对照')
    cleaning.add_argument('--model', choices=(*MODELS, 'all'), default='all')
    sgd = sub.add_parser('sgd-sweep', help='SGD alpha × average 与文本表示对照')
    sgd.add_argument('--model', choices=(*SGD_MODELS, 'all'), default='all')
    sgd.add_argument('--tfidf', choices=['all', 'base', 'ngram12', 'stop', 'nohdr_fixed', 'stem'],
                     default='all')
    sgd.add_argument('--top-k-base', type=int, choices=range(1, 5),
                     help='按原始文本验证 Macro-F1 选前 k 组参数；邮件头和词干化默认 3')
    sgd.add_argument('--focused', action='store_true',
                     help='使用每个模型两个 alpha 与参数平均开关的组合')
    sub.add_parser('feature-budget', help='20000 维下比较三种文本表示')
    sub.add_parser('seed-check', help='统一词干化输入的五个划分配对比较')
    plot = sub.add_parser('loss-figure', help='训练三个模型并绘制逐批损失')
    plot.add_argument('--output', help='PNG 路径，默认 figures/loss_comparison.png')
    plot.add_argument('--plot-only', action='store_true', help='读取已保存的逐批损失重绘图片')
    predict = sub.add_parser('predict-final', help='使用验证最优配置全量训练并预测测试集')
    predict.add_argument('--model', choices=(*MODELS, *SGD_MODELS, 'all'), default='all')
    args = parser.parse_args()
    {'all': run_all, 'cleaning': run_cleaning, 'sgd-sweep': run_sgd_sweep,
     'feature-budget': run_feature_budget, 'seed-check': run_seed_check,
     'loss-figure': run_loss_figure, 'predict-final': run_predict_final}[args.task](args)


if __name__ == '__main__':
    main()
