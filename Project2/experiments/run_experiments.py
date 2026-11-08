"""受控搜索实验，保存逐次结果、汇总表和出版质量图表。"""
import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

from search.alphabeta import TreeNode, alphabeta, minimax  # noqa: E402
from search.astar import manhattan, solve_maze, solve_weighted, zero_heuristic  # noqa: E402
from search.treasure_game import TreasureGame  # noqa: E402
from search.heuristics import landmark_heuristic  # noqa: E402
GROUPS = ('heuristic', 'tiebreak', 'ordering', 'game', 'scale')
FIELDS = ['experiment', 'scenario', 'seed', 'method', 'depth', 'cost', 'cost_ratio',
          'expanded', 'visited', 'evaluated', 'cache_hits', 'peak_frontier',
          'time_ms', 'prep_ms', 'best_target', 'best_value']


def timed(function, repeats=3):
    function()  # 预热不计时；每次调用都重新建立搜索状态和缓存。
    times = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        result = function()
        times.append((time.perf_counter_ns() - begin) / 1e6)
    return result, statistics.median(times)


def random_maze(n, density, rng):
    for _ in range(10000):
        grid = [['X' if rng.random() < density else '.' for _ in range(n)] for _ in range(n)]
        grid[0][0], grid[-1][-1] = 'S', 'E'
        if solve_maze((0, 0), (n-1, n-1), grid)[0]:
            return grid
    raise RuntimeError('无法生成连通地图')


def random_tree(branching, depth, rng):
    if depth == 0:
        return TreeNode(value=rng.randint(-99, 99))
    return TreeNode(children=[random_tree(branching, depth-1, rng) for _ in range(branching)])


def reorder_tree(node, is_max, best_first):
    """用精确子树值形成排序上/下界；预处理成本必须单独计入。"""
    if node.is_leaf:
        return node
    children = [reorder_tree(c, not is_max, best_first) for c in node.children]
    children.sort(key=lambda c: minimax(c, not is_max, {'visited': 0}), reverse=is_max == best_first)
    return TreeNode(children=children)


def scenario(seed, treasures, n=12):
    rng = random.Random(seed)
    for _ in range(1000):
        grid = [['X' if rng.random() < .15 else str(rng.randint(1, 9)) for _ in range(n)] for _ in range(n)]
        free = [(r,c) for r in range(n) for c in range(n) if grid[r][c] != 'X']
        rng.shuffle(free)
        names = ['A','B'] + [f'T{i+1}' for i in range(treasures)]
        points = dict(zip(names, free))
        values = {t: rng.randint(5, 15) for t in names[2:]}
        for name, (r,c) in points.items():
            grid[r][c] = name
        game = TreasureGame(grid, values, points)
        if all(d is not None for row in game.dist.values() for d in row.values()):
            return game
    raise RuntimeError('无法生成连通宝藏地图')


def run_heuristic(args):
    variants = [('Dijkstra', zero_heuristic, 1), ('Manhattan', manhattan, 1),
                ('A* admissible', None, 1), ('WA* 1.5', None, 1.5),
                ('WA* 2', None, 2), ('WA* 3', None, 3), ('ALT 3 landmarks', None, 1)]
    rows = []
    for i in range(args.maps):
        seed = args.seed + i
        rng = random.Random(seed)
        grid = [[rng.randint(1,9) for _ in range(20)] for _ in range(20)]
        grid[0][0], grid[-1][-1] = 'S','E'
        optimum = solve_weighted((0,0), (19,19), grid, zero_heuristic)[0]
        for label, heuristic, weight in variants:
            prep = 0.0
            if label.startswith('ALT'):
                heuristic, prep = timed(lambda: landmark_heuristic(grid, (19,19), [(0,0),(0,19),(19,0)]), args.repeats)
            (cost, _, stats), ms = timed(lambda: solve_weighted((0,0), (19,19), grid, heuristic, weight), args.repeats)
            if label in ('Dijkstra','A* admissible','ALT 3 landmarks'):
                assert cost == optimum
            rows.append(dict(experiment='heuristic', scenario='20x20 weights 1-9', seed=seed,
                             method=label, cost=cost, cost_ratio=cost/optimum, expanded=stats['expanded'],
                             peak_frontier=stats['peak_frontier'], time_ms=ms, prep_ms=prep))
    return rows


def run_tiebreak(args):
    rows = []
    for n in (15,25):
        for density in (.1,.2,.3):
            for i in range(args.maps):
                seed = args.seed+i
                grid = random_maze(n, density, random.Random(seed))
                optimum = solve_maze((0,0),(n-1,n-1),grid,zero_heuristic)[1]['cost']
                for label in ('fifo','deep'):
                    (_, stats), ms = timed(lambda: solve_maze((0,0),(n-1,n-1),grid,tiebreak=label), args.repeats)
                    assert stats['cost'] == optimum
                    rows.append(dict(experiment='tiebreak',scenario=f'{n}x{n} p={density:.1f}',seed=seed,
                                     method=label,cost=stats['cost'],expanded=stats['expanded'],
                                     peak_frontier=stats['peak_frontier'],time_ms=ms))
    return rows


def run_ordering(args):
    rows = []
    for b,d in ((2,6),(3,4),(3,6),(4,4),(4,5)):
        for i in range(args.trees):
            seed=args.seed+i
            root=random_tree(b,d,random.Random(seed))
            def search(tree, method):
                stats={'visited':0,'pruned':0}
                value=minimax(tree,True,stats) if method=='Minimax' else alphabeta(tree,True,stats=stats)
                return value,stats
            optimum=minimax(root,True,{'visited':0})
            for label in ('Minimax','AB natural','AB oracle best','AB oracle worst'):
                prep=0
                if 'oracle' in label:
                    tree,prep=timed(lambda: reorder_tree(root,True,label.endswith('best')),args.repeats)
                else:
                    tree=root
                (value,stats),ms=timed(lambda:search(tree,label),args.repeats)
                assert value==optimum
                rows.append(dict(experiment='ordering',scenario=f'b={b} d={d}',seed=seed,
                                 method=label,depth=d,visited=stats['visited'],best_value=value,time_ms=ms,prep_ms=prep))
    return rows


def game_rows(game, group, name, seed, depths, repeats):
    rows=[]
    for depth in depths:
        optimum=game.best_first_move(depth,False)['best_value']
        for label in ('Minimax','AB name','AB gain','Memoized'):
            def search():
                return game.memoized_first_move(depth) if label=='Memoized' else game.best_first_move(
                    depth,label!='Minimax','gain' if label=='AB gain' else 'name')
            result,ms=timed(search,repeats)
            assert result['best_value']==optimum
            # 验证实际选中的行动，不把剪枝回传的界误当作精确的首步估值。
            root=(game.key_points['A'],game.key_points['B'],tuple(sorted(game.values)),0,0)
            child=dict(game.successors(root,True))[result['best_target']]
            assert game.minimax(child,depth-1,False,{'visited':0})==optimum
            rows.append(dict(experiment=group,scenario=name,seed=seed,method=label,depth=depth,
                             visited=result['visited'],evaluated=result.get('evaluated',result['visited']),
                             cache_hits=result.get('cache_hits',0),time_ms=ms,
                             best_target=result['best_target'],best_value=result['best_value']))
    return rows


def run_game(args):
    game = TreasureGame()
    full_states, future_states = set(), set()
    visits = 0

    def probe(state, depth, is_max):
        nonlocal visits
        visits += 1
        full_states.add((state, depth, is_max))
        future_states.add((state[0], state[1], state[2], depth, is_max))
        if depth == 0 or not state[2]:
            return
        for _, child in game.successors(state, is_max):
            probe(child, depth - 1, not is_max)

    root = (game.key_points['A'], game.key_points['B'], tuple(sorted(game.values)), 0, 0)
    probe(root, 4, True)
    reuse = {'depth': 4, 'visits': visits, 'full_state_unique': len(full_states),
             'future_state_unique': len(future_states),
             'full_state_duplicates': visits - len(full_states),
             'future_state_duplicates': visits - len(future_states)}
    (RESULTS / 'state_reuse.json').write_text(json.dumps(reuse, indent=2))
    return game_rows(game, 'game', '7x7 / 4 treasures', args.seed, range(1, 5), args.repeats)


def run_scale(args):
    rows=[]
    for count in (6,8):
        for i in range(args.scenarios):
            seed=args.seed+i
            rows.extend(game_rows(scenario(seed,count),'scale',f'12x12 / {count} treasures',seed,
                                  (2,4,count),args.repeats))
            print(f'  scale: {count} treasures, seed={seed}',flush=True)
    return rows


def summary(rows):
    groups={}
    for row in rows:
        key=tuple(row.get(k,'') for k in ('experiment','scenario','depth','method'))
        groups.setdefault(key,[]).append(row)
    output=[]
    for key,items in groups.items():
        row=dict(zip(('experiment','scenario','depth','method'),key));row['n']=len(items)
        for metric in ('cost_ratio','expanded','visited','evaluated','cache_hits','peak_frontier','time_ms','prep_ms'):
            vals=[float(r[metric]) for r in items if r.get(metric,'')!='']
            if vals:
                row[metric+'_mean']=statistics.mean(vals)
                row[metric+'_std']=statistics.stdev(vals) if len(vals)>1 else 0
        output.append(row)
    return output


def save(rows,args):
    with (RESULTS / 'experiment_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    data=summary(rows)
    (RESULTS / 'experiment_summary.json').write_text(json.dumps(data,indent=2))
    import matplotlib,numpy
    metadata={'python':platform.python_version(),'matplotlib':matplotlib.__version__,
              'numpy':numpy.__version__,'platform':platform.platform(),'configuration':vars(args),
              'timing':'one warm-up, median of repeats; map generation excluded; oracle sorting measured separately',
              'node_count':'includes root and terminal; memoized evaluated excludes cache hits'}
    (RESULTS / 'experiment_metadata.json').write_text(json.dumps(metadata,indent=2))
    lines=['# 搜索实验结果','', '原始逐次记录见experiment_results.csv；每个时间是预热后重复计时的中位数。表中±为跨地图/树的样本标准差，不是计时误差。',
           '节点统计包含根和终止节点；Memoized的evaluated不含缓存命中，visited包含命中。Oracle排序用真实子树值，只作排序质量参照，排序时间另列。','']
    descriptions = {
        'heuristic': f'{args.maps}张20×20随机权值图（1–9），终点进入免费；三地标为起点、右上角、左下角；Weighted A*使用修正下界。',
        'tiebreak': f'边长15/25、障碍密度0.1/0.2/0.3，每组{args.maps}张起终点连通地图；同图配对比较，仅改变同f下的优先策略。',
        'ordering': f'分支/深度=(2,6)、(3,4)、(3,6)、(4,4)、(4,5)，各{args.trees}棵随机树，叶值−99至99；Oracle排序使用完整子树真值。',
        'game': '题目给定7×7地图、4宝藏，H=1–4。各首步共享根α；双方收益排序均为自身收益降序。状态复用计数见state_reuse.json。',
        'scale': f'12×12地图，障碍密度0.15，权值1–9，6/8宝藏价值5–15；各{args.scenarios}张关键点互通地图，H=2、4及完整深度。',
    }
    for group in GROUPS:
        if not any(row['experiment'] == group for row in data):
            continue
        lines+=['## '+group, '', descriptions[group] + f' 基础种子{args.seed}，逐实例递增；每次预热1次、重复{args.repeats}次取中位数。', '','| 场景 | 深度 | 方法 | 样本数 | 扩展/访问节点 | 实际求值 | 代价/最优 | 时间ms | 预处理ms |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for row in data:
            if row['experiment']!=group:continue
            def fmt(metric):
                return f"{row[metric+'_mean']:.3f}±{row[metric+'_std']:.3f}" if metric+'_mean' in row else '—'
            nodes=fmt('expanded') if 'expanded_mean' in row else fmt('visited')
            lines.append(f"| {row['scenario']} | {row['depth']} | {row['method']} | {row['n']} | {nodes} | {fmt('evaluated')} | {fmt('cost_ratio')} | {fmt('time_ms')} | {fmt('prep_ms')} |")
        lines.append('')
    content='\n'.join(lines)+'\n'
    (RESULTS / 'results.md').write_text(content)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',choices=GROUPS,default=list(GROUPS))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--maps',type=int,default=100)
    parser.add_argument('--trees',type=int,default=30)
    parser.add_argument('--scenarios',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if min(args.maps,args.trees,args.scenarios,args.repeats)<1:parser.error('样本数和重复次数必须大于0')
    if args.plot_only:
        with (RESULTS / 'experiment_results.csv').open() as f:rows=list(csv.DictReader(f))
    else:
        rows=[]
        for group in args.experiments:
            print('Running '+group,flush=True)
            rows.extend(globals()['run_'+group](args))
        save(rows,args)
    from visualization import render_all
    render_all(rows)
    print(f'完成：{len(rows)}条记录；图表见 {ROOT / "figures"}',flush=True)


if __name__=='__main__':main()
