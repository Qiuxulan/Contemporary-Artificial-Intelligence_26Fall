# 实验二：搜索算法

实现A*、Minimax与Alpha-Beta，完成迷宫寻路、加权地形、博弈树和竞争取宝；通过启发函数、并列策略、行动排序、状态压缩与多规模场景分析搜索效率及决策差异。

## 目录结构

```text
Project2/
├── main.py                  # 入口：运行四道题的样例并检查预期结果
├── requirements.txt
├── src/search/              # 核心算法（只依赖Python标准库）
│   ├── api.py               # 按题目规定的输入输出格式封装的接口
│   ├── grid.py              # 网格代价的公共约定
│   ├── astar.py             # 通用A*、迷宫(Task 1)、加权地形(Task 2)
│   ├── heuristics.py        # 正反向地标距离与可采纳ALT启发
│   ├── alphabeta.py         # 博弈树、Minimax、Alpha-Beta(Task 3)
│   └── treasure_game.py     # 距离矩阵、竞争取宝、缓存(Task 4)
├── tests/test_search.py     # 10项性质、边界与输入输出格式测试
├── experiments/             # 对照实验与绘图（依赖NumPy/Matplotlib）
│   ├── run_experiments.py   # 五组实验、计时、写出results/与figures/
│   ├── visualization.py     # 读取实验记录绘图
│   └── export_report.py     # 将report.md导出为提交用PDF
├── results/                 # 实验输出（CSV、JSON、results.md）
├── figures/                 # 报告引用的7张400dpi PNG图片
└── report/                  # report.md正文与最终PDF
```

核心搜索不依赖图形库或报告工具；`main.py`负责给定样例，`src/search/`负责输入、搜索与返回结果，`experiments/`负责实验与统计、绘图。

## 环境与安装

算法和基础自检只使用Python标准库；完整实验及绘图使用NumPy、Matplotlib，报告PDF导出使用WeasyPrint和markdown-it-py。本次完整运行环境为Python 3.14.7、NumPy 2.5.3、Matplotlib 3.11.2、WeasyPrint 70.0、markdown-it-py 4.2.0，具体环境与实验配置保存于`results/experiment_metadata.json`。

```bash
cd Project2
python -m pip install -r requirements.txt
```

PDF导出固定使用Noto Sans CJK SC显示中文与英文正文，代码使用DejaVu Sans Mono；这些字体与安装新字体前PDF的实际用字一致，避免新增系统字体改变回退结果。算法、实验和绘图均可在无显示器环境运行。计时受硬件与系统负载影响；同环境下的固定输入、解、节点计数和种子配置可复现。

## 快速运行

均在`Project2`目录下执行：

```bash
python main.py                                  # 四道题样例，全部通过时退出码为0
python -m unittest discover -s tests -v         # 10项性质、边界与输入输出格式测试
python experiments/run_experiments.py           # 完整受控实验（覆盖results/与figures/）
```

`main.py`失败时退出码为1；实验中任何最优值或所选行动与完整搜索不一致都会触发断言。报告正文位于`report/report.md`，按题目顺序组织，使用`python experiments/export_report.py`导出为`report/report_preview.pdf`，PDF即为最终报告。

## 任务与代码对应

| 任务 | 算法与状态 | 实现入口 | 实验记录与报告 |
| --- | --- | --- | --- |
| Task 1：迷宫 | 四邻域坐标、单位步长、曼哈顿下界 | `astar.py`的`solve_maze` | `tiebreak`组；报告第3、7.1节 |
| Task 2：加权地形 | 进入格代价、免费终点、修正/地标启发 | `solve_weighted`、`heuristics.py` | `heuristic`组；报告第4、7.2节 |
| Task 3：博弈树 | MAX/MIN回传与α/β窗口 | `alphabeta.py` | `ordering`组；报告第5节 |
| Task 4：竞争取宝 | 关键点距离、双方位置与收益、轮次与剩余宝藏 | `treasure_game.py` | `game`、`scale`组；报告第6、7.3–7.4节 |

题目规定格式的接口在`src/search/api.py`（在`src`加入`sys.path`后调用）：

```python
from search.api import (find_shortest_path, min_terrain_cost,
                        best_minimax_score, treasure_first_move)

path = find_shortest_path(start, goal, grid)     # List[Tuple[int,int]]，无解返回[]
cost = min_terrain_cost(start, goal, grid)       # int，无解返回-1
best = best_minimax_score(tree)                  # TreeNode根节点，或题目的JSON字典
move = treasure_first_move(depth=4)              # {"best_target": "T3", "best_value": 3}
```

需要节点数、堆峰值等统计量时，使用底层接口（返回`(结果, stats)`）：

```python
from search.astar import solve_maze, solve_weighted
from search.alphabeta import build_tree, alphabeta
from search.treasure_game import TreasureGame

path, stats = solve_maze(start, goal, grid)
cost, path, stats = solve_weighted(start, goal, weighted_grid)
value = alphabeta(build_tree(tree_spec), True)

# points包含A、B及各宝藏坐标；values保存宝藏面值。
game = TreasureGame(treasure_grid, values, points)
result = game.best_first_move(depth=4, use_alphabeta=True, order="gain")
# result包含best_target、best_value、visited和cutoffs。
```

输出约定（底层接口）：迷宫返回路径及统计；加权地形额外返回最小代价，无解代价为−1；博弈树返回根值；取宝返回最优首步与收益差。具体代价、终止与并列规则见下文。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `results/experiment_results.csv` | 每张地图/每棵树/每种方法的2708条原始运行记录 |
| `results/experiment_summary.json` | 各配置的样本数、均值及样本标准差 |
| `results/experiment_metadata.json` | 依赖版本、运行参数与计时口径 |
| `results/results.md` | 按实验分组的配置与结果表 |
| `results/state_reuse.json` | 四宝藏完整状态与未来状态的重复计数 |
| `figures/` | 报告引用的7张400dpi PNG图片 |

## 实验设计与命令

默认`python experiments/run_experiments.py`执行以下五组实验。相同实例上配对比较，仅更改当前研究的因素。

| 实验参数 | 默认输入与对照 | 记录数 |
| --- | --- | --- |
| `heuristic` | 100张20×20权值图；Dijkstra、原始曼哈顿、修正A*、w=1.5/2/3、三地标ALT | 700 |
| `tiebreak` | 15/25边长×0.1/0.2/0.3障碍密度×100张连通图；FIFO/deep | 1200 |
| `ordering` | 5组分支/深度×30棵随机树；Minimax、AB自然序、Oracle最好/最坏序 | 600 |
| `game` | 四宝藏样例、H=1–4；Minimax、AB名称序、AB自身收益降序、精确缓存 | 16 |
| `scale` | 6/8宝藏，各8张12×12连通地图；H=2/4/完整深度，四种方法 | 192 |

基础种子42，地图逐实例用42–141、树用42–71、大场景用42–49。不同地图尺寸/密度或树结构可复用同一组种子；比较发生在同一配置、同一实例的不同方法之间。迷宫生成时拒绝起终点不连通的布局；宝藏地图拒绝关键点不互通的布局。

```bash
# 选择要研究的实验组
python experiments/run_experiments.py --experiments heuristic tiebreak
python experiments/run_experiments.py --experiments game scale

# 改变实例数、种子和重复计时次数
python experiments/run_experiments.py --seed 42 --maps 100 --trees 30 --scenarios 8 --repeats 3

# 快速检查完整流程（会将结果文件写为该次小规模运行的数据）
python experiments/run_experiments.py --maps 3 --trees 2 --scenarios 1 --repeats 1

# 使用已经保存的CSV重新绘图，不重新搜索或计时
python experiments/run_experiments.py --plot-only

# 导出最终报告（默认输出report/report_preview.pdf）
python experiments/export_report.py
# 调整样式（字体、字号、行距等，可用的键见export_report.py开头的STYLE）
python experiments/export_report.py --set h2_size=13pt --set line_height=1.4
```

每次实际实验会覆盖结果文件，文件只包含此次选择的实验组。默认全量命令重新生成全部2708条记录。`--plot-only`保留原始数据，只刷新当前CSV中可用实验组的图；完整报告需要全量实验结果。修改默认样本数或代码后，应重新核对报告中的配置与数字。

## 算法约定

### A*与启发函数

- 邻居仅上下左右。题一`X`为障碍，其他格步长为1；路径包含起点和终点，所以坐标数比步数多1。
- 题二按进入格代价累积，进入终点为0，起点不计初始费用。默认下界为`c_min * max(Manhattan - 1, 0)`，`c_min`取非终点可通行格的最小进入代价。
- `weight=1`是标准A*；`weight>1`为Weighted A*，可以返回次优解。CSV记录`cost_ratio=cost/Dijkstra_cost`，值越接近1越好。
- `tiebreak="fifo"`在同f下按生成顺序；`"deep"`优先g更大，进一步并列时仍按生成顺序。两者不改变f的第一优先级。
- 发现较小g时重新开放已扩展节点，堆中保留入堆时g并跳过过期条目。无解返回空路径，题二代价返回−1；起终点重合代价0。
- `expanded`统计实际弹出并处理的次数，含终点；重新开放可使同一位置计多次。`expanded_set`是唯一位置集合，`expanded_order`保留扩展顺序，`generated`不含初始入堆。
- `peak_frontier`是堆的物理最大条目数，含待丢弃的过期项，可反映队列存储需求，不等同于所有搜索数据的总内存。
- ALT选三角点为地标，以正反向距离的三角不等式构造下界。构建时间单列；本实现的免费终点与预处理目标绑定。

### 博弈与缓存

- MAX代表A，MIN代表B；U为A累计收益减B累计收益。每轮选一个可达的剩余宝藏，收益为面值减最小移动代价；负收益行动仍然合法。
- 移动途中经过其他宝藏不自动获取，只有指定目标被领取。宝藏取完、达到深度或当前玩家无合法目标时评价当前状态；最后一种情况按本实现约定结束分支。
- `best_first_move(H)`的H是总行动数，至少1。每个候选首步之间共享根层α。
- `order="gain"`让双方都按自身即时收益降序；B收益越大，U越小。排序是访问次序启发，最终行动仍由完整深度的Minimax/Alpha-Beta评价决定。
- 多个行动价值相同时返回访问顺序中最先出现者。实验同时验证根值相同与选中的行动确实达到完整Minimax最优值。
- `memoized_first_move`缓存从当前位置开始的未来收益差，键是双方位置、剩余宝藏、剩余深度、行动方；每次搜索建立新表。历史分数与未来可加地分离，因此可从键中省略。
- 精确缓存独立于Alpha-Beta；没有把截断搜索返回的上下界当成精确值缓存。
- 关键点距离逐方向计算。只有端点进入代价相同等特定条件下才可认为对称；不可达距离为`None`。

## 统计与计时

每个方法先预热一次，再独立搜索`--repeats`次取计时中位数；每次缓存都重新创建。跨地图/树汇总为均值±样本标准差。固定单样例n=1时，汇总标准差为0，不能据此推断方法对随机实例的稳定性。

计时使用`perf_counter_ns`，包含每次算法初始化与搜索，排除地图/树生成、断言核验、绘图与写文件。宝藏策略计时排除共用的距离矩阵构建。ALT构造及Oracle排序时间记为`prep_ms`；Oracle的总时间是搜索与排序之和。Oracle用真实子树值形成排序参照，不代表无成本的实用排序算法。

`visited`包含根、终止节点和缓存命中调用；`evaluated`对缓存方法只统计实际计算，对其他方法等于访问数。统计的都是工作次数而非对象地址数。题三的`pruned`另指直接跳过的子树根数，不能将其当作被跳过子树内的全部节点数；总体节省用Minimax与Alpha-Beta的`visited`差计算。不同算法每节点操作不同，报告同时分析节点数和真实时间。

CSV中未适用字段留空。`best_target`与`best_value`记录博弈决策；`expanded`、`cost`、`cost_ratio`记录路径搜索；`time_ms`是单实例重复计时的中位数，`prep_ms`是该实例预处理的中位数。

## 测试范围与预期结果

`test_search.py`验证80张含零权/浮点权/障碍的随机加权图、12张图的逐状态地标性质、60棵非均匀随机树、16组宝藏价值×4深度的策略值与选中行动，以及不一致启发重新开放、无解、起终点重合、非法输入、非对称距离、不可达目标。

| 基础任务 | 预期 |
| --- | --- |
| 一 | 最短路径8个坐标、7步 |
| 二 | 最小总代价7 |
| 三 | 根节点最优值5 |
| 四H=2 | T2，收益差−1 |
| 四H=4 | T3，收益差+3 |

## 图表索引

| 报告图号 | 文件 | 内容 |
| --- | --- | --- |
| 图1 | `fig1_maze.png` | Dijkstra、A* FIFO/deep的搜索区域与最短路径 |
| 图2 | `fig2_terrain_path_cost.png` | 加权地形、最优路径与累计代价 |
| 图3 | `fig7_tree.png` | 博弈树数值回传、剪枝界与被跳过的叶子 |
| 图4 | `report2_ordering_titled.png` | 访问顺序对节点量及含排序总时间的影响 |
| 图5 | `report3_game_no_depth_label.png` | 地图、距离矩阵与不同深度的首步估值 |
| 图6 | `fig4_tiebreak.png` | 六类地图上FIFO/deep的扩展量及配对削减 |
| 图7 | `report4_scaling.png` | 6/8宝藏及不同深度下的实际求值量 |
