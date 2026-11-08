"""主入口：运行四道题目的测试用例并输出结果与性能指标。

运行方式：python main.py（在 Project2 目录下）
所有预期输出均来自《实验二要求》文档；结果确定性，可直接复现。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from search.alphabeta import alphabeta, build_sample_tree, minimax  # noqa: E402
from search.api import (best_minimax_score, find_shortest_path, min_terrain_cost,  # noqa: E402
                        treasure_first_move)
from search.astar import solve_maze, solve_weighted, zero_heuristic  # noqa: E402
from search.treasure_game import GRID, POINT_NAMES, TreasureGame  # noqa: E402


def check(name: str, got, expected) -> bool:
    ok = got == expected
    print(f"  [{ 'PASS' if ok else 'FAIL' }] {name}: got={got}, expected={expected}")
    return ok


def problem1() -> bool:
    print("=" * 60)
    print("题目一：迷宫寻路（A*，曼哈顿距离）")
    start, goal = (0, 0), (3, 4)
    grid = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    path, stats = solve_maze(start, goal, grid)
    print(f"  最短路径: {path}")
    print(f"  坐标数: {len(path)}（步数 {len(path) - 1}），扩展节点数: {stats['expanded']}，生成节点数: {stats['generated']}")

    # 对照实验：h = 0（Dijkstra）vs 曼哈顿距离
    path0, stats0 = solve_maze(start, goal, grid, heuristic=zero_heuristic)
    print(f"  对照 h=0: 扩展节点数 {stats0['expanded']}（曼哈顿: {stats['expanded']}）")

    ok = len(path) == 8 and path[0] == start and path[-1] == goal
    ok &= len(path0) == 8  # h=0 同样保证最优
    print(f"  [{ 'PASS' if ok else 'FAIL' }] 路径含 8 个坐标、7 步（与预期输出等长，允许不同走法）")
    # 题目要求的输入输出格式：只返回坐标列表；无解返回空列表
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    ok &= check("接口输出（与题目样例逐项一致）", find_shortest_path(start, goal, grid), expected)
    blocked = [["S", "X", "E"]]
    ok &= check("无解时返回空列表", find_shortest_path((0, 0), (0, 2), blocked), [])
    return ok


def problem2() -> bool:
    print("=" * 60)
    print("题目二：加权地形（A*，修正曼哈顿下界）")
    start, goal = (0, 0), (4, 4)
    grid = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    cost, path, stats = solve_weighted(start, goal, grid)
    print(f"  最小总代价: {cost}")
    print(f"  路径: {path}")
    print(f"  扩展节点数: {stats['expanded']}，生成节点数: {stats['generated']}")

    cost0, _, stats0 = solve_weighted(start, goal, grid, heuristic=zero_heuristic)
    print(f"  对照 h=0: 代价 {cost0}，扩展节点数 {stats0['expanded']}（修正下界: {stats['expanded']}）")
    ok = check("最小总代价", cost, 7) & check("h=0 对照代价", cost0, 7)
    answer = min_terrain_cost(start, goal, grid)
    ok &= check("接口输出（int类型）", (answer, type(answer).__name__), (7, "int"))
    ok &= check("无解时返回-1", min_terrain_cost((0, 0), (0, 2), [["S", "X", "E"]]), -1)
    return ok


def problem3() -> bool:
    print("=" * 60)
    print("题目三：极大极小博弈树（Minimax / Alpha-Beta）")
    root = build_sample_tree()

    stats_mm = {"visited": 0}
    value_mm = minimax(root, True, stats_mm)
    stats_ab = {"visited": 0, "pruned": 0}
    value_ab = alphabeta(root, True, stats=stats_ab)

    print(f"  Minimax:    最佳得分 {value_mm}，访问节点数 {stats_mm['visited']}")
    print(f"  Alpha-Beta: 最佳得分 {value_ab}，访问节点数 {stats_ab['visited']}，剪枝跳过 {stats_ab['pruned']} 个节点")
    ok = check("Minimax 最佳得分", value_mm, 5) & check("Alpha-Beta 最佳得分", value_ab, 5)
    # 题目给出的JSON输入样例（tree_structure形式），以及直接传入TreeNode根节点
    leaf = lambda v: {"value": v}
    max_ = lambda *kids: {"is_maximizing_player": True, "children": list(kids)}
    min_ = lambda *kids: {"is_maximizing_player": False, "children": list(kids)}
    sample = {"tree_structure": max_(
        min_(max_(leaf(3), leaf(5)), max_(leaf(6), leaf(9))),
        min_(max_(leaf(6), leaf(9)), max_(leaf(1), leaf(2))))}
    ok &= check("JSON样例输入 → 最佳得分", best_minimax_score(sample), 5)
    ok &= check("TreeNode根节点输入 → 最佳得分", best_minimax_score(root), 5)
    return ok


def problem4() -> bool:
    print("=" * 60)
    print("题目四：竞争环境下的宝藏分配（A* + Minimax / Alpha-Beta）")
    game = TreasureGame(GRID)

    # 任务 1：6x6 距离矩阵
    print("  距离矩阵（A* 最小移动代价）:")
    header = "        " + "".join(f"{name:>6}" for name in POINT_NAMES)
    print(header)
    for a in POINT_NAMES:
        row = "".join(f"{game.dist[a][b]:>6g}" for b in POINT_NAMES)
        print(f"    {a:>3} {row}")

    ok = True
    for h, expected in ((2, {"best_target": "T2", "best_value": -1}),
                        (4, {"best_target": "T3", "best_value": 3})):
        for use_ab, label in ((False, "Minimax"), (True, "Alpha-Beta")):
            t0 = time.perf_counter()
            result = game.best_first_move(depth=h, use_alphabeta=use_ab)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  H={h} {label:<10}: best_target={result['best_target']}, "
                  f"best_value={result['best_value']}，访问节点数 {result['visited']}，耗时 {elapsed:.3f} ms")
            ok &= check(f"H={h} {label} 最优首步", result["best_target"], expected["best_target"])
            ok &= check(f"H={h} {label} 最优收益差", result["best_value"], expected["best_value"])
    for h, expected in ((2, {"best_target": "T2", "best_value": -1}),
                        (4, {"best_target": "T3", "best_value": 3})):
        ok &= check(f"接口输出 H={h}（与测试用例格式一致）", treasure_first_move(h), expected)
    return ok


def main() -> None:
    results = [problem1(), problem2(), problem3(), problem4()]
    print("=" * 60)
    print(f"汇总: {sum(results)}/{len(results)} 道题目通过全部检查")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
