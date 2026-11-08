"""按题目要求的输入输出格式封装的接口（实验要求第3–15页）。

底层算法在 astar.py / alphabeta.py / treasure_game.py，这里只做格式适配：
    题一  find_shortest_path(start, goal, grid)   -> List[Tuple[int, int]]，无解返回 []
    题二  min_terrain_cost(start, goal, grid)     -> int，无解返回 -1
    题三  best_minimax_score(tree)                -> 根节点（MAX）的最佳得分
    题四  treasure_first_move(depth)              -> {"best_target": "T2", "best_value": -1}
"""
from typing import Dict, List, Tuple, Union

from .alphabeta import TreeNode, alphabeta, build_tree
from .astar import solve_maze, solve_weighted
from .treasure_game import GRID, TreasureGame

Coord = Tuple[int, int]


def _as_int(value):
    """代价在内部用float累加；结果是整数时按题目要求返回int。"""
    return int(value) if float(value).is_integer() else value


def find_shortest_path(start: Coord, goal: Coord, grid: List[List[str]]) -> List[Coord]:
    """题目一：按顺序返回起点到终点最短路径上的所有坐标；不可达返回空列表。"""
    path, _ = solve_maze(start, goal, grid)
    return path


def min_terrain_cost(start: Coord, goal: Coord, grid: List[List]) -> int:
    """题目二：起点到终点的最小总代价（进入终点代价为0）；不可达返回-1。"""
    cost, _, _ = solve_weighted(start, goal, grid)
    return _as_int(cost)


def best_minimax_score(tree: Union[TreeNode, Dict]):
    """题目三：返回根节点（MAX玩家）的最佳得分。

    tree可以是TreeNode根节点，也可以是题目给出的JSON字典
    （含"tree_structure"/"is_maximizing_player"/"children"/"value"）。
    """
    is_max = True
    if isinstance(tree, dict):
        spec = tree.get("tree_structure", tree)
        is_max = spec.get("is_maximizing_player", True)
        tree = build_tree(spec)
    return _as_int(alphabeta(tree, is_max))


def treasure_first_move(depth: int, use_alphabeta: bool = True) -> Dict:
    """题目四：深度H下A的最优首步与最优收益差，格式与测试用例一致。"""
    result = TreasureGame(GRID).best_first_move(depth, use_alphabeta)
    return {"best_target": result["best_target"], "best_value": _as_int(result["best_value"])}
