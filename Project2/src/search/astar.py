"""A* 搜索算法：题目一（迷宫寻路）与题目二（加权地形）。

两种场景共用同一个 A* 骨架，通过邻居函数区分代价与障碍：
- 题目一：'X' 为障碍物，其余格子可通过，每步代价为 1；
- 题目二：数字格子的权值即进入代价，终点 'E' 进入代价为 0。

同时记录扩展节点数等统计信息，供实验报告中比较不同启发函数的效率。
"""

import heapq
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .grid import entry_cost

Coord = Tuple[int, int]

# 只允许上下左右四个方向移动
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def manhattan(a: Coord, b: Coord) -> int:
    """单位四邻域的曼哈顿距离。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zero_heuristic(a: Coord, b: Coord) -> int:
    """h(n) = 0，退化为 Dijkstra / 一致代价搜索，用于对照实验。"""
    return 0


def astar(
    start: Coord,
    goal: Coord,
    neighbors_fn: Callable[[Coord], Iterable[Tuple[Coord, float]]],
    heuristic: Callable[[Coord, Coord], float] = manhattan,
    weight: float = 1.0,
    tiebreak: str = "fifo",
) -> Tuple[List[Coord], Dict]:
    """通用 A* 搜索。

    Args:
        start: 起点坐标 (row, col)。
        goal: 终点坐标 (row, col)。
        neighbors_fn: 给定坐标，返回 [(邻居坐标, 进入该邻居的代价), ...]。
        heuristic: 启发函数 h(n)，w=1时需可采纳且h(goal)=0以保证最优解。
        weight: 启发函数权重 w，f = g + w*h；w > 1 即 Weighted A*（更快但可能次优）。
        tiebreak: f 值相同时的并列打破策略，
            "fifo" 按生成顺序，"deep" 优先扩展 g 更大（更深）的节点。

    Returns:
        (path, stats)：path 为起点到终点的坐标序列，无解时为空列表；
        stats 包含 expanded（扩展节点数）、generated（生成节点数）、cost（总代价）、
        expanded_set（所有被扩展节点的集合，用于可视化）。
    """
    if weight < 0 or not math.isfinite(weight):
        raise ValueError("weight 必须为有限非负数")
    if tiebreak not in ("fifo", "deep"):
        raise ValueError("tiebreak 必须为 fifo 或 deep")
    stats = {"expanded": 0, "generated": 0, "cost": math.inf,
             "reopened": 0, "peak_frontier": 1, "expanded_set": set(),
             "expanded_order": []}
    counter = 0
    # 保存入堆时的 g，跳过过期条目；计数器保证同优先级按生成顺序处理。
    open_heap = [(weight * heuristic(start, goal), 0, counter, 0.0, start)]
    g_score = {start: 0.0}
    came_from = {}
    closed = set()
    while open_heap:
        _, _, _, popped_g, current = heapq.heappop(open_heap)
        if popped_g != g_score[current] or current in closed:
            continue
        closed.add(current)
        stats["expanded"] += 1
        stats["expanded_set"].add(current)
        stats["expanded_order"].append(current)
        if current == goal:
            stats["cost"] = popped_g
            return _reconstruct(came_from, start, goal), stats
        for nxt, step_cost in neighbors_fn(current):
            if step_cost < 0 or not math.isfinite(step_cost):
                raise ValueError("A* 要求有限非负边权")
            tentative_g = popped_g + step_cost
            if tentative_g < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                if nxt in closed:
                    closed.remove(nxt)
                    stats["reopened"] += 1
                counter += 1
                tie = 0 if tiebreak == "fifo" else -tentative_g
                heapq.heappush(open_heap, (tentative_g + weight * heuristic(nxt, goal),
                                          tie, counter, tentative_g, nxt))
                stats["generated"] += 1
                stats["peak_frontier"] = max(stats["peak_frontier"], len(open_heap))
    return [], stats


def _reconstruct(came_from: Dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def _validate_grid(grid, start, goal):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图必须是非空矩形")
    for r, c in (start, goal):
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            raise ValueError("起点或终点越界")
        if grid[r][c] == "X":
            raise ValueError("起点或终点不能是障碍")


def solve_maze(
    start: Coord,
    goal: Coord,
    grid: List[List[str]],
    heuristic: Callable[[Coord, Coord], float] = manhattan,
    weight: float = 1.0,
    tiebreak: str = "fifo",
) -> Tuple[List[Coord], Dict]:
    """题目一：在 'X' 为障碍的网格中找最短路径（每步代价 1）。"""
    _validate_grid(grid, start, goal)
    rows, cols = len(grid), len(grid[0])

    def neighbors(pos: Coord) -> List[Tuple[Coord, float]]:
        r, c = pos
        result = []
        for dr, dc in _DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "X":
                result.append(((nr, nc), 1.0))
        return result

    return astar(start, goal, neighbors, heuristic, weight=weight, tiebreak=tiebreak)


def solve_weighted(
    start: Coord,
    goal: Coord,
    grid: List[List],
    heuristic: Optional[Callable[[Coord, Coord], float]] = None,
    weight: float = 1.0,
    tiebreak: str = "fifo",
) -> Tuple[float, List[Coord], Dict]:
    """题目二：加权地形上的最小移动代价。

    从 A 到相邻 B 的代价等于进入 B 的地形权值；终点 'E' 进入代价为 0，
    起点/特殊字符格按 1 处理。返回 (最小总代价, 路径, 统计信息)，无解时代价为 -1。
    """
    _validate_grid(grid, start, goal)
    rows, cols = len(grid), len(grid[0])

    def cell_cost(pos: Coord) -> float:
        if pos == goal:             # 题目规定：最后一步没有代价，将 E 视为 0
            return 0.0
        return entry_cost(grid[pos[0]][pos[1]])

    def neighbors(pos: Coord) -> List[Tuple[Coord, float]]:
        r, c = pos
        result = []
        for dr, dc in _DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "X":
                nxt = (nr, nc)
                result.append((nxt, cell_cost(nxt)))
        return result

    costs = [cell_cost((r, c)) for r in range(rows) for c in range(cols)
             if grid[r][c] != "X" and (r, c) != goal]
    if any(c < 0 or not math.isfinite(c) for c in costs):
        raise ValueError("地形代价必须为有限非负数")
    if heuristic is None:
        minimum = min(costs, default=0)
        heuristic = lambda a, b: minimum * max(manhattan(a, b) - 1, 0)
    path, stats = astar(start, goal, neighbors, heuristic, weight=weight, tiebreak=tiebreak)
    if not path:
        return -1, [], stats
    return stats["cost"], path, stats
