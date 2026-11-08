"""题目四：竞争环境下的宝藏分配（A* + Minimax / Alpha-Beta 综合题）。

流程：
1. 用 A* 计算 A、B、T1~T4 两两之间的最小移动代价，得到 6x6 距离矩阵；
2. 博弈状态记录 (A 位置, B 位置, 剩余宝藏, A 累计收益, B 累计收益, 当前玩家)；
3. 分别用 Minimax 与 Alpha-Beta 在指定深度 H 下求 A 的最优首步与最优收益差。

收益规则：Gain(p, Ti) = Vi - d(p, Ti)，U = Score_A - Score_B；
A 最大化 U，B 最小化 U；到达搜索深度或宝藏取完时以当前 U 作为评价值。
"""

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .astar import astar, manhattan
from .grid import entry_cost

Coord = Tuple[int, int]

# 题目给定地图：X 为障碍，数字为进入代价，A/B/T 所在格代价为 1
GRID: List[List[str]] = [
    ["A", "1", "1", "X", "2", "1", "T1"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T2", "1", "1", "1", "1"],
    ["X", "1", "X", "8", "X", "1", "X"],
    ["1", "1", "1", "1", "T3", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]

TREASURE_VALUES: Dict[str, int] = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}

# 所有关键节点：两名玩家出发点 + 四个宝藏
KEY_POINTS: Dict[str, Coord] = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
POINT_NAMES = ["A", "B", "T1", "T2", "T3", "T4"]


def _cell_cost(grid: List[List[str]], pos: Coord) -> float:
    return entry_cost(grid[pos[0]][pos[1]])  # A、B、T 所在格代价为 1


def astar_cost(grid: List[List[str]], start: Coord, goal: Coord) -> Optional[float]:
    """A* 计算两点间最小移动代价；不可达返回 None。"""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图必须是非空矩形")
    rows, cols = len(grid), len(grid[0])
    for r, c in (start, goal):
        if not (0 <= r < rows and 0 <= c < cols) or grid[r][c] == "X":
            raise ValueError("关键点必须位于可通行的地图范围内")
    costs = [_cell_cost(grid, (r, c)) for r in range(rows)
             for c in range(cols) if grid[r][c] != "X"]
    if any(cost < 0 or not math.isfinite(cost) for cost in costs):
        raise ValueError("地形代价必须为有限非负数")
    minimum = min(costs)

    def neighbors(pos: Coord):
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "X":
                yield ((nr, nc), _cell_cost(grid, (nr, nc)))

    path, stats = astar(start, goal, neighbors, lambda a, b: minimum * manhattan(a, b))
    return stats["cost"] if path else None


def build_distance_matrix(
    grid: List[List[str]],
    key_points: Dict[str, Coord] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """任务 1：计算关键节点两两之间的最小移动代价（默认 A、B、T1~T4 的 6x6 矩阵）。"""
    if key_points is None:
        key_points = KEY_POINTS
    names = list(key_points)
    dist = {name: {} for name in names}
    for a in names:
        for b in names:
            d = astar_cost(grid, key_points[a], key_points[b])
            dist[a][b] = d
    return dist


class TreasureGame:
    """博弈状态与搜索。

    状态五元组 (pos_a, pos_b, remaining, score_a, score_b)：
    双方位置、剩余宝藏集合、双方累计收益；当前玩家由 is_max_turn 表示。
    """

    def __init__(
        self,
        grid: List[List[str]] = GRID,
        values: Dict[str, int] = TREASURE_VALUES,
        key_points: Dict[str, Coord] = None,
    ):
        self.grid = grid
        self.values = dict(values)
        self.key_points = dict(KEY_POINTS if key_points is None else key_points)
        self.dist = build_distance_matrix(grid, self.key_points)

    def gain(self, pos: Coord, treasure: str) -> float:
        """Gain(p, Ti) = Vi - d(p, Ti)"""
        return self.values[treasure] - self.dist_at(pos, treasure)

    def dist_at(self, pos: Coord, treasure: str) -> Optional[float]:
        """从任意关键节点位置到宝藏的最小代价（查距离矩阵）。"""
        for name, coord in self.key_points.items():
            if coord == pos:
                return self.dist[name][treasure]
        raise ValueError(f"位置 {pos} 不是关键节点")

    def successors(self, state: Tuple, is_max_turn: bool, order: str = "name"):
        """当前玩家的所有合法行动：选取任一剩余宝藏，收益累加并更新位置。

        order="name"：按 T1->T4 的固定顺序（从左到右）；
        order="gain"：双方均按自己的即时收益降序（B 收益大使 U 更低），
        让更可能改变 alpha/beta 的走法先被评估，以提高剪枝效率。
        """
        pos_a, pos_b, remaining, score_a, score_b = state
        if order not in ("name", "gain"):
            raise ValueError("order 必须为 name 或 gain")
        mover_pos = pos_a if is_max_turn else pos_b
        treasures = [t for t in sorted(remaining) if self.dist_at(mover_pos, t) is not None]
        if order == "gain":
            treasures.sort(key=lambda t: self.gain(mover_pos, t), reverse=True)
        for t in treasures:
            g = self.gain(pos_a if is_max_turn else pos_b, t)
            new_remaining = tuple(x for x in remaining if x != t)
            if is_max_turn:
                yield t, (self.key_points[t], pos_b, new_remaining, score_a + g, score_b)
            else:
                yield t, (pos_a, self.key_points[t], new_remaining, score_a, score_b + g)

    @staticmethod
    def evaluate(state: Tuple) -> int:
        """U = Score_A - Score_B"""
        return state[3] - state[4]

    def minimax(self, state: Tuple, depth: int, is_max_turn: bool, stats: Dict,
                order: str = "name") -> int:
        stats["visited"] += 1
        if depth <= 0 or not state[2]:
            return self.evaluate(state)
        values = [
            self.minimax(nxt, depth - 1, not is_max_turn, stats, order)
            for _, nxt in self.successors(state, is_max_turn, order)
        ]
        return (max(values) if is_max_turn else min(values)) if values else self.evaluate(state)

    def alphabeta(
        self,
        state: Tuple,
        depth: int,
        is_max_turn: bool,
        alpha: float = -math.inf,
        beta: float = math.inf,
        stats: Optional[Dict] = None,
        order: str = "name",
    ) -> int:
        if stats is None:
            stats = {"visited": 0}
        stats["visited"] += 1
        if depth <= 0 or not state[2]:
            return self.evaluate(state)

        children = list(self.successors(state, is_max_turn, order))
        if not children:
            return self.evaluate(state)
        if is_max_turn:
            value = -math.inf
            for _, nxt in children:
                value = max(value, self.alphabeta(nxt, depth - 1, False, alpha, beta, stats, order))
                alpha = max(alpha, value)
                if value >= beta:
                    stats["cutoffs"] = stats.get("cutoffs", 0) + 1
                    break
            return value
        else:
            value = math.inf
            for _, nxt in children:
                value = min(value, self.alphabeta(nxt, depth - 1, True, alpha, beta, stats, order))
                beta = min(beta, value)
                if value <= alpha:
                    stats["cutoffs"] = stats.get("cutoffs", 0) + 1
                    break
            return value

    def best_first_move(self, depth: int, use_alphabeta: bool = True,
                        order: str = "name") -> Dict:
        """任务 3：在深度 H 下求 A 的最优首步与最优收益差。"""
        if depth < 1:
            raise ValueError("首步搜索深度至少为 1")
        state = (self.key_points["A"], self.key_points["B"], tuple(sorted(self.values)), 0, 0)
        stats = {"visited": 1, "cutoffs": 0}
        best_target, best_value = None, -math.inf
        for t, nxt in self.successors(state, True, order):
            if use_alphabeta:
                value = self.alphabeta(nxt, depth - 1, False, best_value, math.inf, stats, order)
            else:
                value = self.minimax(nxt, depth - 1, False, stats, order)
            if value > best_value:
                best_target, best_value = t, value
        if best_target is None:
            best_value = self.evaluate(state)
        return {"best_target": best_target, "best_value": best_value, **stats}

    def memoized_first_move(self, depth: int) -> Dict:
        """精确置换表：缓存未来收益差，历史累计收益不影响后续最优行动。"""
        if depth < 1:
            raise ValueError("首步搜索深度至少为 1")

        @lru_cache(maxsize=None)
        def future(a, b, remaining, h, is_max):
            if h <= 0 or not remaining:
                return 0
            pos = a if is_max else b
            values = []
            for t in remaining:
                distance = self.dist[pos][t]
                if distance is None:
                    continue
                gain = self.values[t] - distance
                rest = tuple(x for x in remaining if x != t)
                if is_max:
                    values.append(gain + future(t, b, rest, h - 1, False))
                else:
                    values.append(-gain + future(a, t, rest, h - 1, True))
            return (max(values) if is_max else min(values)) if values else 0

        remaining = tuple(sorted(self.values))
        target, value = None, -math.inf
        for t in remaining:
            if self.dist["A"][t] is None:
                continue
            candidate = self.values[t] - self.dist["A"][t] + future(
                t, "B", tuple(x for x in remaining if x != t), depth - 1, False)
            if candidate > value:
                target, value = t, candidate
        info = future.cache_info()
        return {"best_target": target, "best_value": value if target else 0,
                "visited": info.hits + info.misses + 1, "evaluated": info.misses + 1,
                "cache_hits": info.hits, "cache_size": info.currsize}

