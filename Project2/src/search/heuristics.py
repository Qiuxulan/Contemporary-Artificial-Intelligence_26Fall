"""加权有向网格的地标启发：用三角不等式构造可采纳下界。"""
import heapq
import math
from .astar import manhattan
from .grid import entry_cost


def landmark_heuristic(grid, goal, landmarks):
    """返回启发函数；构建耗时应计入预处理而非隐藏在在线搜索开销之外。

    进入格子的权值构成有向代价，即使相邻关系无向，也需要正、反向距离。
    h(n)=max_L{d(L,G)-d(L,n), d(n,L)-d(G,L), 0}。
    """
    cells = {(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell != 'X'}
    if goal not in cells or not set(landmarks) <= cells:
        raise ValueError('终点与地标必须是可通行位置')

    def entry(pos):
        return 0.0 if pos == goal else entry_cost(grid[pos[0]][pos[1]])

    costs = {p: entry(p) for p in cells}
    if any(c < 0 or not math.isfinite(c) for c in costs.values()):
        raise ValueError('地形代价必须为有限非负数')

    def distances(source, reverse=False):
        result = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            g, pos = heapq.heappop(queue)
            if g != result[pos]:
                continue
            r, c = pos
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = r + dr, c + dc
                if nxt not in cells:
                    continue
                new = g + costs[pos if reverse else nxt]
                if new < result.get(nxt, math.inf):
                    result[nxt] = new
                    heapq.heappush(queue, (new, nxt))
        return result

    tables = [(distances(p), distances(p, True)) for p in dict.fromkeys(landmarks)]
    minimum = min((v for p, v in costs.items() if p != goal), default=0)

    def heuristic(pos, target):
        if target != goal:
            raise ValueError('地标预处理与搜索终点必须一致')
        lower = minimum * max(manhattan(pos, target) - 1, 0)
        for forward, backward in tables:
            if goal in forward and pos in forward:
                lower = max(lower, forward[goal] - forward[pos])
            if goal in backward and pos in backward:
                lower = max(lower, backward[pos] - backward[goal])
        return lower

    return heuristic
