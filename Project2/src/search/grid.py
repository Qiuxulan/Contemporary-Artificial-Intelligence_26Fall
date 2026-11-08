"""网格地图的公共约定，三个模块共用，避免各处重复解析格子代价。"""


def entry_cost(cell, default: float = 1.0) -> float:
    """进入某个格子的地形代价。

    数字（或数字字符串）就是权值；S、A、B、T1 等特殊标记格按 default 处理。
    终点免费、障碍不可进入等规则由调用方根据题目自行处理。
    """
    if isinstance(cell, (int, float)):
        return float(cell)
    if isinstance(cell, str) and cell.isdigit():
        return float(cell)
    return default
