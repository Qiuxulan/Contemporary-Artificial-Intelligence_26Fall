"""题目三：极大极小博弈树 —— Minimax 与 Alpha-Beta 剪枝。

非叶子节点通过 ``children`` 给出子节点（从左到右依次评估），
叶子节点通过 ``value`` 给出得分；根节点为 MAX 玩家。
同时统计访问节点数，供报告中比较剪枝效率与访问顺序的影响。
"""

import math
from typing import Dict, List, Optional


class TreeNode:
    def __init__(self, value: Optional[float] = None, children: Optional[List["TreeNode"]] = None):
        self.value = value  # 叶子节点的得分
        self.children = children if children is not None else []  # 子节点列表

    @property
    def is_leaf(self) -> bool:
        return not self.children


def minimax(node: TreeNode, is_maximizing: bool, stats: Dict) -> float:
    """标准 Minimax：MAX 取子节点最大值，MIN 取最小值。"""
    stats["visited"] = stats.get("visited", 0) + 1
    if node.is_leaf:
        return node.value
    values = [minimax(child, not is_maximizing, stats) for child in node.children]
    return max(values) if is_maximizing else min(values)


def alphabeta(
    node: TreeNode,
    is_maximizing: bool,
    alpha: float = -math.inf,
    beta: float = math.inf,
    stats: Optional[Dict] = None,
) -> float:
    """Alpha-Beta 剪枝。子节点按从左到右的顺序评估。

    MAX 层：当前值 >= beta 时停止扩展其余子节点（beta 剪枝）；
    MIN 层：当前值 <= alpha 时停止扩展其余子节点（alpha 剪枝）。
    visited 统计求值调用，pruned 统计直接跳过的子树根数（不是子树总节点数）。
    """
    if stats is None:
        stats = {"visited": 0, "pruned": 0}
    stats["visited"] = stats.get("visited", 0) + 1
    if node.is_leaf:
        return node.value

    if is_maximizing:
        value = -math.inf
        for index, child in enumerate(node.children):
            value = max(value, alphabeta(child, False, alpha, beta, stats))
            alpha = max(alpha, value)
            if value >= beta:
                stats["pruned"] = stats.get("pruned", 0) + len(node.children) - index - 1
                break
        return value
    else:
        value = math.inf
        for index, child in enumerate(node.children):
            value = min(value, alphabeta(child, True, alpha, beta, stats))
            beta = min(beta, value)
            if value <= alpha:
                stats["pruned"] = stats.get("pruned", 0) + len(node.children) - index - 1
                break
        return value


def build_tree(spec: Dict) -> TreeNode:
    """从 JSON 字典构建 TreeNode（对应文档题目三的输入样例格式）。

    支持题目样例的完整形式：最外层{"tree_structure": {...}}，节点里的
    "is_maximizing_player"字段只描述层级，递归时按根为MAX、逐层交替处理。
    """
    spec = spec.get("tree_structure", spec)
    if "value" in spec:
        return TreeNode(value=spec["value"])
    return TreeNode(children=[build_tree(child) for child in spec["children"]])


def build_sample_tree() -> TreeNode:
    """文档给出的测试用例树（根为 MAX，预期最佳得分 5）。"""
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])
