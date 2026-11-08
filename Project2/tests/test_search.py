"""搜索算法的独立最优值对照与边界测试：python -m unittest discover -s Project2/tests。"""
import heapq
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from search.astar import astar, solve_maze, solve_weighted, zero_heuristic  # noqa: E402
from search.heuristics import landmark_heuristic  # noqa: E402
from search.alphabeta import TreeNode, alphabeta, build_sample_tree, minimax  # noqa: E402
from search.treasure_game import TreasureGame, astar_cost, build_distance_matrix  # noqa: E402
from search.api import best_minimax_score, find_shortest_path, min_terrain_cost, treasure_first_move  # noqa: E402


def reference_cost(grid, start, goal):
    queue, distance = [(0, start)], {start: 0}
    while queue:
        cost, pos = heapq.heappop(queue)
        if cost != distance[pos]:
            continue
        if pos == goal:
            return cost
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nxt = pos[0] + dr, pos[1] + dc
            r, c = nxt
            if not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] == 'X':
                continue
            step = 0 if nxt == goal else (grid[r][c] if isinstance(grid[r][c], (int, float)) else 1)
            candidate = cost + step
            if candidate < distance.get(nxt, math.inf):
                distance[nxt] = candidate
                heapq.heappush(queue, (candidate, nxt))
    return -1


class SearchTests(unittest.TestCase):
    def test_weighted_against_independent_reference(self):
        for seed in range(80):
            rng = random.Random(seed)
            grid = [[rng.choice([0, 0.5, 1, 3, 9, 'X']) for _ in range(7)] for _ in range(7)]
            grid[0][0], grid[-1][-1] = 'S', 'E'
            expected = reference_cost(grid, (0, 0), (6, 6))
            for tie in ('fifo', 'deep'):
                cost, path, _ = solve_weighted((0, 0), (6, 6), grid, tiebreak=tie)
                self.assertEqual(cost, expected)
                if path:
                    self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(path,path[1:])))

    def test_landmark_admissibility_and_consistency(self):
        for seed in range(12):
            rng = random.Random(seed)
            grid = [[rng.randint(1, 9) for _ in range(6)] for _ in range(6)]
            grid[0][0], grid[-1][-1] = 'S', 'E'
            goal = (5, 5)
            h = landmark_heuristic(grid, goal, [(0, 0), (0, 5), (5, 0)])
            for r in range(6):
                for c in range(6):
                    self.assertLessEqual(h((r, c), goal), reference_cost(grid, (r, c), goal))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nxt = r+dr, c+dc
                        if 0 <= nxt[0] < 6 and 0 <= nxt[1] < 6:
                            cell = grid[nxt[0]][nxt[1]]
                            step = 0 if nxt == goal else (cell if isinstance(cell, int) else 1)
                            self.assertLessEqual(h((r, c), goal), step+h(nxt, goal))
            self.assertEqual(solve_weighted((0, 0), goal, grid, h)[0], reference_cost(grid, (0, 0), goal))

    def test_reopening_with_inconsistent_admissible_heuristic(self):
        edges = {'S': [('A', 3), ('B', 1)], 'B': [('A', 1)], 'A': [('G', 2)], 'G': []}
        h = {'S': 4, 'B': 3, 'A': 0, 'G': 0}
        path, stats = astar('S', 'G', edges.__getitem__, lambda a, b: h[a])
        self.assertEqual(path, ['S', 'B', 'A', 'G'])
        self.assertEqual(stats['cost'], 4)
        self.assertEqual(stats['reopened'], 1)

    def test_edge_cases(self):
        path, stats = solve_maze((0, 0), (0, 0), [['S']])
        self.assertEqual((path, stats['cost'], stats['expanded']), ([(0, 0)], 0, 1))
        self.assertEqual(solve_maze((0, 0), (0, 2), [['S', 'X', 'E']])[0], [])
        with self.assertRaises(ValueError):
            solve_maze((0, 0), (0, 0), [])
        with self.assertRaises(ValueError):
            solve_weighted((0, 0), (0, 2), [['S', -1, 'E']])
        with self.assertRaises(ValueError):
            astar(0, 1, lambda _: [(1, -1)], zero_heuristic)

        for invalid in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                astar_cost([["A", "B"], [invalid, "1"]], (0, 0), (0, 0))

    def test_tree_pruning_equivalence(self):
        def tree(rng, depth):
            return TreeNode(value=rng.randint(-10, 10)) if depth == 0 else TreeNode(children=[tree(rng, depth-1) for _ in range(rng.randint(1, 4))])
        for seed in range(60):
            root = tree(random.Random(seed), 4)
            mm, ab = {'visited': 0}, {'visited': 0, 'pruned': 0}
            self.assertEqual(minimax(root, True, mm), alphabeta(root, True, stats=ab))
            self.assertLessEqual(ab['visited'], mm['visited'])

        # 调用方只提供visited或空统计字典时，剪枝统计仍能正常累计。
        for stats in ({"visited": 0}, {}):
            self.assertEqual(alphabeta(build_sample_tree(), True, stats=stats), 5)
            self.assertEqual(stats["visited"], 14)
            self.assertEqual(stats["pruned"], 1)

    def test_directed_distance(self):
        distance = build_distance_matrix([['2', '5']], {'A': (0, 0), 'B': (0, 1)})
        self.assertEqual(distance['A']['B'], 5)
        self.assertEqual(distance['B']['A'], 2)
        fractional = build_distance_matrix([[0.5, 0.25]], {'A': (0, 0), 'B': (0, 1)})
        self.assertEqual(fractional['A']['B'], .25)
        self.assertEqual(fractional['B']['A'], .5)

    def test_game_algorithms_and_selected_moves(self):
        for seed in range(16):
            rng = random.Random(seed)
            game = TreasureGame(values={f'T{i}': rng.randint(1, 15) for i in range(1, 5)})
            for depth in range(1, 5):
                reference = game.best_first_move(depth, False)
                for result in [game.best_first_move(depth, True, order) for order in ('name', 'gain')] + [game.memoized_first_move(depth)]:
                    self.assertEqual(result['best_value'], reference['best_value'])
                    root = (game.key_points['A'], game.key_points['B'], tuple(sorted(game.values)), 0, 0)
                    chosen = dict(game.successors(root, True))[result['best_target']]
                    self.assertEqual(game.minimax(chosen, depth-1, False, {'visited': 0}), reference['best_value'])

    def test_unreachable_game_and_depth(self):
        game = TreasureGame([['A','X','T1'], ['B','X','1']], {'T1': 3}, {'A': (0,0),'B': (1,0),'T1': (0,2)})
        self.assertEqual(game.best_first_move(3)['best_value'], 0)
        self.assertIsNone(game.memoized_first_move(3)['best_target'])
        with self.assertRaises(ValueError):
            game.best_first_move(0)


class SpecInterfaceTests(unittest.TestCase):
    """题目规定的输入输出格式：返回类型、无解约定、JSON输入。"""

    def test_output_formats(self):
        grid = [["S", ".", "X"], [".", "X", "E"], [".", ".", "."]]
        path = find_shortest_path((0, 0), (1, 2), grid)
        self.assertIsInstance(path, list)
        self.assertTrue(all(isinstance(p, tuple) for p in path))
        self.assertEqual(find_shortest_path((0, 0), (0, 2), [["S", "X", "E"]]), [])
        cost = min_terrain_cost((0, 0), (0, 2), [["S", 2, "E"]])
        self.assertEqual((cost, type(cost)), (2, int))
        self.assertEqual(min_terrain_cost((0, 0), (0, 2), [["S", "X", "E"]]), -1)
        self.assertEqual(treasure_first_move(2), {"best_target": "T2", "best_value": -1})
        self.assertEqual(treasure_first_move(4), {"best_target": "T3", "best_value": 3})

    def test_json_tree_input(self):
        leaf = lambda v: {"value": v}
        node = lambda m, *k: {"is_maximizing_player": m, "children": list(k)}
        spec = {"tree_structure": node(True,
                node(False, node(True, leaf(3), leaf(5)), node(True, leaf(6), leaf(9))),
                node(False, node(True, leaf(6), leaf(9)), node(True, leaf(1), leaf(2))))}
        self.assertEqual(best_minimax_score(spec), 5)
        self.assertEqual(best_minimax_score(build_sample_tree()), 5)


if __name__ == '__main__':
    unittest.main()
