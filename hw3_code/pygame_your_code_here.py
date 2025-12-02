
# your_code_here.py
# Your job is to implement the rrt function

from __future__ import annotations
from typing import List, Optional, Tuple
import math
import numpy as np

Point = Tuple[float, float]

def _nearest_node_index(utils, nodes: List[Point], q: Point) -> int:
    """Return index of node in `nodes` closest to point q (linear scan)."""
    best_idx = 0
    best_dist = float("inf")
    for i, p in enumerate(nodes):
        d = utils.dist(p, q)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def rrt(ctx, max_iters: int, step: float) -> Tuple[bool, int]:
    """Single-tree RRT that returns (goal_found, num_samples)."""
    utils = ctx.utils
    viz = ctx.viz

    # the nodes of the graph should be added to this list. Each node is represented by a point (x,y)
    nodes: List[Point] = [ctx.env.start]
    parents: List[Optional[int]] = [None]

    goal_found = False
    num_samples = 0

    # small goal bias helps convergence without KD-Tree
    goal_bias = 0.05

    for _ in range(max_iters):
        num_samples += 1

        # 1) sample a random free configuration (with slight bias towards goal)
        q_rand = utils.sample_free(goal_bias=goal_bias)

        # 2) find nearest node in the tree
        nearest_idx = _nearest_node_index(utils, nodes, q_rand)
        q_nearest = nodes[nearest_idx]

        # 3) steer from nearest towards random sample by at most `step`
        q_new = utils.steer(q_nearest, q_rand, step)

        # 4) add if collision-free
        if utils.collision_free(q_nearest, q_new):
            nodes.append(q_new)
            parents.append(nearest_idx)
            viz.add_edge(q_nearest, q_new)

            # 5) check goal
            if utils.reached(q_new):
                goal_idx = len(nodes) - 1
                path = utils.reconstruct_path(nodes, parents, goal_idx)
                viz.draw_path(path)
                goal_found = True
                return goal_found, num_samples

    return goal_found, num_samples

