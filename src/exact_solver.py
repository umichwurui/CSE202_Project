"""
Exact solver (but not really exact because of discretization...)

Tries different discrete accelerations and uses A* to find paths
Exponential time - gets slow quickly but useful as baseline

TODO: might be better to use dynamic programming instead of A*?
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from problem import (ProblemInstance, DefenderState, Outcome, KinematicModel,
                     CostFunction, ShotType)
import heapq
from dataclasses import dataclass


@dataclass
class SearchNode:
    """
    Node for A* search
    Stores state, cost so far, and parent for backtracking
    """
    state: DefenderState
    cost: float  # how much we spent to get here
    parent: Optional['SearchNode'] = None

    def __lt__(self, other):
        # for heapq to work
        return self.cost < other.cost


class ExactSolver:
    """
    'Exact' solver - tries all discretized accelerations
    Not truly exact due to discretization but close enough
    """

    def __init__(self, problem: ProblemInstance, n_directions: int = 16,
                 n_magnitudes: int = 5):
        """
        problem: the problem to solve
        n_directions: how many angles to try (e.g. 16 = every 22.5 degrees)
        n_magnitudes: how many different acceleration magnitudes
        """
        self.problem = problem
        self.n_dir = n_directions
        self.n_mag = n_magnitudes

        # precompute all the accelerations we'll try
        self.accel_set = self._generate_acceleration_set()

        # if DEBUG:
        #     print(f"Exact solver: {len(self.accel_set)} accelerations to try")

    def _generate_acceleration_set(self) -> List[Tuple[float, float]]:
        """
        Generate discrete set of accelerations to try
        Returns list of (ux, uy) tuples
        """
        # start with zero acceleration (do nothing option)
        accelerations = [(0.0, 0.0)]
        # print(f"Generating {self.n_mag} magnitudes x {self.n_dir} directions")

        # for each magnitude level
        for mag_idx in range(1, self.n_mag + 1):
            magnitude = (mag_idx / self.n_mag) * self.problem.a_max
            # print(f"  magnitude {mag_idx}: {magnitude:.2f}")

            # try different directions (evenly spaced angles)
            for dir_idx in range(self.n_dir):
                angle = 2 * np.pi * dir_idx / self.n_dir  # radians
                ux = magnitude * np.cos(angle)
                uy = magnitude * np.sin(angle)
                accelerations.append((ux, uy))

        # print(f"Total accelerations: {len(accelerations)}")
        return accelerations

    def compute_min_cost_to_region(self, state: DefenderState,
                                   region: int, deadline: float,
                                   max_steps: int = 100) -> Tuple[float, Optional[List]]:
        """
        Compute minimum cost trajectory from state to region within deadline

        Uses A* search with kinematic constraints

        Returns:
            (min_cost, trajectory) or (inf, None) if unreachable
        """
        # Check if already in region
        if self.problem.is_in_region(state.x, state.y, region):
            return 0.0, [(state.x, state.y)]

        # Priority queue: (f_cost, step_count, node)
        x_min, x_max, y_min, y_max = self.problem.get_region_bounds(region)
        target_x = (x_min + x_max) / 2
        target_y = (y_min + y_max) / 2

        def heuristic(s: DefenderState) -> float:
            """Admissible heuristic: straight-line distance with min cost"""
            dx = target_x - s.x
            dy = target_y - s.y
            return min(self.problem.c_h, self.problem.c_v) * (abs(dx) + abs(dy))

        initial_node = SearchNode(state=state, cost=0.0)
        open_set = [(heuristic(state), 0, id(initial_node), initial_node)]
        closed = set()

        step_count = 0
        best_cost = float('inf')
        best_trajectory = None

        while open_set and step_count < max_steps:
            f_cost, steps, _, current = heapq.heappop(open_set)

            # Create state key for visited check
            state_key = (round(current.state.x, 1), round(current.state.y, 1),
                        round(current.state.t, 2))

            if state_key in closed:
                continue
            closed.add(state_key)

            # Check if reached target region
            if self.problem.is_in_region(current.state.x, current.state.y, region):
                if current.state.t <= deadline:
                    # Reconstruct trajectory
                    trajectory = []
                    node = current
                    while node is not None:
                        trajectory.append((node.state.x, node.state.y))
                        node = node.parent
                    trajectory.reverse()

                    if current.cost < best_cost:
                        best_cost = current.cost
                        best_trajectory = trajectory
                    continue

            # Check time deadline
            if current.state.t >= deadline:
                continue

            # Expand neighbors: try all accelerations
            for ax, ay in self.acceleration_set:
                # Check acceleration bound
                a_mag = np.sqrt(ax**2 + ay**2)
                if a_mag > self.problem.a_max + 1e-6:
                    continue

                # Apply acceleration for one step
                new_state = KinematicModel.step(current.state, ax, ay, self.problem.dt)

                # Check bounds
                if new_state.x < 0 or new_state.x >= self.problem.grid_size:
                    continue
                if new_state.y < 0 or new_state.y >= self.problem.grid_size:
                    continue

                # Compute step cost
                dx = new_state.x - current.state.x
                dy = new_state.y - current.state.y
                step_cost = CostFunction.step_cost(dx, dy,
                                                   self.problem.c_h, self.problem.c_v)

                new_cost = current.cost + step_cost
                new_node = SearchNode(state=new_state, cost=new_cost, parent=current)

                f = new_cost + heuristic(new_state)
                heapq.heappush(open_set, (f, steps + 1, id(new_node), new_node))

            step_count += 1

        if best_trajectory is not None:
            return best_cost, best_trajectory
        else:
            return float('inf'), None

    def evaluate_acceleration(self, ux: float, uy: float) -> float:
        """
        Evaluate expected cost of choosing acceleration (ux, uy)

        Returns expected cost over all outcomes
        """
        # Apply acceleration window
        state_after_window = KinematicModel.apply_acceleration_window(
            self.problem.initial_state, ux, uy,
            self.problem.H, self.problem.dt
        )

        # Compute cost during anticipation window
        window_trajectory = [self.problem.initial_state]
        current = self.problem.initial_state.copy()
        for _ in range(self.problem.H):
            current = KinematicModel.step(current, ux, uy, self.problem.dt)
            window_trajectory.append(current)

        window_positions = [(s.x, s.y) for s in window_trajectory]
        window_cost = CostFunction.trajectory_cost(window_positions,
                                                   self.problem.c_h, self.problem.c_v)

        # Compute expected cost over all outcomes
        expected_cost = 0.0
        total_unreachable_prob = 0.0

        for outcome in self.problem.outcomes:
            # Compute minimum cost from end of window to target region
            post_window_cost, _ = self.compute_min_cost_to_region(
                state_after_window,
                outcome.region,
                outcome.deadline
            )

            if post_window_cost == float('inf'):
                # Unreachable: assign large penalty
                total_unreachable_prob += outcome.probability
                post_window_cost = 1e6
            else:
                total_cost = window_cost + post_window_cost
                expected_cost += outcome.probability * total_cost

        # Add penalty for unreachable outcomes
        if total_unreachable_prob > 0:
            expected_cost += total_unreachable_prob * 1e6

        return expected_cost

    def solve(self, verbose: bool = True) -> Tuple[Tuple[float, float], float]:
        """
        Find optimal acceleration by exhaustive search

        Returns:
            ((ux*, uy*), min_expected_cost)
        """
        best_acceleration = None
        best_cost = float('inf')

        if verbose:
            print(f"Exact solver: searching over {len(self.acceleration_set)} accelerations")

        for idx, (ux, uy) in enumerate(self.acceleration_set):
            if verbose and idx % 10 == 0:
                print(f"  Progress: {idx}/{len(self.acceleration_set)}")

            cost = self.evaluate_acceleration(ux, uy)

            if cost < best_cost:
                best_cost = cost
                best_acceleration = (ux, uy)

        if verbose:
            print(f"Optimal acceleration: {best_acceleration}")
            print(f"Minimum expected cost: {best_cost:.2f}")

        return best_acceleration, best_cost


if __name__ == "__main__":
    from problem import DefenderState, Outcome, ShotType, CourtPosition, MovementDirection

    print("Exact Solver Test")
    print("=" * 60)

    # Create test problem
    initial_state = DefenderState(x=15.0, y=15.0, vx=0.0, vy=0.0, t=0.0)

    outcomes = [
        Outcome(region=0, shot_type=ShotType.NET, probability=0.5, deadline=1.5),
        Outcome(region=8, shot_type=ShotType.CLEAR, probability=0.5, deadline=2.0)
    ]

    problem = ProblemInstance(
        initial_state=initial_state,
        opponent_position=CourtPosition.FRONT,
        racquet_posture="overhead",
        movement_direction=MovementDirection.NEUTRAL,
        outcomes=outcomes
    )

    # Solve
    solver = ExactSolver(problem, n_directions=8, n_magnitudes=3)
    best_accel, best_cost = solver.solve(verbose=True)

    print(f"\nSolution: acceleration = {best_accel}, cost = {best_cost:.2f}")
