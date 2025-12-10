"""
Approximation Algorithms for the project

This file has the grid-based and sampling approximation methods.
Trying to get good approximation ratios while being fast enough for real-time use.

TODO: maybe add adaptive grid spacing?
"""

import numpy as np
from typing import Tuple, List
from problem import (ProblemInstance, DefenderState, KinematicModel,
                     CostFunction)

# debugging flag
DEBUG = False


class GridApproximation:
    """
    Grid-based approximation - divide acceleration space into a grid and check each point
    Should give us (1+ε)-approximation according to theory
    """

    def __init__(self, problem: ProblemInstance, grid_resolution: int = 10):
        """
        problem: the problem we're solving
        grid_resolution: how many points per side of grid (higher = better but slower)
        """
        self.problem = problem
        self.res = grid_resolution  # shorter name

        # compute theoretical error bound
        self.epsilon = self._compute_epsilon()

        # if DEBUG:
        #     print(f"Grid init: res={self.res}, epsilon={self.epsilon}")

    def _compute_epsilon(self) -> float:
        """
        Figure out the approximation error bound
        Based on how coarse the grid is
        """
        # how far apart grid points are
        spacing = 2 * self.problem.a_max / (self.res - 1)
        # print(f"spacing = {spacing}")  # debug

        # worst case position error from grid discretization
        # this is based on kinematic formula: s = 0.5*a*t^2
        pos_error = 0.5 * spacing * (self.problem.H * self.problem.dt)**2
        # print(f"max position error: {pos_error}")

        # translate to cost error (use bigger cost coeff to be safe)
        cost_error = 2 * self.problem.c_v * pos_error * self.problem.H

        # relative error
        # TODO: this bound might be too conservative?
        min_cost = self.problem.c_h * self.problem.delta
        eps = cost_error / min_cost
        # print(f"epsilon = {eps:.4f}")

        return eps

    def _generate_grid_points(self) -> List[Tuple[float, float]]:
        """
        Make all the grid points we need to check
        Returns list of (ux, uy) tuples
        """
        points = []
        a_max = self.problem.a_max

        # just loop through i,j and map to acceleration space
        for i in range(self.res):
            for j in range(self.res):
                # map [0, res-1] to [-a_max, a_max]
                ux = -a_max + (2 * a_max * i) / (self.res - 1)
                uy = -a_max + (2 * a_max * j) / (self.res - 1)

                # only keep if inside the circle (acceleration bound)
                if np.sqrt(ux**2 + uy**2) <= a_max:
                    points.append((ux, uy))

        # print(f"Generated {len(points)} grid points")  # debug
        return points

    def _estimate_cost(self, ux: float, uy: float) -> float:
        """
        Estimate cost for a given acceleration
        Uses straight-line distance as a heuristic (faster than exact A*)
        """
        # where we'll be after the anticipation window
        state_after = KinematicModel.apply_acceleration_window(
            self.problem.initial_state, ux, uy,
            self.problem.H, self.problem.dt
        )
        # print(f"After window: pos=({state_after.x:.2f}, {state_after.y:.2f}), vel=({state_after.vx:.2f}, {state_after.vy:.2f})")

        # cost during the window (actual movement cost)
        traj = [self.problem.initial_state]
        curr = self.problem.initial_state.copy()
        for step in range(self.problem.H):
            curr = KinematicModel.step(curr, ux, uy, self.problem.dt)
            traj.append(curr)

        positions = [(s.x, s.y) for s in traj]
        window_cost = CostFunction.trajectory_cost(positions,
                                                   self.problem.c_h, self.problem.c_v)
        # print(f"Window cost: {window_cost:.2f}")

        # estimate cost after window for each possible outcome
        total_expected_cost = 0.0

        for outcome in self.problem.outcomes:
            # target region center
            r = outcome.region
            x_min, x_max, y_min, y_max = self.problem.get_region_bounds(r)
            target_x = (x_min + x_max) / 2.0
            target_y = (y_min + y_max) / 2.0

            # manhattan distance as lower bound
            # (this is optimistic - actual path will cost more)
            dx = abs(target_x - state_after.x)
            dy = abs(target_y - state_after.y)

            # use minimum cost coefficient (optimistic estimate)
            min_coeff = min(self.problem.c_h, self.problem.c_v)
            cost_to_target = min_coeff * (dx + dy)

            # check if we have time
            time_left = outcome.deadline - state_after.t
            if time_left > 0:
                total_expected_cost += outcome.probability * cost_to_target
                # print(f"  region {r}: p={outcome.probability:.2f}, dist={dx+dy:.1f}, cost={cost_to_target:.1f}")
            else:
                # can't reach in time - penalize heavily
                # print(f"  region {r}: INFEASIBLE (time_left={time_left:.2f})")
                total_expected_cost += outcome.probability * 1000000

        # print(f"Total cost for ({ux:.2f}, {uy:.2f}): {window_cost + total_expected_cost:.2f}")
        return window_cost + total_expected_cost

    def solve(self, verbose: bool = True) -> Tuple[Tuple[float, float], float, dict]:
        """
        Main solve function - checks all grid points and picks best
        Returns: (best_acceleration, cost, info)
        """
        # generate all the grid points we're going to check
        points = self._generate_grid_points()
        # print(f"Total points to check: {len(points)}")

        if verbose:
            print(f"Grid Approximation: checking {len(points)} points")
            print(f"Theoretical ratio: 1 + {self.epsilon:.4f}")

        best_acc = None
        best_c = float('inf')

        # just try all points and keep track of best
        # (brute force but works for small grids)
        for i, (ux, uy) in enumerate(points):
            if verbose and i % 50 == 0:
                print(f"  {i}/{len(points)} done")

            c = self._estimate_cost(ux, uy)
            # print(f"Point {i}: ({ux:.2f}, {uy:.2f}) -> cost {c:.2f}")

            if c < best_c:
                best_c = c
                best_acc = (ux, uy)
                # if DEBUG:
                #     print(f"    new best: ({ux:.2f}, {uy:.2f}) cost={c:.2f}")

        # return some info about the search
        info = {
            'epsilon': self.epsilon,
            'grid_size': len(points),
            'grid_resolution': self.res,
            'approximation_ratio': 1 + self.epsilon
        }

        if verbose:
            print(f"Best accel: {best_acc}")
            print(f"Cost: {best_c:.2f}")

        return best_acc, best_c, info


class SamplingApproximation:
    """
    Random sampling method
    Just pick random accelerations and try them
    """

    def __init__(self, problem: ProblemInstance, n_samples: int = 100):
        self.problem = problem
        self.n = n_samples  # shorter

    def solve(self, verbose: bool = True) -> Tuple[Tuple[float, float], float, dict]:
        """
        Solve by random sampling
        """
        if verbose:
            print(f"Sampling: trying {self.n} random points")

        best_acc = None
        best_c = float('inf')

        # always include (0, 0) as one option (do nothing)
        samples = [(0.0, 0.0)]

        # generate random samples
        # need to sample uniformly in a disk - that's why we use sqrt
        # (if you just use random() for radius, points cluster at center)
        for i in range(self.n - 1):
            r = self.problem.a_max * np.sqrt(np.random.random())  # radius
            theta = 2 * np.pi * np.random.random()  # angle
            ux = r * np.cos(theta)
            uy = r * np.sin(theta)
            # print(f"Sample {i}: ({ux:.2f}, {uy:.2f})")
            samples.append((ux, uy))

        # reuse grid approximation's cost function (lazy but works)
        # could write our own but this is faster
        grid_helper = GridApproximation(self.problem, grid_resolution=5)

        for idx, (ux, uy) in enumerate(samples):
            if verbose and idx % 20 == 0:
                print(f"  {idx}/{len(samples)} checked")

            c = grid_helper._estimate_cost(ux, uy)
            # print(f"Sample {idx}: cost={c:.2f}")

            if c < best_c:
                best_c = c
                best_acc = (ux, uy)
                # print(f"  -> new best!")

        info = {
            'n_samples': self.n,
            'method': 'monte_carlo'  # fancy name for random sampling
        }

        if verbose:
            print(f"Best: {best_acc}")
            print(f"Cost: {best_c:.2f}")

        return best_acc, best_c, info


if __name__ == "__main__":
    from problem import DefenderState, Outcome, ShotType, CourtPosition, MovementDirection

    print("Approximation Algorithms Test")
    print("=" * 60)

    # Create test problem
    initial_state = DefenderState(x=15.0, y=15.0, vx=0.0, vy=0.0, t=0.0)

    outcomes = [
        Outcome(region=0, shot_type=ShotType.NET, probability=0.3, deadline=1.5),
        Outcome(region=2, shot_type=ShotType.SMASH, probability=0.4, deadline=1.0),
        Outcome(region=6, shot_type=ShotType.DROP, probability=0.3, deadline=1.8)
    ]

    problem = ProblemInstance(
        initial_state=initial_state,
        opponent_position=CourtPosition.FRONT,
        racquet_posture="overhead",
        movement_direction=MovementDirection.NEUTRAL,
        outcomes=outcomes
    )

    print("\n1. Grid Approximation")
    print("-" * 60)
    grid_solver = GridApproximation(problem, grid_resolution=8)
    accel, cost, info = grid_solver.solve(verbose=True)
    print(f"Result: {accel}, cost={cost:.2f}")
    print(f"Approximation ratio: {info['approximation_ratio']:.4f}")

    print("\n2. Sampling Approximation")
    print("-" * 60)
    sampling_solver = SamplingApproximation(problem, n_samples=50)
    accel, cost, info = sampling_solver.solve(verbose=True)
    print(f"Result: {accel}, cost={cost:.2f}")
