"""
Fast heuristic methods

Simple strategies that run really fast but might not be optimal:
- Greedy: go towards most likely shot
- Expected Position: go towards average position
- MinMax: minimize worst case
- Weighted Direction: probability-weighted direction

All of these are O(n) where n is number of outcomes, so super fast
"""

import numpy as np
from typing import Tuple, List
from problem import (ProblemInstance, DefenderState, Outcome, KinematicModel,
                     CostFunction)

# could make this True for debugging
VERBOSE = False


class GreedyHeuristic:
    """
    Greedy: just go towards the most likely outcome
    Simple and fast
    """

    def __init__(self, problem: ProblemInstance):
        self.prob = problem

    def solve(self) -> Tuple[Tuple[float, float], str]:
        """
        Returns (ux, uy) and a string explaining what we did
        """
        # find outcome with highest probability
        max_p = 0.0
        best_outcome = None

        # print(f"Greedy: looking through {len(self.prob.outcomes)} outcomes")
        for out in self.prob.outcomes:
            # print(f"  region {out.region}: p={out.probability:.3f}")
            if out.probability > max_p:
                max_p = out.probability
                best_outcome = out

        # get center of that region (just use midpoint)
        r = best_outcome.region
        x_min, x_max, y_min, y_max = self.prob.get_region_bounds(r)
        target_x = (x_min + x_max) / 2.0
        target_y = (y_min + y_max) / 2.0
        # print(f"Target center: ({target_x:.2f}, {target_y:.2f})")

        # direction to target
        dx = target_x - self.prob.initial_state.x
        dy = target_y - self.prob.initial_state.y
        dist = np.sqrt(dx**2 + dy**2)
        # print(f"Distance to target: {dist:.2f}")

        if dist < 0.000001:  # already there (unlikely but check anyway)
            return (0.0, 0.0), "already at target"

        # accelerate max in that direction (simple!)
        ux = (dx / dist) * self.prob.a_max
        uy = (dy / dist) * self.prob.a_max

        msg = f"Greedy: region {r} (p={max_p:.2f}), acc=({ux:.2f},{uy:.2f})"

        return (ux, uy), msg


class ExpectedPositionHeuristic:
    """
    Expected position strategy: compute probability-weighted centroid of all
    possible landing regions and accelerate towards it
    """

    def __init__(self, problem: ProblemInstance):
        self.problem = problem

    def solve(self) -> Tuple[Tuple[float, float], str]:
        """
        Returns:
            ((ux, uy), explanation_string)
        """
        # Compute expected landing position
        expected_x = 0.0
        expected_y = 0.0

        for outcome in self.problem.outcomes:
            region = outcome.region
            x_min, x_max, y_min, y_max = self.problem.get_region_bounds(region)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            expected_x += outcome.probability * center_x
            expected_y += outcome.probability * center_y

        # Compute direction to expected position
        dx = expected_x - self.problem.initial_state.x
        dy = expected_y - self.problem.initial_state.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 1e-6:
            return (0.0, 0.0), "Already at expected position"

        # Scale to max acceleration
        ux = (dx / distance) * self.problem.a_max
        uy = (dy / distance) * self.problem.a_max

        explanation = (f"Expected Position: Centroid at ({expected_x:.1f}, {expected_y:.1f}), "
                      f"accel=({ux:.2f}, {uy:.2f})")

        return (ux, uy), explanation


class MinMaxHeuristic:
    """
    Min-Max strategy: choose acceleration that minimizes the worst-case
    straight-line distance to any outcome
    """

    def __init__(self, problem: ProblemInstance, n_samples: int = 16):
        self.problem = problem
        self.n_samples = n_samples

    def solve(self) -> Tuple[Tuple[float, float], str]:
        """
        Sample acceleration directions and choose one with best worst-case
        """
        best_accel = (0.0, 0.0)
        best_worst_case = float('inf')

        # Sample acceleration directions
        for i in range(self.n_samples):
            angle = 2 * np.pi * i / self.n_samples
            ux = self.problem.a_max * np.cos(angle)
            uy = self.problem.a_max * np.sin(angle)

            # Compute state after acceleration window
            state_after = KinematicModel.apply_acceleration_window(
                self.problem.initial_state, ux, uy,
                self.problem.H, self.problem.dt
            )

            # Find worst-case distance to any region
            worst_distance = 0.0
            for outcome in self.problem.outcomes:
                region = outcome.region
                x_min, x_max, y_min, y_max = self.problem.get_region_bounds(region)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2

                distance = np.sqrt((state_after.x - center_x)**2 +
                                 (state_after.y - center_y)**2)
                worst_distance = max(worst_distance, distance)

            if worst_distance < best_worst_case:
                best_worst_case = worst_distance
                best_accel = (ux, uy)

        explanation = (f"MinMax: Best worst-case distance = {best_worst_case:.2f}, "
                      f"accel={best_accel}")

        return best_accel, explanation


class WeightedDirectionHeuristic:
    """
    Weighted direction: compute acceleration as probability-weighted sum of
    directions to each outcome
    """

    def __init__(self, problem: ProblemInstance):
        self.problem = problem

    def solve(self) -> Tuple[Tuple[float, float], str]:
        """
        Returns:
            ((ux, uy), explanation_string)
        """
        # Compute weighted direction vector
        weighted_ux = 0.0
        weighted_uy = 0.0

        for outcome in self.problem.outcomes:
            region = outcome.region
            x_min, x_max, y_min, y_max = self.problem.get_region_bounds(region)
            target_x = (x_min + x_max) / 2
            target_y = (y_min + y_max) / 2

            # Direction to this outcome
            dx = target_x - self.problem.initial_state.x
            dy = target_y - self.problem.initial_state.y
            distance = np.sqrt(dx**2 + dy**2)

            if distance > 1e-6:
                # Normalize and weight by probability
                weighted_ux += outcome.probability * (dx / distance)
                weighted_uy += outcome.probability * (dy / distance)

        # Normalize weighted direction
        magnitude = np.sqrt(weighted_ux**2 + weighted_uy**2)
        if magnitude < 1e-6:
            return (0.0, 0.0), "Weighted direction is zero"

        ux = (weighted_ux / magnitude) * self.problem.a_max
        uy = (weighted_uy / magnitude) * self.problem.a_max

        explanation = (f"Weighted Direction: accel=({ux:.2f}, {uy:.2f})")

        return (ux, uy), explanation


class AdaptiveHeuristic:
    """
    Adaptive strategy: choose heuristic based on problem characteristics
    """

    def __init__(self, problem: ProblemInstance):
        self.problem = problem

    def solve(self) -> Tuple[Tuple[float, float], str]:
        """
        Adaptively select best heuristic based on problem structure
        """
        # If one outcome has very high probability, use greedy
        max_prob = max(o.probability for o in self.problem.outcomes)
        if max_prob > 0.7:
            heuristic = GreedyHeuristic(self.problem)
            result, msg = heuristic.solve()
            return result, f"Adaptive (Greedy): {msg}"

        # If outcomes are spread out, use expected position
        positions = []
        for outcome in self.problem.outcomes:
            region = outcome.region
            x_min, x_max, y_min, y_max = self.problem.get_region_bounds(region)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            positions.append((center_x, center_y))

        # Compute variance in positions
        mean_x = np.mean([p[0] for p in positions])
        mean_y = np.mean([p[1] for p in positions])
        variance = np.mean([(p[0] - mean_x)**2 + (p[1] - mean_y)**2
                           for p in positions])

        if variance > 50:  # High variance: outcomes spread out
            heuristic = WeightedDirectionHeuristic(self.problem)
            result, msg = heuristic.solve()
            return result, f"Adaptive (Weighted): {msg}"
        else:
            heuristic = ExpectedPositionHeuristic(self.problem)
            result, msg = heuristic.solve()
            return result, f"Adaptive (Expected): {msg}"


if __name__ == "__main__":
    from problem import DefenderState, Outcome, ShotType, CourtPosition, MovementDirection

    print("Heuristic Algorithms Test")
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

    # Test all heuristics
    heuristics = [
        ("Greedy", GreedyHeuristic(problem)),
        ("Expected Position", ExpectedPositionHeuristic(problem)),
        ("MinMax", MinMaxHeuristic(problem)),
        ("Weighted Direction", WeightedDirectionHeuristic(problem)),
        ("Adaptive", AdaptiveHeuristic(problem))
    ]

    for name, heuristic in heuristics:
        accel, explanation = heuristic.solve()
        print(f"\n{name}:")
        print(f"  Acceleration: ({accel[0]:.2f}, {accel[1]:.2f})")
        print(f"  {explanation}")
