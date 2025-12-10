"""
Anticipatory Defensive Movement Problem - Core Problem Definition

This module defines the badminton defensive movement optimization problem
where a defender must choose an initial acceleration to minimize expected
movement cost under uncertainty.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ShotType(Enum):
    """Types of badminton shots with different speeds"""
    NET = "net"
    DROP = "drop"
    LIFT = "lift"
    CLEAR = "clear"
    DRIVE = "drive"
    SMASH = "smash"


class CourtPosition(Enum):
    """Opponent court position"""
    FRONT = "front"
    MID = "mid"
    REAR = "rear"


class MovementDirection(Enum):
    """Opponent movement direction"""
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    NEUTRAL = "neutral"


@dataclass
class DefenderState:
    """State of the defender including position and velocity"""
    x: float  # x-coordinate on grid
    y: float  # y-coordinate on grid
    vx: float  # velocity in x-direction
    vy: float  # velocity in y-direction
    t: float  # current time

    def copy(self):
        """Create a copy of the state"""
        return DefenderState(self.x, self.y, self.vx, self.vy, self.t)


@dataclass
class Outcome:
    """Shuttle landing outcome: region and shot type"""
    region: int  # region index (0-8 for 3x3 grid)
    shot_type: ShotType
    probability: float  # P(region, shot_type | observations)
    deadline: float  # Tland(region, shot_type)


@dataclass
class ProblemInstance:
    """
    Complete instance of the Anticipatory Movement Optimization Problem
    """
    # Initial defender state
    initial_state: DefenderState

    # Opponent observations
    opponent_position: CourtPosition
    racquet_posture: str
    movement_direction: MovementDirection

    # Uncertain outcomes with probabilities
    outcomes: List[Outcome]

    # Problem parameters
    grid_size: int = 30  # Grid is grid_size x grid_size
    region_size: int = 10  # Each region is region_size x region_size
    delta: float = 1.0  # Spatial step size
    dt: float = 0.1  # Time step
    H: int = 5  # Anticipation window length (steps)
    a_max: float = 5.0  # Maximum acceleration magnitude
    c_h: float = 1.0  # Horizontal movement cost coefficient
    c_v: float = 1.5  # Vertical movement cost coefficient (cv > ch)

    def get_region_bounds(self, region_idx: int) -> Tuple[float, float, float, float]:
        """
        Get the spatial bounds of a region.
        Returns (x_min, x_max, y_min, y_max)
        """
        region_row = region_idx // 3
        region_col = region_idx % 3

        x_min = region_col * self.region_size
        x_max = (region_col + 1) * self.region_size
        y_min = region_row * self.region_size
        y_max = (region_row + 1) * self.region_size

        return x_min, x_max, y_min, y_max

    def is_in_region(self, x: float, y: float, region_idx: int) -> bool:
        """Check if position (x,y) is in the given region"""
        x_min, x_max, y_min, y_max = self.get_region_bounds(region_idx)
        return x_min <= x < x_max and y_min <= y < y_max

    def validate(self) -> bool:
        """Validate problem instance"""
        # Check probabilities sum to 1
        total_prob = sum(o.probability for o in self.outcomes)
        if not np.isclose(total_prob, 1.0):
            return False

        # Check all parameters are positive
        if self.dt <= 0 or self.H <= 0 or self.a_max <= 0:
            return False

        if self.c_h <= 0 or self.c_v <= 0 or self.c_v <= self.c_h:
            return False

        return True


class KinematicModel:
    """
    Discrete-time kinematic model with bounded acceleration
    """

    @staticmethod
    def step(state: DefenderState, ax: float, ay: float, dt: float) -> DefenderState:
        """
        Perform one kinematic step with acceleration (ax, ay)

        Returns new state after dt time
        """
        new_state = state.copy()

        # Update velocity: v_{k+1} = v_k + a * dt
        new_state.vx = state.vx + ax * dt
        new_state.vy = state.vy + ay * dt

        # Update position: x_{k+1} = x_k + v_k * dt + 0.5 * a * dt^2
        new_state.x = state.x + state.vx * dt + 0.5 * ax * dt**2
        new_state.y = state.y + state.vy * dt + 0.5 * ay * dt**2

        # Update time
        new_state.t = state.t + dt

        return new_state

    @staticmethod
    def apply_acceleration_window(state: DefenderState, ux: float, uy: float,
                                   H: int, dt: float) -> DefenderState:
        """
        Apply constant acceleration (ux, uy) for H time steps

        Returns state after anticipation window
        """
        current = state.copy()
        for _ in range(H):
            current = KinematicModel.step(current, ux, uy, dt)
        return current


class CostFunction:
    """
    Anisotropic movement cost function
    """

    @staticmethod
    def step_cost(dx: float, dy: float, c_h: float, c_v: float) -> float:
        """
        Cost of a single step displacement (dx, dy)

        cost = c_h * |dx| + c_v * |dy|
        """
        return c_h * abs(dx) + c_v * abs(dy)

    @staticmethod
    def trajectory_cost(trajectory: List[Tuple[float, float]],
                       c_h: float, c_v: float) -> float:
        """
        Total cost of a trajectory (list of (x, y) positions)
        """
        total_cost = 0.0
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            total_cost += CostFunction.step_cost(dx, dy, c_h, c_v)
        return total_cost


def compute_shot_deadline(shot_type: ShotType, distance: float) -> float:
    """
    Compute the deadline for reaching a landing region given shot type

    This is a simplified model based on shuttle speed
    """
    # Approximate shuttle speeds (grid units per second)
    speeds = {
        ShotType.SMASH: 50.0,
        ShotType.DRIVE: 35.0,
        ShotType.CLEAR: 25.0,
        ShotType.LIFT: 20.0,
        ShotType.DROP: 15.0,
        ShotType.NET: 10.0
    }

    # Estimate flight time based on distance and shot type
    # Add some buffer time for reaction and stroke preparation
    flight_time = distance / speeds[shot_type]
    buffer_time = 0.3  # 300ms buffer

    return flight_time + buffer_time


if __name__ == "__main__":
    # Example usage
    print("Anticipatory Defensive Movement Problem - Core Module")
    print("=" * 60)

    # Create a simple problem instance
    initial_state = DefenderState(x=15.0, y=15.0, vx=0.0, vy=0.0, t=0.0)

    outcomes = [
        Outcome(region=0, shot_type=ShotType.NET, probability=0.3, deadline=0.8),
        Outcome(region=2, shot_type=ShotType.SMASH, probability=0.4, deadline=0.5),
        Outcome(region=6, shot_type=ShotType.DROP, probability=0.3, deadline=1.0)
    ]

    problem = ProblemInstance(
        initial_state=initial_state,
        opponent_position=CourtPosition.FRONT,
        racquet_posture="overhead-prepare",
        movement_direction=MovementDirection.FORWARD,
        outcomes=outcomes
    )

    print(f"Problem instance created: {problem.validate()}")
    print(f"Initial state: ({problem.initial_state.x}, {problem.initial_state.y})")
    print(f"Number of outcomes: {len(problem.outcomes)}")
    print(f"Grid size: {problem.grid_size}x{problem.grid_size}")
    print(f"Anticipation window: {problem.H} steps = {problem.H * problem.dt}s")
