"""
Test Data Generator

Generates diverse problem instances for evaluation:
1. Easy instances (few outcomes, concentrated)
2. Medium instances (moderate uncertainty)
3. Hard instances (many outcomes, spread out, tight deadlines)
4. Adversarial instances (designed to challenge heuristics)
"""

import numpy as np
from typing import List
from problem import (ProblemInstance, DefenderState, Outcome, ShotType,
                     CourtPosition, MovementDirection)


class DataGenerator:
    """Generate test instances with controlled properties"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_easy_instance(self, instance_id: int = 0) -> ProblemInstance:
        """
        Easy instance: 2-3 outcomes, high probability concentration
        """
        # Start from center
        initial_state = DefenderState(
            x=15.0, y=15.0,
            vx=self.rng.uniform(-1, 1),
            vy=self.rng.uniform(-1, 1),
            t=0.0
        )

        # 2-3 outcomes concentrated in one area
        n_outcomes = self.rng.randint(2, 4)

        # Choose a dominant region
        dominant_region = self.rng.randint(0, 9)

        outcomes = []
        dominant_prob = self.rng.uniform(0.6, 0.8)
        outcomes.append(Outcome(
            region=dominant_region,
            shot_type=self.rng.choice(list(ShotType)),
            probability=dominant_prob,
            deadline=self.rng.uniform(1.0, 2.0)
        ))

        # Add other outcomes with lower probabilities
        remaining_prob = 1.0 - dominant_prob
        for i in range(n_outcomes - 1):
            prob = remaining_prob / (n_outcomes - 1)
            # Choose nearby region
            region = (dominant_region + self.rng.randint(-1, 2)) % 9
            outcomes.append(Outcome(
                region=region,
                shot_type=self.rng.choice(list(ShotType)),
                probability=prob,
                deadline=self.rng.uniform(1.0, 2.5)
            ))

        return ProblemInstance(
            initial_state=initial_state,
            opponent_position=self.rng.choice(list(CourtPosition)),
            racquet_posture="overhead",
            movement_direction=self.rng.choice(list(MovementDirection)),
            outcomes=outcomes
        )

    def generate_medium_instance(self, instance_id: int = 0) -> ProblemInstance:
        """
        Medium instance: 4-6 outcomes, moderate spread
        """
        # Random starting position
        initial_state = DefenderState(
            x=self.rng.uniform(10, 20),
            y=self.rng.uniform(10, 20),
            vx=self.rng.uniform(-2, 2),
            vy=self.rng.uniform(-2, 2),
            t=0.0
        )

        # 4-6 outcomes with moderate probabilities
        n_outcomes = self.rng.randint(4, 7)

        # Generate Dirichlet distribution for probabilities
        alpha = np.ones(n_outcomes) * 2.0  # Moderate concentration
        probs = self.rng.dirichlet(alpha)

        outcomes = []
        used_regions = set()

        for i in range(n_outcomes):
            # Choose distinct regions
            region = self.rng.randint(0, 9)
            while region in used_regions and len(used_regions) < 9:
                region = self.rng.randint(0, 9)
            used_regions.add(region)

            outcomes.append(Outcome(
                region=region,
                shot_type=self.rng.choice(list(ShotType)),
                probability=float(probs[i]),
                deadline=self.rng.uniform(0.8, 2.0)
            ))

        return ProblemInstance(
            initial_state=initial_state,
            opponent_position=self.rng.choice(list(CourtPosition)),
            racquet_posture="overhead",
            movement_direction=self.rng.choice(list(MovementDirection)),
            outcomes=outcomes
        )

    def generate_hard_instance(self, instance_id: int = 0) -> ProblemInstance:
        """
        Hard instance: 7-9 outcomes, uniform distribution, tight deadlines
        """
        # Random starting position
        initial_state = DefenderState(
            x=self.rng.uniform(5, 25),
            y=self.rng.uniform(5, 25),
            vx=self.rng.uniform(-3, 3),
            vy=self.rng.uniform(-3, 3),
            t=0.0
        )

        # Many outcomes with nearly uniform probabilities
        n_outcomes = self.rng.randint(7, 10)

        # Uniform-ish probabilities
        alpha = np.ones(n_outcomes) * 0.5  # Low concentration = more uniform
        probs = self.rng.dirichlet(alpha)

        outcomes = []
        regions = self.rng.choice(9, size=n_outcomes, replace=True)

        for i in range(n_outcomes):
            outcomes.append(Outcome(
                region=int(regions[i]),
                shot_type=self.rng.choice(list(ShotType)),
                probability=float(probs[i]),
                deadline=self.rng.uniform(0.5, 1.5)  # Tighter deadlines
            ))

        return ProblemInstance(
            initial_state=initial_state,
            opponent_position=self.rng.choice(list(CourtPosition)),
            racquet_posture="overhead",
            movement_direction=self.rng.choice(list(MovementDirection)),
            outcomes=outcomes
        )

    def generate_adversarial_instance(self, instance_id: int = 0) -> ProblemInstance:
        """
        Adversarial: designed to challenge greedy heuristics
        Highest probability outcome is far, but moderate probability outcome is near
        """
        # Use instance_id to vary the instances
        corners = [(5.0, 5.0), (25.0, 5.0), (5.0, 25.0), (25.0, 25.0)]
        corner_idx = instance_id % len(corners)
        start_x, start_y = corners[corner_idx]

        # Start from corner
        initial_state = DefenderState(
            x=start_x, y=start_y,
            vx=self.rng.uniform(-0.5, 0.5),
            vy=self.rng.uniform(-0.5, 0.5),
            t=0.0
        )

        # Vary the probabilities slightly based on instance_id
        high_prob = 0.5 + (instance_id % 3) * 0.05
        remaining = 1.0 - high_prob

        # Choose far vs near regions based on starting corner
        far_region = 8 - (corner_idx * 2)  # Opposite corners
        near_region = corner_idx

        outcomes = [
            # High probability but far away (opposite corner)
            Outcome(
                region=far_region,
                shot_type=ShotType.CLEAR,
                probability=high_prob,
                deadline=1.0 + instance_id * 0.1  # Vary deadline
            ),
            # Medium probability but nearby
            Outcome(
                region=near_region,
                shot_type=ShotType.NET,
                probability=remaining * 0.6,
                deadline=1.5 + instance_id * 0.05
            ),
            # Low probability, medium distance
            Outcome(
                region=4,  # Center
                shot_type=ShotType.DRIVE,
                probability=remaining * 0.4,
                deadline=1.2 + instance_id * 0.08
            )
        ]

        return ProblemInstance(
            initial_state=initial_state,
            opponent_position=CourtPosition.REAR,
            racquet_posture="overhead",
            movement_direction=MovementDirection.FORWARD,
            outcomes=outcomes
        )

    def generate_dataset(self, n_easy: int = 10, n_medium: int = 10,
                        n_hard: int = 10, n_adversarial: int = 5) -> List[ProblemInstance]:
        """
        Generate complete dataset with mix of difficulties

        Returns list of (difficulty, instance) tuples
        """
        dataset = []

        for i in range(n_easy):
            instance = self.generate_easy_instance(i)
            dataset.append(("easy", instance))

        for i in range(n_medium):
            instance = self.generate_medium_instance(i)
            dataset.append(("medium", instance))

        for i in range(n_hard):
            instance = self.generate_hard_instance(i)
            dataset.append(("hard", instance))

        for i in range(n_adversarial):
            instance = self.generate_adversarial_instance(i)
            dataset.append(("adversarial", instance))

        return dataset


if __name__ == "__main__":
    print("Data Generator Test")
    print("=" * 60)

    generator = DataGenerator(seed=42)

    # Generate samples of each difficulty
    difficulties = ["easy", "medium", "hard", "adversarial"]
    methods = [
        generator.generate_easy_instance,
        generator.generate_medium_instance,
        generator.generate_hard_instance,
        generator.generate_adversarial_instance
    ]

    for difficulty, method in zip(difficulties, methods):
        print(f"\n{difficulty.upper()} Instance:")
        print("-" * 60)

        instance = method(0)
        print(f"  Initial position: ({instance.initial_state.x:.1f}, {instance.initial_state.y:.1f})")
        print(f"  Initial velocity: ({instance.initial_state.vx:.1f}, {instance.initial_state.vy:.1f})")
        print(f"  Number of outcomes: {len(instance.outcomes)}")

        print("  Outcomes:")
        for outcome in instance.outcomes:
            print(f"    Region {outcome.region}: {outcome.shot_type.value}, "
                  f"p={outcome.probability:.2f}, deadline={outcome.deadline:.2f}s")

    # Generate full dataset
    print("\n\nGenerating full dataset...")
    dataset = generator.generate_dataset(n_easy=5, n_medium=5, n_hard=5, n_adversarial=3)
    print(f"Total instances: {len(dataset)}")

    difficulty_counts = {}
    for difficulty, _ in dataset:
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    print("Distribution:")
    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty}: {count}")
