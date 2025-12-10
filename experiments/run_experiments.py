"""
Experimental Evaluation

Compares all algorithms on generated test instances:
- Exact solver (baseline)
- Approximation algorithms
- Heuristic algorithms

Metrics:
- Solution quality (cost)
- Runtime
- Approximation ratio (relative to optimal)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import json
import numpy as np
from typing import Dict, List, Tuple

from problem import ProblemInstance
from exact_solver import ExactSolver
from heuristics import (GreedyHeuristic, ExpectedPositionHeuristic,
                       MinMaxHeuristic, WeightedDirectionHeuristic,
                       AdaptiveHeuristic)
from approximation import GridApproximation, SamplingApproximation
from data_generator import DataGenerator


class ExperimentRunner:
    """Run comprehensive experiments"""

    def __init__(self, output_dir: str = "../results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_algorithm(self, name: str, problem: ProblemInstance,
                          solver_func) -> Dict:
        """
        Evaluate a single algorithm on a problem instance

        Returns dict with: cost, runtime, acceleration
        """
        start_time = time.time()

        try:
            result = solver_func()
            runtime = time.time() - start_time

            if len(result) == 2:
                accel, cost = result
                info = {}
            else:
                accel, cost, info = result

            return {
                'name': name,
                'acceleration': accel,
                'cost': float(cost),
                'runtime': runtime,
                'info': info,
                'success': True
            }

        except Exception as e:
            runtime = time.time() - start_time
            return {
                'name': name,
                'acceleration': None,
                'cost': float('inf'),
                'runtime': runtime,
                'info': {},
                'success': False,
                'error': str(e)
            }

    def run_single_instance(self, difficulty: str, instance: ProblemInstance,
                           instance_id: int) -> Dict:
        """
        Run all algorithms on a single instance

        Returns results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Instance {instance_id} ({difficulty})")
        print(f"{'='*70}")
        print(f"Initial: ({instance.initial_state.x:.1f}, {instance.initial_state.y:.1f}), "
              f"v=({instance.initial_state.vx:.1f}, {instance.initial_state.vy:.1f})")
        print(f"Outcomes: {len(instance.outcomes)}")

        results = {
            'instance_id': instance_id,
            'difficulty': difficulty,
            'n_outcomes': len(instance.outcomes),
            'algorithms': []
        }

        # Exact solver (baseline) - use reduced resolution for speed
        print("\n--- Exact Solver ---")
        exact_solver = ExactSolver(instance, n_directions=8, n_magnitudes=3)
        exact_result = self.evaluate_algorithm(
            "Exact",
            instance,
            lambda: exact_solver.solve(verbose=False)
        )
        results['algorithms'].append(exact_result)
        optimal_cost = exact_result['cost']
        print(f"Cost: {exact_result['cost']:.2f}, Time: {exact_result['runtime']:.3f}s")

        # Heuristics
        heuristics = [
            ("Greedy", GreedyHeuristic(instance)),
            ("Expected Position", ExpectedPositionHeuristic(instance)),
            ("MinMax", MinMaxHeuristic(instance)),
            ("Weighted Direction", WeightedDirectionHeuristic(instance)),
            ("Adaptive", AdaptiveHeuristic(instance))
        ]

        for name, heuristic in heuristics:
            print(f"\n--- {name} Heuristic ---")
            result = self.evaluate_algorithm(
                name,
                instance,
                lambda h=heuristic: h.solve()
            )

            # Compute approximation ratio
            if result['success'] and optimal_cost < float('inf'):
                result['approx_ratio'] = result['cost'] / optimal_cost
            else:
                result['approx_ratio'] = float('inf')

            results['algorithms'].append(result)
            print(f"Cost: {result['cost']:.2f}, Time: {result['runtime']:.3f}s, "
                  f"Ratio: {result.get('approx_ratio', 'N/A')}")

        # Approximation algorithms
        approx_algorithms = [
            ("Grid Approximation (res=8)", lambda: GridApproximation(instance, 8).solve(verbose=False)),
            ("Grid Approximation (res=12)", lambda: GridApproximation(instance, 12).solve(verbose=False)),
            ("Sampling (n=50)", lambda: SamplingApproximation(instance, 50).solve(verbose=False)),
            ("Sampling (n=100)", lambda: SamplingApproximation(instance, 100).solve(verbose=False)),
        ]

        for name, solver_func in approx_algorithms:
            print(f"\n--- {name} ---")
            result = self.evaluate_algorithm(name, instance, solver_func)

            # Compute approximation ratio
            if result['success'] and optimal_cost < float('inf'):
                result['approx_ratio'] = result['cost'] / optimal_cost
            else:
                result['approx_ratio'] = float('inf')

            results['algorithms'].append(result)
            print(f"Cost: {result['cost']:.2f}, Time: {result['runtime']:.3f}s, "
                  f"Ratio: {result.get('approx_ratio', 'N/A')}")

        return results

    def run_full_experiment(self, n_easy: int = 5, n_medium: int = 5,
                           n_hard: int = 5, n_adversarial: int = 3):
        """
        Run full experimental evaluation
        """
        print("="*70)
        print("EXPERIMENTAL EVALUATION")
        print("="*70)

        # Generate dataset
        generator = DataGenerator(seed=42)
        dataset = generator.generate_dataset(n_easy, n_medium, n_hard, n_adversarial)

        print(f"\nDataset: {len(dataset)} instances")
        print(f"  Easy: {n_easy}, Medium: {n_medium}, Hard: {n_hard}, Adversarial: {n_adversarial}")

        all_results = []

        for idx, (difficulty, instance) in enumerate(dataset):
            result = self.run_single_instance(difficulty, instance, idx)
            all_results.append(result)

        # Save results
        output_file = os.path.join(self.output_dir, "experiment_results.json")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n\nResults saved to: {output_file}")

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, all_results: List[Dict]):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)

        # Group by algorithm
        algorithm_stats = {}

        for result in all_results:
            for alg_result in result['algorithms']:
                name = alg_result['name']
                if name not in algorithm_stats:
                    algorithm_stats[name] = {
                        'costs': [],
                        'runtimes': [],
                        'ratios': []
                    }

                if alg_result['success']:
                    algorithm_stats[name]['costs'].append(alg_result['cost'])
                    algorithm_stats[name]['runtimes'].append(alg_result['runtime'])
                    if 'approx_ratio' in alg_result and alg_result['approx_ratio'] < 1e6:
                        algorithm_stats[name]['ratios'].append(alg_result['approx_ratio'])

        # Print table
        print(f"\n{'Algorithm':<25} {'Avg Cost':<12} {'Avg Time':<12} {'Avg Ratio':<12}")
        print("-" * 70)

        for name in sorted(algorithm_stats.keys()):
            stats = algorithm_stats[name]

            avg_cost = np.mean(stats['costs']) if stats['costs'] else float('inf')
            avg_time = np.mean(stats['runtimes']) if stats['runtimes'] else 0
            avg_ratio = np.mean(stats['ratios']) if stats['ratios'] else float('inf')

            cost_str = f"{avg_cost:.2f}" if avg_cost < 1e6 else "INF"
            time_str = f"{avg_time:.4f}s"
            ratio_str = f"{avg_ratio:.3f}" if avg_ratio < 1e6 else "N/A"

            print(f"{name:<25} {cost_str:<12} {time_str:<12} {ratio_str:<12}")

        # Group by difficulty
        print("\n\nPerformance by Difficulty:")
        print("-" * 70)

        for difficulty in ["easy", "medium", "hard", "adversarial"]:
            print(f"\n{difficulty.upper()}:")

            difficulty_results = [r for r in all_results if r['difficulty'] == difficulty]
            if not difficulty_results:
                continue

            for alg_name in ["Exact", "Greedy", "Expected Position", "Adaptive"]:
                costs = []
                for result in difficulty_results:
                    for alg_result in result['algorithms']:
                        if alg_result['name'] == alg_name and alg_result['success']:
                            costs.append(alg_result['cost'])

                if costs:
                    avg = np.mean(costs)
                    std = np.std(costs)
                    print(f"  {alg_name:<20}: {avg:.2f} ± {std:.2f}")


def main():
    """Main entry point"""
    runner = ExperimentRunner()

    # Run experiments (reduced size for faster execution)
    results = runner.run_full_experiment(
        n_easy=3,
        n_medium=3,
        n_hard=3,
        n_adversarial=2
    )

    print("\n✓ Experiments completed successfully!")


if __name__ == "__main__":
    main()
