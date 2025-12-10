# Anticipatory Defensive Movement in Badminton: NP-Hardness and Approximation Algorithms

**CSE 202 Final Project**

## Overview

This project studies the computational complexity of anticipatory defensive movement in badminton, where a defender must choose an initial acceleration vector to minimize expected movement cost under uncertainty about where the shuttle will land.

## Problem Formulation

**Key Components:**
- **State**: Defender position (x, y) and velocity (vx, vy) on 30×30 grid
- **Uncertainty**: Probabilistic outcomes over 9 landing regions and 6 shot types
- **Constraints**:
  - Time-window control: constant acceleration for H steps
  - Kinematic bounds: ||a||₂ ≤ aₘₐₓ
  - Time deadlines: must reach region g by Tₗₐₙd(g, type)
- **Cost**: Anisotropic movement cost: cₕ|Δx| + cᵥ|Δy| where cᵥ > cₕ

**Objective**: Choose acceleration u* to minimize expected movement cost

## Main Results

### 1. Complexity Theory

**Theorem**: The Anticipatory Defensive Movement problem is **NP-hard**.

**Proof Method**: Reduction from Time-Constrained Shortest Path (TCSP)

**Key Insight**: Even the deterministic case (single outcome) is NP-hard due to:
- Time deadline constraints
- Kinematic constraints (bounded acceleration + velocity dynamics)
- Continuous acceleration space optimization
- Anisotropic costs creating direction-dependent paths

### 2. Algorithms Implemented

#### Exact Algorithm
- **Method**: Discretize acceleration space + A* search for each outcome
- **Complexity**: O(|U_disc| · |G| · |T| · K²)
- **Implementation**: `src/exact_solver.py`

#### Approximation Algorithms

**Grid Approximation**
- Uniform grid over acceleration space
- Provable (1+ε)-approximation guarantee
- **Complexity**: O(r² · |G| · |T|)
- **Implementation**: `src/approximation.py`

**Monte Carlo Sampling**
- Random sampling from acceleration space
- Probabilistic guarantees
- **Complexity**: O(n · |G| · |T|)

#### Heuristic Strategies

1. **Greedy**: Accelerate toward highest-probability outcome
2. **Expected Position (Centroid)**: Toward probability-weighted center
3. **Min-Max**: Minimize worst-case distance
4. **Weighted Direction**: Probability-weighted direction sum
5. **Adaptive**: Select heuristic based on problem structure

**Implementation**: `src/heuristics.py`

### 3. Experimental Results

Dataset: 11 synthetic instances (easy, medium, hard, adversarial)

| Algorithm | Avg Cost | Avg Time (s) | Speedup |
|-----------|----------|--------------|---------|
| Exact (baseline) | 853,342 | 0.581 | 1× |
| Grid (res=8) | 17.39 | 0.0002 | **2900×** |
| Grid (res=12) | 17.33 | 0.0006 | 970× |
| Sampling (n=50) | 17.32 | 0.0006 | 970× |
| Sampling (n=100) | 17.30 | 0.0011 | 530× |

**Key Findings:**
- Approximation algorithms achieve **3 orders of magnitude speedup**
- High solution quality (near-optimal for feasible instances)
- Exact solver often fails on complex instances (high penalty costs)
- Grid and sampling methods scale well to hard problems

## Project Structure

```
cse202_project/
├── src/
│   ├── problem.py           # Problem definition & core models
│   ├── exact_solver.py      # Exact exponential-time solver
│   ├── heuristics.py        # Fast heuristic strategies
│   ├── approximation.py     # Approximation algorithms with guarantees
│   └── data_generator.py    # Synthetic instance generator
├── experiments/
│   └── run_experiments.py   # Experimental evaluation script
├── results/
│   └── experiment_results.json  # Detailed results
├── report/
│   └── report.tex           # Full technical report (LaTeX)
└── README.md                # This file
```

## Running the Code

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy
```

### Running Experiments

```bash
# Run full experimental evaluation
source venv/bin/activate
python experiments/run_experiments.py
```

This will:
1. Generate 11 test instances (easy/medium/hard/adversarial)
2. Run all algorithms on each instance
3. Compare solution quality and runtime
4. Save results to `results/experiment_results.json`

### Testing Individual Modules

```bash
# Test problem formulation
python src/problem.py

# Test exact solver
python src/exact_solver.py

# Test heuristics
python src/heuristics.py

# Test approximation algorithms
python src/approximation.py

# Test data generator
python src/data_generator.py
```

## Technical Highlights

### NP-Hardness Proof

The proof constructs a polynomial-time reduction from TCSP:

1. **Vertex mapping**: Map graph vertices to grid positions
2. **Single outcome**: Set deterministic landing region (destination vertex)
3. **Cost encoding**: Design anisotropic costs matching edge weights
4. **Time encoding**: Set deadlines matching edge traversal times
5. **Query**: Minimum cost ≤ budget?

Since even computing C_u(g, type) for one outcome is NP-hard, and the full problem optimizes over continuous space U, the problem is NP-hard.

### Approximation Guarantee

**Grid Approximation Theorem:**

Let ε = (max_pos_error · 2c_v · H) / min_meaningful_cost where:
- max_pos_error = ½ · (2a_max/(r-1)) · (H·δt)²

Then grid approximation with resolution r returns (1+ε)-approximate solution.

**Proof Sketch**: Grid discretization introduces bounded position error. This propagates through kinematic evolution and cost function. By controlling grid resolution r, we bound approximation ratio.

### Experimental Insights

1. **Exact solver limitations**: Coarse discretization (8 directions × 3 magnitudes = 25 candidates) insufficient for complex instances

2. **Approximation effectiveness**: Finer-grained grid/sampling methods find feasible solutions where exact fails

3. **Speed-quality tradeoff**:
   - Grid (res=8): Fastest, slight quality loss
   - Grid (res=12): Best balance
   - Sampling (n=100): Highest quality, slightly slower

4. **Practical applicability**: All approximation methods < 1ms average runtime → suitable for real-time decision support

## Future Work

1. **Learning-based approaches**: RL to learn policies from game data
2. **Multi-step planning**: Sequential decision-making over rally
3. **Opponent modeling**: Incorporate learned opponent behavior models
4. **Real-world validation**: Test on actual badminton match data
5. **Tighter bounds**: Improve theoretical approximation guarantees
6. **Parallelization**: Exploit GPU for acceleration space search

## References

- Garey & Johnson (1979): Computers and Intractability
- Papadimitriou & Steiglitz (1998): Combinatorial Optimization
- LaValle (2006): Planning Algorithms
- Karaman & Frazzoli (2011): Sampling-based Optimal Motion Planning

## Authors

Rui Wu & Yiqian Liu
UC San Diego

## License

This project is for educational purposes (CSE 202 course project).
