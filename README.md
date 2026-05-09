# UAV Metaheuristic Optimization — Benchmark Simulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![IEEE COMST](https://img.shields.io/badge/Journal-IEEE%20COMST-orange)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9739)

> **Reproducible simulation benchmark** for the paper:  
> *"Metaheuristic Optimization for UAV-Assisted Wireless Networks: A Systematic Survey on Trajectory Planning, Node Deployment, Resource Allocation, and Swarm Coordination"*  
> **Ali Jameel Hashim, Abbas I. Al-Zaki** — *IEEE Communications Surveys & Tutorials, 2026*

---

## Overview

This repository provides the complete open-source simulation code that supports **Section VIII** of the survey paper. It compares four metaheuristic algorithms — Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Differential Evolution (DE), and Ant Colony Optimization (ACO) — on two canonical UAV network optimization problems using a standardized synthetic environment.

The simulation environment is built on:
- **ITU-R probabilistic air-to-ground channel model** (path loss and LoS probability)

---

## What This Code Produces

| Output | Description |
|---|---|
| **Fig. 8** | Convergence curves with mean ± std bands and IQR bands |
| **Fig. 8b** | Optimized UAV trajectory maps per algorithm with signal heatmap |
| **Fig. 10** | UAV deployment coverage maps with node association lines |
| **Fig. 17** | Scalability analysis — runtime vs. network size (3 panels) |
| **Fig. 18** | Performance heatmap + radar chart + box plots |
| **Fig. Extra** | Statistical significance matrix (Wilcoxon p-values + Cohen's d) |
| **results_table_full.csv** | Complete descriptive statistics and pairwise test results |

---

## Problems Evaluated

### 1 — Trajectory Planning
A single UAV navigates from `(0, 0)` to `(1000, 1000)` through 10 optimizable waypoints in a 1000 m × 1000 m area with 4 no-fly zones. The objective minimizes a weighted combination of path length, energy consumption, and no-fly zone penalty, while maximizing coverage of 20 ground nodes.

- **Search space dimensionality:** d = 20
- **Fitness:** `0.35*(dist/1000) + 0.30*(energy/1000) - 0.25*(coverage/100) + 0.10*nfz_penalty`

### 2 — Node Deployment
4 UAVs are placed as aerial base stations to maximize coverage of 50 ground nodes within a 150 m communication radius, subject to interference and no-fly zone constraints.

- **Search space dimensionality:** d = 8
- **Fitness:** `-(coverage_ratio) + 0.002*interference + nfz_penalty`

---

## Algorithm Parameters

| Algorithm | Parameter | Value |
|---|---|---|
| **GA** | Population | 80 |
| | Mutation rate | Adaptive [0.05, 0.40], linearly decreasing |
| | Elitism | 10% top individuals preserved |
| **PSO** | Swarm size | 80 |
| | Inertia w | [0.4, 0.9], linear decay |
| | c1, c2 | 2.05 (Clerc-Kennedy constriction) |
| **DE** | Population | 80 |
| | F (scale factor) | [0.5, 0.8], adaptive |
| | CR (crossover) | [0.8, 0.95], adaptive |
| **ACO** | Archive size | 30 |
| | q (rank weight) | 0.5 (Gaussian kernel) |
| | xi (evaporation) | 0.85 |

**Standard IEEE experimental settings:** 30 independent runs × 200 iterations × population = 80

---

## Installation

```bash
# Clone the repository
git clone https://github.com/programmer-alijameel2020/UAV-Metaheuristic-Survey-Benchmark.git
cd UAV-Metaheuristic-Survey-Benchmark

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Full Simulation (IEEE COMST — 30 runs, ~25 minutes)

```bash
python uav_simulation_comst.py
```

This runs the full 30-run experiment as reported in the paper. Each figure appears as an interactive window — close one to proceed to the next.

### Quick Test (5 runs, ~5 minutes)

To verify the code runs without waiting for the full experiment, set the quick test flag at the top of the script:

```python
# Line 56 in uav_simulation_comst.py
N_RUNS = 5      # Change to 30 for paper results
N_ITER = 50     # Change to 200 for paper results
```

---

## Environment Parameters

```python
AREA          = 1000.0   # m  — simulation field size
ALT           = 100.0    # m  — fixed UAV altitude
V_MAX         = 30.0     # m/s — maximum UAV speed
P_TX          = 0.1      # W  — transmit power
BW            = 1e6      # Hz — bandwidth
FC            = 2.4e9    # Hz — carrier frequency

# ITU-R urban LoS parameters
A_ITU, B_ITU  = 9.61, 0.16
ETA_LOS       = 1.0      # dB — LoS excess path loss
ETA_NLOS      = 20.0     # dB — NLoS excess path loss

# Zeng 2019 rotary-wing energy model
P0, P1        = 79.856, 88.628  # W — blade profile, induced power
V_TIP         = 120.0    # m/s — rotor tip speed
V0            = 4.03     # m/s — mean rotor induced velocity
```

---

## Output Files

All outputs are saved to the `outputs/` folder automatically created at runtime.

| File | Description |
|---|---|
| `results_table_full.csv` | Full statistics table: mean, std, best, worst, median, IQR, CV%, convergence iteration, runtime, Friedman rank, improvement %, Wilcoxon W-statistic, p-value, Cohen's d |

All figures are displayed interactively. To save them as files, add `plt.savefig(...)` before each `plt.show()` call, or replace `plt.show()` with `plt.savefig('figXX.pdf', dpi=300, bbox_inches='tight')`.

---

## Reported Results (30 runs, from paper)

### Trajectory Planning

| Algorithm | Mean ± Std | Best | Friedman Rank |
|---|---|---|---|
| **ACO** | **4.9539 ± 0.2091** | **4.6550** | **1.47** |
| PSO | 5.0170 ± 0.1858 | 4.7100 | 1.77 |
| DE | 5.4665 ± 0.3971 | 4.9970 | 3.10 |
| GA | 5.6542 ± 0.2494 | 5.2104 | 3.67 |

Friedman test: χ² = 59.88, p < 10⁻⁶ (highly significant)

### Node Deployment

| Algorithm | Mean ± Std | Best | Friedman Rank |
|---|---|---|---|
| **PSO** | **−0.6040 ± 0.0367** | **−0.6400** | **1.33** |
| GA | −0.5693 ± 0.0415 | −0.6400 | 1.87 |
| ACO | −0.5041 ± 0.0572 | −0.6400 | 3.10 |
| DE | −0.4620 ± 0.0195 | −0.5200 | 3.70 |

Friedman test: χ² = 67.02, p < 10⁻⁶ (highly significant)

> **Note:** Lower fitness = better for both problems. Deployment fitness is negative (coverage ratio negated).

---

## Key Findings

1. **No single algorithm dominates across problem classes.** ACO leads on trajectory planning while PSO leads on deployment, consistent with the no-free-lunch theorem.

2. **Convergence speed and solution quality are inversely correlated.** GA converges fastest (iteration 10.9) but achieves the worst mean fitness. PSO and ACO converge near iteration 187–198 but find significantly better solutions.

3. **DE exhibits problem-dependent sensitivity.** DE's worst-case trajectory fitness (7.2744 vs. mean 5.4665) indicates occasional catastrophic divergence in high-dimensional spaces (d = 20).

4. **PSO and ACO are statistically equivalent on trajectory planning.** Wilcoxon test p = 0.1403, Cohen's d = 0.319 (small effect) — PSO may be preferred for its simpler implementation.

---

## Citation

If you use this code in your research, please cite both the paper and this repository:

```bibtex
@article{hashim2026metaheuristic,
  author    = {Hashim, Ali Jameel and Al-Zaki, Abbas I.},
  title     = {Metaheuristic Optimization for {UAV}-Assisted Wireless Networks:
               {A} Systematic Survey on Trajectory Planning, Node Deployment,
               Resource Allocation, and Swarm Coordination},
  journal   = {IEEE Communications Surveys \& Tutorials},
  year      = {2026},
  note      = {Under review}
}

@misc{hashim2026uavsim,
  author    = {Hashim, Ali Jameel and Al-Zaki, Abbas I.},
  title     = {{UAV} Metaheuristic Survey Benchmark --- Simulation Code},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/programmer-alijameel2020/UAV-Metaheuristic-Survey-Benchmark}
}
```

---

## Repository Structure

```
UAV-Metaheuristic-Survey-Benchmark/
│
├── uav_simulation_comst.py   # Main simulation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
│
└── outputs/                   # Auto-created at runtime
    └── results_table_full.csv # Statistics and test results
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Ali Jameel Hashim**  
Communication and Media Commission of Iraq (CMC)  
Ministry of Education, Baghdad, Iraq  
Email: a.jameel@cmc.iq

**Abbas I. Al-Zaki**  
Ministry of Education, Baghdad, Iraq  
Email: abbas.alzaki18@ec.edu.iq
