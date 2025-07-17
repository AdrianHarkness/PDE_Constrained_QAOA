# Solving Discretized PDE-Constrained Optimization Problems with QAOA

This project demonstrates how to solve discretized partial differential equation (PDE) constrained optimization problems using the Quantum Approximate Optimization Algorithm (QAOA). Two main approaches are explored:

- **Penalty Method** (`PDE_Penalty.ipynb`): Encodes the Hamming weight constraint as a penalty term in the QUBO formulation.
- **XY-Mixer Method** (`PDE_XY_Mixer.ipynb`): Uses a custom XY-mixer in QAOA to enforce the Hamming weight constraint directly, leveraging quantum circuit design.

## Notebooks

- **PDE_Penalty.ipynb**: 
  - Loads a discretized PDE control problem from MATLAB `.mat` files (see `ifiss3.6/QUBO_4.mat`).
  - Constructs the QUBO with a penalty for the Hamming weight constraint.
  - Solves the problem using classical and quantum (QAOA) optimizers.
  - Compares noiseless and noisy QAOA performance.
  - Generates the plot `Penalty_hamming_and_approx_ratio.png` showing approximation ratios and Hamming weights.

- **PDE_XY_Mixer.ipynb**:
  - Loads the same PDE control problem.
  - Constructs the problem as a binary quadratic optimization (BQP) with an explicit Hamming weight constraint.
  - Implements a custom XY-mixer (ring or full) to enforce the constraint in QAOA.
  - Solves using classical and quantum (QAOA) optimizers.
  - Compares noiseless and noisy QAOA performance.
  - Generates the plot `XY_hamming_and_approx_ratio.png`.

## Data Requirements

- MATLAB `.mat` files are required for the problem instance. Example: `ifiss3.6/QUBO_4.mat`.
- The `ifiss3.6/` directory contains all necessary data and scripts for generating or understanding the problem instances.

## Dependencies

- Python 3.10+
- All required Python packages are listed in `requirements.txt`.
- Key packages: `qiskit`, `qiskit-aer`, `qiskit-algorithms`, `qiskit-optimization`, `numpy`, `scipy`, `matplotlib`, `networkx`.

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Running the Notebooks

1. Ensure the required data files (e.g., `QUBO_4.mat`) are present in `ifiss3.6/`.
2. Install dependencies as above.
3. Open either notebook in Jupyter:
   - `PDE_Penalty.ipynb` for the penalty approach.
   - `PDE_XY_Mixer.ipynb` for the XY-mixer approach.
4. Run all cells to reproduce the results and plots.

## Output and Results

- Both notebooks output solution statistics and generate plots:
  - `Penalty_hamming_and_approx_ratio.png`: Shows approximation ratio and Hamming weight distributions for the penalty method.
  - `XY_hamming_and_approx_ratio.png`: Same, for the XY-mixer method.
- Results compare classical, noiseless QAOA, and noisy QAOA performance.

## References
- See the notebooks for detailed code, mathematical background, and further discussion.
- Data and problem setup are based on the IFISS MATLAB toolbox (see `ifiss3.6/`).
