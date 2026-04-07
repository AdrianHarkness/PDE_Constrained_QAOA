"""
Shared utilities for PDE-constrained QAOA experiments.

Sections:
    - Data Loading
    - Problem Setup (QUBO generation, penalty, BQO)
    - QAOA Components (initial states, mixers, callbacks)
    - Execution (parallel optimizer runner)
    - Sample Processing & Analysis
    - Plotting
    - Depth Sweep & Resource Estimation
    - Landscape (statevector simulation)
"""

import math
import time
import concurrent.futures
from itertools import combinations

import numpy as np
import scipy.io
from scipy.optimize import minimize as scipy_minimize
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import Initialize
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2

from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA, SPSA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram


# ============================================================
# Data Loading
# ============================================================

def load_pde_data(mat_path):
    """Load PDE-QUBO data from a .mat file.

    Returns (Phi, M, Kinv, yhat_vec) as dense NumPy arrays.
    """
    data = scipy.io.loadmat(mat_path)
    Phi = np.array(data["Phi"])
    M = np.array(data["M"].toarray())
    Kinv = np.array(data["Kinv"])
    yhat_vec = np.array(data["yhat_vec"]).squeeze()
    return Phi, M, Kinv, yhat_vec


# ============================================================
# Problem Setup
# ============================================================

def generate_pde_qubo(M, Kinv, Phi, yd):
    """Build quadratic objective matrices Q, q, c from PDE matrices."""
    Q = np.array(0.5 * (Phi.T @ M @ Kinv @ M @ Kinv @ M @ Phi))
    q = np.array(0.5 * (-(yd.T @ M @ Kinv @ M @ Phi).T - (Phi.T @ M @ Kinv.T @ M @ yd)))
    c = np.array(0.5 * yd.T @ M @ yd)
    return Q, q, c


def pad_zeros(Q, q, n_extra):
    """Zero-pad Q and q with n_extra additional variables."""
    n = Q.shape[0]
    total = n + n_extra
    padded_Q = np.zeros((total, total))
    padded_Q[:n, :n] = Q
    padded_q = np.zeros(total)
    padded_q[:n] = q
    return padded_Q, padded_q


def compute_lambda(Q, q, c, s):
    """Compute tight penalty parameter via continuous relaxation gap."""
    n = q.shape[0]

    def objective(u):
        return u @ Q @ u + q @ u + c

    result = scipy_minimize(objective, np.zeros(n), bounds=[(0, 1)] * n, method="SLSQP")
    if not result.success:
        raise RuntimeError("Relaxed problem did not converge")
    lb = result.fun

    top_s = np.argsort(-result.x)[:s]
    u_feas = np.zeros(n)
    u_feas[top_s] = 1
    ub = u_feas @ Q @ u_feas + q @ u_feas + c
    return ub - lb


def build_penalty_matrices(Q, q, c, s, lam):
    """Build raw penalty QUBO matrices with binary-expansion slack.

    Returns (Q_total, q_total, c_total, total_vars).
    """
    n = q.shape[0]
    log_s = int(np.ceil(np.log2(s + 1)))
    total_vars = n + log_s

    padded_Q = np.zeros((total_vars, total_vars))
    padded_Q[:n, :n] = Q
    padded_q = np.zeros(total_vars)
    padded_q[:n] = q

    e = np.ones(n)
    a = np.array([2**j for j in range(log_s)])
    t = np.concatenate([e, a])

    Q_total = padded_Q + lam * np.outer(t, t)
    q_total = padded_q + lam * (-2 * s * t)
    c_total = float(c + lam * s**2)
    return Q_total, q_total, c_total, total_vars


def build_penalty_qubo(Q, q, c, s, lam, name="Penalty QUBO"):
    """Build penalty QUBO as a QuadraticProgram (unconstrained)."""
    Q_total, q_total, c_total, total_vars = build_penalty_matrices(Q, q, c, s, lam)
    qp = QuadraticProgram(name)
    for i in range(total_vars):
        qp.binary_var(name=f"x{i}")
    qp.minimize(constant=c_total, linear=q_total, quadratic=Q_total)
    return qp


def build_bqo_matrices(Q, q, c, s):
    """Build raw padded matrices for BQO (n + s variables, no constraint).

    Returns (padded_Q, padded_q, c, total_vars).
    """
    n = q.shape[0]
    total = n + s
    padded_Q = np.zeros((total, total))
    padded_Q[:n, :n] = Q
    padded_q = np.zeros(total)
    padded_q[:n] = q
    return padded_Q, padded_q, float(c), total


def build_bqo(Q, q, c, s, constraint_sense="==", name="PDE constrained BQO"):
    """Build constrained BQO as a QuadraticProgram with Hamming weight constraint.

    Args:
        constraint_sense: "==" or "<=" for the Hamming weight constraint.
    """
    padded_Q, padded_q, c_val, total_vars = build_bqo_matrices(Q, q, c, s)
    qp = QuadraticProgram(name)
    for i in range(total_vars):
        qp.binary_var(name=f"x{i}")
    qp.linear_constraint(
        linear={f"x{i}": 1 for i in range(total_vars)},
        sense=constraint_sense,
        rhs=s,
        name="hamming_weight",
    )
    qp.minimize(constant=c_val, linear=padded_q, quadratic=padded_Q)
    return qp


# ============================================================
# QAOA Components
# ============================================================

def prepare_dicke_state(n, s):
    """
    Prepare a Dicke state with hamming weight s on n qubits.
    Creates a uniform superposition over all states with exactly s qubits in |1⟩.
    """
    qc = QuantumCircuit(n)
    
    dicke_basis = list(combinations(range(n), s))
    norm = 1 / math.sqrt(len(dicke_basis))
    
    state = [0] * (2**n)
    for basis in dicke_basis:
        index = sum([1 << (n - 1 - i) for i in basis])
        state[index] = norm
    
    # Initialize the state (non-unitary operation, but works for initial state)
    qc.initialize(state, list(range(n)))
    return qc


def build_ring_xy_mixer(n_qubits, param_name="β"):
    """Ring-topology XY mixer: RXX + RYY on adjacent qubit pairs."""
    beta = Parameter(param_name)
    qc = QuantumCircuit(n_qubits, name="Ring XY Mixer")
    for idx in range(n_qubits):
        idy = (idx + 1) % n_qubits
        qc.rxx(2 * beta, idx, idy)
        qc.ryy(2 * beta, idx, idy)
        qc.barrier()
    return qc


def build_full_xy_mixer(n_qubits, param_name="β"):
    """All-to-all XY mixer: RXX + RYY on every qubit pair."""
    beta = Parameter(param_name)
    qc = QuantumCircuit(n_qubits, name="Full XY Mixer")
    for idx in range(n_qubits):
        for idy in range(idx):
            qc.rxx(2 * beta, idx, idy)
            qc.ryy(2 * beta, idx, idy)
            qc.barrier()
    return qc


def make_callback(store):
    """Create a QAOA optimization callback that logs to *store*."""
    def callback(eval_count, parameters, value, metadata):
        store["evals"].append(eval_count)
        store["values"].append(value)
    return callback


# ============================================================
# Classical Solve
# ============================================================

def solve_classically(model):
    """Exact solve via NumPyMinimumEigensolver. Returns the result object."""
    solver = NumPyMinimumEigensolver()
    opt = MinimumEigenOptimizer(solver)
    return opt.solve(model)


# ============================================================
# Execution
# ============================================================

def run_optimizer(optimizer, name, model):
    """Thin wrapper for ThreadPoolExecutor compatibility."""
    result = optimizer.solve(model)
    return {"name": name, "result": result}


def run_parallel_qaoa(noiseless_opt, noisy_opt, model):
    """Run noiseless and noisy QAOA in parallel via threads.

    Returns (noiseless_result, noisy_result).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_clean = executor.submit(run_optimizer, noiseless_opt, "Noiseless", model)
        f_noisy = executor.submit(run_optimizer, noisy_opt, "Noisy", model)
        return f_clean.result()["result"], f_noisy.result()["result"]


def setup_noiseless_backend():
    """Return (backend, pass_manager) for noiseless Aer simulation."""
    backend = AerSimulator()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    return backend, pm


def setup_noisy_backend():
    """Return (backend, pass_manager, noise_model) from FakeAlmadenV2."""
    noisy_backend = AerSimulator.from_backend(FakeAlmadenV2())
    pm = generate_preset_pass_manager(optimization_level=3, backend=noisy_backend)
    noise_model = noisy_backend.options.noise_model
    return noisy_backend, pm, noise_model


# ============================================================
# Sample Processing & Analysis
# ============================================================

def process_samples(samples, objective, n_original, sort=True):
    """Re-evaluate fvals via *objective*, strip x to first n_original bits."""
    for samp in samples:
        samp.fval = objective.evaluate(samp.x)
        samp.x = samp.x[:n_original]
    if sort:
        samples.sort(key=lambda s: s.fval)
    return samples


def filter_feasible(samples, s, n_vars=None):
    """Return samples where Hamming weight of first *n_vars* bits <= s."""
    if n_vars is None:
        return [samp for samp in samples if sum(samp.x) <= s]
    return [samp for samp in samples if sum(samp.x[:n_vars]) <= s]


def compute_feasibility_prob(samples, s, n_vars=None):
    """P(feasible) = sum of probabilities for feasible samples."""
    feasible = filter_feasible(samples, s, n_vars)
    return sum(samp.probability for samp in feasible)


def compute_conditional_expected_ar(samples, optimal_value, s, n_vars=None):
    """E[approximation ratio | feasible], renormalized by P(feasible).

    Returns (expected_ar, feasibility_prob).
    """
    feasible = [
        samp for samp in filter_feasible(samples, s, n_vars) if samp.fval > 0
    ]
    feas_prob = sum(samp.probability for samp in filter_feasible(samples, s, n_vars))
    if feas_prob == 0:
        return 0.0, 0.0
    exp_ar = sum(
        samp.probability * (optimal_value / samp.fval) for samp in feasible
    ) / feas_prob
    return exp_ar, feas_prob


def check_feasibility_penalty(samples, n_original, s):
    """Check feasibility for penalty QUBO samples (with binary-expansion slack).

    Samples must NOT have been stripped yet (full x including slack bits).
    """
    log_s = int(np.ceil(np.log2(s + 1)))
    total_count = len(samples)
    infeasible = 0
    for samp in samples:
        x = samp.x
        u = x[:n_original]
        v = x[n_original : n_original + log_s]
        weights = np.array([2**j for j in range(log_s)])
        if np.sum(u) + np.dot(weights, v) != s:
            infeasible += 1
    ratio = (total_count - infeasible) / total_count
    print(f"Total: {total_count}, Infeasible: {infeasible}, Feasibility: {ratio:.2f}")
    return ratio


def check_feasibility_hamming(samples, s, n_vars=None):
    """Check feasibility using simple Hamming weight <= s on first n_vars bits."""
    total = len(samples)
    infeasible = sum(
        1 for samp in samples
        if sum(samp.x[:n_vars] if n_vars else samp.x) > s
    )
    ratio = (total - infeasible) / total
    print(f"Total: {total}, Infeasible: {infeasible}, Feasibility: {ratio:.2f}")
    return ratio


def print_feasible_summary(samples, optimal_value, s, label=""):
    """Print best and most-probable feasible sample statistics."""
    feasible = [samp for samp in samples if sum(samp.x) <= s and samp.fval > 0]
    header = f"===={label} (feasible only)===="
    print(header)

    if not feasible:
        print("No feasible samples found.")
        return

    feasible.sort(key=lambda x: x.fval)
    best = feasible[0]
    fmt = [int(xi) for xi in best.x]
    print(f"Best sample: {fmt} with probability {best.probability}")
    print(f"Best value: {best.fval}")
    print(f"Best approx ratio: {optimal_value / best.fval}")
    print()

    feasible.sort(key=lambda x: x.probability, reverse=True)
    mp = feasible[0]
    fmt_mp = [int(xi) for xi in mp.x]
    print(f"Most probable feasible sample: {fmt_mp} with probability {mp.probability}")
    print(f"Most probable feasible value: {mp.fval}")
    print(f"Most probable feasible approx ratio: {optimal_value / mp.fval}")


# ============================================================
# Plotting
# ============================================================

def plot_pde_inputs(Phi, M, Kinv, yhat_vec):
    """2x2 heatmap overview of PDE input matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, mat, title in [
        (axes[0, 0], Phi, "Phi"),
        (axes[0, 1], M, "M"),
        (axes[1, 0], Kinv, "Kinv"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(yhat_vec.reshape(1, -1), aspect="auto", cmap="viridis")
    axes[1, 1].set_title("yhat_vec")
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel("Index")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle("QUBO Inputs Overview", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_qubo_coefficients(Q, q, c):
    """Heatmaps of Q and q with constant c in the title."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    im_Q = axes[0].imshow(Q, aspect="auto", cmap="plasma")
    axes[0].set_title(f"Q matrix (shape {Q.shape})")
    fig.colorbar(im_Q, ax=axes[0], fraction=0.046, pad=0.04)

    im_q = axes[1].imshow(q.reshape(1, -1), aspect="auto", cmap="plasma")
    axes[1].set_title(f"q vector (shape {q.shape})")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Index")
    fig.colorbar(im_q, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"QUBO Coefficients Overview (c = {float(c):.3g})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_convergence(noiseless_conv, noisy_conv, title="QAOA Convergence", save_path=None):
    """Overlay noiseless vs noisy convergence traces."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(noiseless_conv["evals"], noiseless_conv["values"], label="Noiseless", linewidth=0.8)
    ax.plot(noisy_conv["evals"], noisy_conv["values"], label="Noisy", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Objective Value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Noiseless: {len(noiseless_conv['evals'])} evals, "
          f"final = {noiseless_conv['values'][-1]:.4f}")
    print(f"Noisy:     {len(noisy_conv['evals'])} evals, "
          f"final = {noisy_conv['values'][-1]:.4f}")


def plot_approx_ratio_and_hamming(
    samples, noisy_samples, optimal_value, s, save_path=None
):
    """Side-by-side weighted histograms of approximation ratio and Hamming weight.

    Only feasible samples (Hamming weight <= s) are included;
    probabilities are renormalized to condition on feasibility.
    """
    feasible = filter_feasible(samples, s)
    noisy_feasible = filter_feasible(noisy_samples, s)

    fp = sum(samp.probability for samp in feasible)
    nfp = sum(samp.probability for samp in noisy_feasible)

    print(f"Feasibility prob (noiseless): {fp:.4f}, (noisy): {nfp:.4f}")

    probs = [samp.probability / fp for samp in feasible] if fp > 0 else []
    nprobs = [samp.probability / nfp for samp in noisy_feasible] if nfp > 0 else []

    ar = [optimal_value / samp.fval for samp in feasible]
    nar = [optimal_value / samp.fval for samp in noisy_feasible]

    hw = [sum(samp.x) for samp in feasible]
    nhw = [sum(samp.x) for samp in noisy_feasible]

    ratio_bins = np.linspace(0.0, 1.0, 21)
    max_h = max(max(hw, default=0), max(nhw, default=0), s)
    hamming_bins = np.arange(-0.5, max_h + 1.5, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    ax1.hist(ar, bins=ratio_bins, weights=probs, alpha=0.7, edgecolor="black", label="Noiseless")
    ax1.hist(nar, bins=ratio_bins, weights=nprobs, alpha=0.7, edgecolor="black", label="Noisy")
    ax1.set_xlabel("Approximation Ratio", fontsize=10)
    ax1.set_ylabel("Probability", fontsize=10)
    ax1.set_xlim(0, 1.0)
    ax1.set_xticks(np.linspace(0, 1, 6))
    ax1.legend(loc="upper left", fontsize=9)

    ax2.hist(hw, bins=hamming_bins, weights=probs, alpha=0.7, edgecolor="black", label="Noiseless")
    ax2.hist(nhw, bins=hamming_bins, weights=nprobs, alpha=0.7, edgecolor="black", label="Noisy")
    ax2.vlines(s, ymin=0, ymax=ax2.get_ylim()[1], linestyles="dashed",
               linewidth=1, color="black", label=r"$s$")
    ax2.set_xlabel("Hamming Weight", fontsize=10)
    ax2.set_xlim(-0.5, max_h + 0.5)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return fig


def plot_depth_sweep(depths, ar, feas, title="", save_path=None):
    """Dual-panel plot of E[AR] and feasibility probability vs QAOA depth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1.plot(depths, ar, "o-", color="tab:blue", linewidth=2, markersize=8)
    for i, v in enumerate(ar):
        ax1.annotate(f"{v:.3f}", (depths[i], v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    ax1.set_xlabel("QAOA Depth (P)", fontsize=12)
    ax1.set_ylabel("Expected Approximation Ratio", fontsize=12)
    ax1.set_title("Approximation Ratio vs Depth")
    ax1.set_xticks(depths)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.plot(depths, feas, "s-", color="tab:red", linewidth=2, markersize=8)
    for i, v in enumerate(feas):
        ax2.annotate(f"{v:.3f}", (depths[i], v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    ax2.set_xlabel("QAOA Depth (P)", fontsize=12)
    ax2.set_ylabel("Feasibility Probability", fontsize=12)
    ax2.set_title("Feasibility Probability vs Depth")
    ax2.set_xticks(depths)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return fig


# ============================================================
# Depth Sweep & Resource Estimation
# ============================================================

def run_depth_sweep(model, depths, optimal_value, n_original, s,
                    model_name="", initial_state=None, mixer=None):
    """Run noiseless QAOA at each depth, computing conditional E[AR] and P(feasible).

    Returns (expected_ars, feas_probs).
    """
    expected_ars = []
    feas_probs = []

    for p in depths:
        t0 = time.time()

        backend = AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        sampler = SamplerV2(default_shots=10_000)

        qaoa_kwargs = dict(
            sampler=sampler,
            optimizer=SPSA(),
            reps=p,
            initial_point=np.random.uniform(0, 2 * np.pi, 2 * p),
            aggregation=0.25,
            pass_manager=pm,
        )
        if initial_state is not None:
            qaoa_kwargs["initial_state"] = initial_state
        if mixer is not None:
            qaoa_kwargs["mixer"] = mixer

        qaoa = QAOA(**qaoa_kwargs)
        qaoa_opt = MinimumEigenOptimizer(qaoa)
        result = qaoa_opt.solve(model)
        samples = result.samples

        for samp in samples:
            samp.fval = model.objective.evaluate(samp.x)

        feas_prob = sum(
            samp.probability for samp in samples
            if sum(samp.x[:n_original]) <= s
        )

        if feas_prob > 0:
            exp_ar = sum(
                samp.probability * (optimal_value / samp.fval)
                for samp in samples
                if sum(samp.x[:n_original]) <= s and samp.fval > 0
            ) / feas_prob
        else:
            exp_ar = 0.0

        expected_ars.append(exp_ar)
        feas_probs.append(feas_prob)

        elapsed = time.time() - t0
        print(f"  [{model_name}] P={p}: E[AR]={exp_ar:.4f}, "
              f"Feas={feas_prob:.4f} ({elapsed:.1f}s)")

    return expected_ars, feas_probs


def estimate_resources(model, depths, model_name, initial_state=None, mixer=None):
    """Transpiled circuit depth and 2-qubit depth vs QAOA depth.

    Returns (total_depths, two_qubit_depths).
    """
    import qiskit
    from qiskit.circuit.library import QAOAAnsatz

    print(f"\nResource Estimation for {model_name}")
    print(f"{'P':<6} | {'Circuit Depth':<14} | {'2-Qubit Depth':<14}")
    print("-" * 40)

    operator, _offset = model.to_ising()
    depth_list, depth_2q_list = [], []

    for p in depths:
        ansatz = QAOAAnsatz(
            cost_operator=operator, reps=p,
            initial_state=initial_state, mixer_operator=mixer,
        )
        params = [0.1] * ansatz.num_parameters
        qc = ansatz.assign_parameters(params)
        transpiled = qiskit.transpile(qc, basis_gates=["u", "cx"], optimization_level=3)

        d = transpiled.depth()
        d2q = transpiled.depth(filter_function=lambda inst: inst.operation.num_qubits == 2)
        depth_list.append(d)
        depth_2q_list.append(d2q)
        print(f"{p:<6} | {d:<14} | {d2q:<14}")

    return depth_list, depth_2q_list


# ============================================================
# Landscape (statevector simulation, no Qiskit needed at runtime)
# ============================================================

def precompute_costs(Q_mat, q_vec, const, n_qubits):
    """Vectorized QUBO cost for every 2^n bitstring."""
    N = 2**n_qubits
    indices = np.arange(N, dtype=np.int64)
    bits = np.array([((indices >> j) & 1) for j in range(n_qubits)], dtype=np.float64).T
    return np.einsum("bi,ij,bj->b", bits, Q_mat, bits) + bits @ q_vec + const


def apply_rx_all(state, theta, n_qubits):
    """In-place RX(theta) on every qubit of a statevector."""
    c, s_ = np.cos(theta / 2), np.sin(theta / 2)
    for q in range(n_qubits):
        sv = state.reshape(2**(n_qubits - q - 1), 2, 2**q)
        old_0, old_1 = sv[:, 0, :].copy(), sv[:, 1, :].copy()
        sv[:, 0, :] = c * old_0 - 1j * s_ * old_1
        sv[:, 1, :] = -1j * s_ * old_0 + c * old_1
    return state


def apply_xy_gate(state, theta, q0, q1, n_qubits):
    """Combined RYY(theta) @ RXX(theta) on qubits (q0, q1).

    Identity on |00>, |11>; rotation by theta in |01> <-> |10> subspace.
    """
    if q0 > q1:
        q0, q1 = q1, q0
    ct, st = np.cos(theta), np.sin(theta)
    sv = state.reshape(2**(n_qubits - q1 - 1), 2, 2**(q1 - q0 - 1), 2, 2**q0)
    old_01, old_10 = sv[:, 0, :, 1, :].copy(), sv[:, 1, :, 0, :].copy()
    sv[:, 0, :, 1, :] = ct * old_01 - 1j * st * old_10
    sv[:, 1, :, 0, :] = -1j * st * old_01 + ct * old_10
    return state


def landscape_penalty(costs, n_qubits, gamma_range, beta_range):
    """P=1 penalty QAOA landscape: |+>^n initial state, product-RX mixer."""
    N = 2**n_qubits
    init = np.full(N, 1.0 / np.sqrt(N), dtype=complex)
    landscape = np.zeros((len(beta_range), len(gamma_range)))

    for i, bv in enumerate(beta_range):
        for j, gv in enumerate(gamma_range):
            state = init * np.exp(-1j * gv * costs)
            state = apply_rx_all(state, 2 * bv, n_qubits)
            landscape[i, j] = np.abs(state) ** 2 @ costs
    return landscape


def landscape_xy_mixer(costs, n_qubits, s, gamma_range, beta_range):
    """P=1 XY-mixer QAOA landscape: weight-s initial state, ring XY mixer."""
    N = 2**n_qubits
    init_index = (1 << s) - 1
    init = np.zeros(N, dtype=complex)
    init[init_index] = 1.0

    edges = [(idx, (idx + 1) % n_qubits) for idx in range(n_qubits)]
    landscape = np.zeros((len(beta_range), len(gamma_range)))

    for i, bv in enumerate(beta_range):
        for j, gv in enumerate(gamma_range):
            state = init * np.exp(-1j * gv * costs)
            for q0, q1 in edges:
                state = apply_xy_gate(state, 2 * bv, q0, q1, n_qubits)
            landscape[i, j] = np.abs(state) ** 2 @ costs
    return landscape
