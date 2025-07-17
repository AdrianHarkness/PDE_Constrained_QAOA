import numpy as np
import scipy.io
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

# Load MATLAB file
data = scipy.io.loadmat('ifiss3.6/QUBO_4.mat')

# Extract matrices as NumPy arrays
Phi = np.array(data['Phi'])
M = np.array(data['M'].toarray())
Kinv = np.array(data['Kinv'])
yhat_vec = np.array(data['yhat_vec']).squeeze()  # for 1D array

# Hamming weight upper bound
s = 2

def generate_pde_qubo(M, Kinv, Phi, yd):
    Q = np.array((1/2) * Phi.T @ M @ Kinv @ M @ Kinv @ M @ Phi)
    q = np.array((1/2) * (yd.T @ M @ Kinv @ M @ Phi).T - (Phi.T @ M @ Kinv.T @ M @ yd))
    c = np.array((1/2) * yd.T @ M @ yd)
    return Q, q, c

def build_qubo(Q, q, c, s, lam):
    """
    Build the QUBO matrix and vector for the binary expansion slack formulation.
    """
    n = q.shape[0]
    log_s = int(np.ceil(np.log2(s + 1)))  # number of slack variables
    
    total_vars = n + log_s

    # Step 1: Pad original Q and q to size (n + log_s)
    padded_Q = np.zeros((total_vars, total_vars))
    padded_Q[:n, :n] = Q  # original Q in top-left block

    padded_q = np.zeros(total_vars)
    padded_q[:n] = q  # original q

    # Step 2: Build the penalty matrices
    # e = vector of ones for original variables u
    e = np.ones(n)
    # a = binary weights for slack variables v
    a = np.array([2**j for j in range(log_s)])

    # Stack e and a to form t = (e, a)
    t = np.concatenate([e, a])

    # Compute hat_Q = t * t^T
    hat_Q = np.outer(t, t)

    # Compute hat_q = -2 * s * t
    hat_q = -2 * s * t

    # Compute penalty constant term
    penalty_const = s**2

    # Step 3: Add penalty terms to padded Q and q
    Q_total = padded_Q + lam * hat_Q
    q_total = padded_q + lam * hat_q
    c_total = c + lam * penalty_const

    return Q_total, q_total, c_total, total_vars

# Generate QUBO matrices
Q, q, c = generate_pde_qubo(M, Kinv, Phi, yhat_vec)

# Build penalty QUBO
lam = 10**6
Q_total, q_total, c_total, total_vars = build_qubo(Q, q, c, s, lam)

# Create penalty QUBO
qubo = QuadraticProgram("Binary expansion QUBO")
for i in range(total_vars):
    qubo.binary_var(name=f"x{i}")

qubo.minimize(
    constant=c_total,
    linear=q_total,
    quadratic=Q_total
)

# Set up QAOA
backend = AerSimulator()
sampler = Sampler()
cvar_alpha = 1
shots = 1024/cvar_alpha
sampler.set_options(shots=shots, backend=backend)
optimizer = COBYLA()
reps = 3
initial_point = np.zeros(2 * reps)

qaoa = QAOA(
    sampler=sampler, 
    optimizer=optimizer, 
    reps=reps, 
    initial_point=initial_point,
    aggregation=cvar_alpha
)

qaoa_optimizer = MinimumEigenOptimizer(qaoa)

# Solve with QAOA
print("Running QAOA...")
qaoa_result = qaoa_optimizer.solve(qubo)

print("QAOA completed!")
print(f"Number of samples: {len(qaoa_result.samples)}")

# Check constraint satisfaction for each sample
n = q.shape[0]  # number of original variables
log_s = int(np.ceil(np.log2(s + 1)))  # number of slack variables

print(f"\n=== Constraint Analysis ===")
print(f"Original variables: {n}")
print(f"Slack variables: {log_s}")
print(f"Total variables: {total_vars}")
print(f"Target Hamming weight: {s}")

constraint_violations = []
valid_solutions = []

for i, sample in enumerate(qaoa_result.samples):
    x = sample.x
    u = x[:n]                      # original variables
    v = x[n:n+log_s]               # slack variables

    # Compute e^T u + a^T v
    sum_u = np.sum(u)
    weights = np.array([2**j for j in range(log_s)])
    sum_v = np.dot(weights, v)
    total = sum_u + sum_v

    constraint_violation = abs(total - s)
    constraint_violations.append(constraint_violation)
    
    if constraint_violation < 1e-6:
        valid_solutions.append(i)
    
    print(f"Sample {i}: u={u}, v={v}, sum_u={sum_u}, sum_v={sum_v}, total={total}, violation={constraint_violation}")

print(f"\n=== Summary ===")
print(f"Total samples: {len(qaoa_result.samples)}")
print(f"Valid solutions: {len(valid_solutions)}")
print(f"Constraint violations: {len(constraint_violations) - len(valid_solutions)}")
print(f"Average constraint violation: {np.mean(constraint_violations):.6f}")
print(f"Max constraint violation: {np.max(constraint_violations):.6f}")

# Check if the issue is with the binary expansion encoding
print(f"\n=== Binary Expansion Analysis ===")
print(f"log_s = {log_s}")
print(f"Binary weights: {[2**j for j in range(log_s)]}")
print(f"Maximum representable value: {sum([2**j for j in range(log_s)])}")

# Test if s=2 can be represented with log_s=2 slack variables
max_representable = sum([2**j for j in range(log_s)])
print(f"Can represent s={s} with {log_s} slack variables: {s <= max_representable}")

# Check the actual values in the slack variables
print(f"\n=== Slack Variable Analysis ===")
for i, sample in enumerate(qaoa_result.samples[:5]):  # Check first 5 samples
    x = sample.x
    v = x[n:n+log_s]
    weights = np.array([2**j for j in range(log_s)])
    sum_v = np.dot(weights, v)
    print(f"Sample {i}: slack variables {v}, weighted sum {sum_v}") 