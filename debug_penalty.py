import numpy as np
import scipy.io
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver

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

def pad_zeros(Q, q, s):
    # Pad Q and q with zeros to match the shape of the QUBO matrix
    n = Q.shape[0]
    padded_Q = np.zeros((n + s, n + s))
    padded_Q[:n, :n] = Q

    padded_q = np.zeros(n + s)
    padded_q[:n] = q

    return padded_Q, padded_q

# Generate QUBO matrices
Q, q, c = generate_pde_qubo(M, Kinv, Phi, yhat_vec)

print("=== Original QUBO ===")
print(f"Q shape: {Q.shape}")
print(f"q shape: {q.shape}")
print(f"c: {c}")

# Test penalty method
lam = 10**6
Q_total, q_total, c_total, total_vars = build_qubo(Q, q, c, s, lam)

print("\n=== Penalty Method ===")
print(f"Total variables: {total_vars}")
print(f"Original variables: {q.shape[0]}")
print(f"Slack variables: {total_vars - q.shape[0]}")

# Create penalty QUBO
qubo_penalty = QuadraticProgram("Binary expansion QUBO")
for i in range(total_vars):
    qubo_penalty.binary_var(name=f"x{i}")

qubo_penalty.minimize(
    constant=c_total,
    linear=q_total,
    quadratic=Q_total
)

# Test XY mixer method
padded_Q, padded_q = pad_zeros(Q, q, s)

print("\n=== XY Mixer Method ===")
print(f"Padded Q shape: {padded_Q.shape}")
print(f"Padded q shape: {padded_q.shape}")

bqo = QuadraticProgram("PDE constrained QBO")
binary_vars = [bqo.binary_var(name=f"x{i}") for i in range(padded_Q.shape[0])]

# Add the Hamming weight constraint: sum(x[i]) == s
bqo.linear_constraint(
    linear={f"x{i}": 1 for i in range(padded_Q.shape[0])},
    sense="==",
    rhs=s,
    name="hamming_weight"
)

bqo.minimize(
    constant=c,
    linear=padded_q,
    quadratic=padded_Q
)

# Solve both problems
eigen_solver = NumPyMinimumEigensolver()
optimizer = MinimumEigenOptimizer(eigen_solver)

print("\n=== Solving Penalty Method ===")
result_penalty = optimizer.solve(qubo_penalty)
print("Penalty solution:", result_penalty.prettyprint())

print("\n=== Solving XY Mixer Method ===")
result_xy = optimizer.solve(bqo)
print("XY Mixer solution:", result_xy.prettyprint())

# Check constraint satisfaction for penalty method
n = q.shape[0]
log_s = int(np.ceil(np.log2(s + 1)))

x_penalty = result_penalty.x
u_penalty = x_penalty[:n]  # original variables
v_penalty = x_penalty[n:n+log_s]  # slack variables

# Compute e^T u + a^T v
sum_u = np.sum(u_penalty)
weights = np.array([2**j for j in range(log_s)])
sum_v = np.dot(weights, v_penalty)
total_penalty = sum_u + sum_v

print(f"\n=== Constraint Check for Penalty Method ===")
print(f"Original variables sum: {sum_u}")
print(f"Slack variables weighted sum: {sum_v}")
print(f"Total: {total_penalty}")
print(f"Target: {s}")
print(f"Constraint satisfied: {abs(total_penalty - s) < 1e-6}")

# Check constraint satisfaction for XY mixer method
x_xy = result_xy.x
total_xy = np.sum(x_xy)

print(f"\n=== Constraint Check for XY Mixer Method ===")
print(f"Total Hamming weight: {total_xy}")
print(f"Target: {s}")
print(f"Constraint satisfied: {abs(total_xy - s) < 1e-6}")

# Analyze the penalty formulation
print(f"\n=== Penalty Formulation Analysis ===")
print(f"Penalty parameter λ: {lam}")
print(f"Penalty term: λ * (e^T u + a^T v - s)^2")

# Check if the penalty is working correctly
t = np.concatenate([np.ones(n), np.array([2**j for j in range(log_s)])])
constraint_violation = np.dot(t, x_penalty) - s
penalty_value = lam * constraint_violation**2

print(f"Constraint violation: {constraint_violation}")
print(f"Penalty value: {penalty_value}")
print(f"Objective value: {result_penalty.fval}")

# Check if increasing lambda helps
print(f"\n=== Testing Different Lambda Values ===")
for lam_test in [10**4, 10**5, 10**6, 10**7]:
    Q_test, q_test, c_test, _ = build_qubo(Q, q, c, s, lam_test)
    
    qubo_test = QuadraticProgram("Test QUBO")
    for i in range(total_vars):
        qubo_test.binary_var(name=f"x{i}")
    
    qubo_test.minimize(
        constant=c_test,
        linear=q_test,
        quadratic=Q_test
    )
    
    result_test = optimizer.solve(qubo_test)
    x_test = result_test.x
    u_test = x_test[:n]
    v_test = x_test[n:n+log_s]
    
    sum_u_test = np.sum(u_test)
    sum_v_test = np.dot(weights, v_test)
    total_test = sum_u_test + sum_v_test
    
    print(f"λ = {lam_test}: constraint violation = {abs(total_test - s):.6f}") 