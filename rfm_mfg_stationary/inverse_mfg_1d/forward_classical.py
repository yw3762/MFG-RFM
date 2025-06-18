import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Parameters
N = 1000
Lx = 1.0
dx = Lx / N
x = np.linspace(0, Lx, N, endpoint=False)
eps = 0.3
alpha = 0.3
b = 0.1 * (np.sin(2 * np.pi * x - np.sin(4 * np.pi * x)) + np.exp(np.cos(2 * np.pi * x)))
max_iter = 10

# Spectral Laplacian (most accurate for periodic BCs)
def laplacian(u, dx):
    N = len(u)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    u_hat = np.fft.fft(u)
    lap_u_hat = -(k**2) * u_hat
    return np.fft.ifft(lap_u_hat).real

# Spectral divergence (centered, adjoint to gradient)
def divergence(f, dx):
    N = len(f)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    f_hat = np.fft.fft(f)
    div_f_hat = 1j * k * f_hat
    return -np.fft.ifft(div_f_hat).real

# Centered gradient
def gradient(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

# Initialization
q = 0.1 * np.random.randn(N)
m_history = []
u_history = []

for k in range(max_iter):
    m_prev = m_history[-1] if k > 0 else np.ones(N)
    rhs_m = divergence(m_prev * q, dx)

    # Solve -eps * Lap(m) = rhs_m in Fourier space
    k_freq = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    rhs_m_hat = np.fft.fft(rhs_m)
    lap_operator = -(k_freq**2)
    lap_operator[0] = 1e-10  # avoid divide-by-zero at k=0
    m_hat = rhs_m_hat / (-eps * lap_operator)
    m_hat[0] = 0  # enforce zero-mean
    m = np.fft.ifft(m_hat).real
    m /= np.sum(m) * dx  # normalize
    m_history.append(m)

    Lq = 0.5 * q**2
    Fm = m**2
    RHS_u = b + Fm + Lq

    def A_u(u):
        return -eps * laplacian(u, dx) + q * gradient(u, dx)

    def solve_augmented(RHS):
        from scipy.optimize import minimize

        def loss(u):
            u = u - np.mean(u)
            res = A_u(u) - RHS
            return 0.5 * np.sum(res**2)

        u0 = np.zeros(N)
        result = minimize(loss, u0, method='L-BFGS-B')
        u = result.x - np.mean(result.x)
        return u

    u = solve_augmented(RHS_u)
    u_history.append(u)

    q_new = gradient(u, dx)
    q = (1 - alpha) * q + alpha * q_new

    print(f"Iteration {k+1}, ||q|| = {np.linalg.norm(q):.5f}")

    plt.figure(figsize=(10, 3))
    plt.plot(x, u, label=f'u (iter {k+1})')
    plt.legend()
    plt.xlabel('x')
    plt.title(f'u at Iteration {k+1}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.plot(x, m, label=f'm (iter {k+1})')
    plt.legend()
    plt.xlabel('x')
    plt.title(f'm at Iteration {k+1}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 3))
plt.plot(x, u_history[-1], label='u (final)')
plt.legend()
plt.xlabel('x')
plt.title('Final u after Iteration')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(x, m_history[-1], label='m (final)')
plt.legend()
plt.xlabel('x')
plt.title('Final m after Iteration')
plt.grid(True)
plt.show()
