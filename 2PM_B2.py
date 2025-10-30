# Two point model for B2
import numpy as np
from scipy.linalg import eig, expm, inv
import matplotlib.pyplot as plt

# Geometry factors
x = [0, 1]  # x-coordinates of the grid points
y = [0, 1]  # y-coordinates of the grid points
R = 0.5  # Major radius of torus in meters
hx = 1/np.linalg.norm(np.gradient(x))
hy = 1/np.linalg.norm(np.gradient(y))
g = hx**2 * hy**2 *(2*np.pi*R)**2
alpha = 10*np.pi/180  # Angle between magnetic field and target plate normal in radians
b = np.sin(alpha)  
b_u = 0.07501  # Upstream magnetic field factor
g_tu = 2.7555 # Geometric factor at target to upstream
L = 12  # Characteristic length in meters
sy = np.sqrt(g) / (hx)  # Cross-sectional area factor



# Plasma parameters
m_i = 1.67e-27  # Mass of ion (kg)
e = 1.6e-19    # Elementary charge (C)
z = 1          # Ion charge state
k_B = 1.38e-23 # Boltzmann constant (J/K)
T_i = 25         # Ion temperature (J)
S_en = 1e6      # Energy source term (J m^-3 s^-1)
S_mom = 1e6     # Momentum source term (kg m^-2 s^-2)

# Equilibrium parameters
T_e0 = 100         # Electron temperature at equilibrium (J)
n_e0 = 1e19         # Electron density at equilibrium (m^-3)
Gamma_e0 = 1e23      # Electron particle flux at equilibrium (m^-2 s^-1)
Q_u0 = 1e6         # Upstream power input at equilibrium (W/m^2)
P_u0 = 1e6        # Upstream particle input at equilibrium (Pa)


# Dynamical coefficients
Tau_Te = 0.1  # Electron temperature confinement time (s)
Tau_ne = 0.2  # Electron density confinement time (s)
Tau_Gammae = 0.15  # Electron particle flux confinement time (s)

# Partial derivatives
K_E = S_en*np.sqrt(g) / (sy * (5.6 * T_e0 + 3 * T_i) * Gamma_e0)
K_M = S_mom*np.sqrt(g) / (sy * np.sqrt(m_i) * Gamma_e0 * np.sqrt(T_e0 + T_i))
dKEdTe =  -5.6*S_en*np.sqrt(g)/(sy*(5.6*T_e0 + 3*T_i)**2 * Gamma_e0)
dKEdGammae =  -S_en*np.sqrt(g)/(sy*(5.6*T_e0 + 3*T_i) * Gamma_e0**2)
dKMdTe = -S_mom*np.sqrt(g)/(sy*np.sqrt(m_i)*Gamma_e0*(T_e0 + T_i)**1.5)
dKMdGammae = -S_mom*np.sqrt(g)/(sy*np.sqrt(m_i)*Gamma_e0**2 * np.sqrt(T_e0 + T_i))

K_T = m_i/(b_u)**2 * (1+T_i/T_e0)*(2 + K_M)**2/((5.6 + 3*T_i/T_e0)**2 * (1+K_E)**2)
K_Gamma = b_u**2/(m_i * g_tu) * (5.6 + 3*T_i/T_e0)*(1 + K_E)/((2 + K_M)**2 * (1+T_i/T_e0))
K_N = b_u**3/(m_i * g_tu * b) * (5.6 + 3*T_i/T_e0)**2 * (1 + K_E)**2/((2 + K_M)**3 * (1+T_i/T_e0)**2)

dKTdTe = m_i/(b_u)**2 * (-T_i/(T_e0**2) * (2 + K_M)**2/((5.6 + 3*T_i/T_e0)**2 * (1+K_E)**2) + (1+T_i/T_e0)*2*(2+K_M)*dKMdTe/((5.6 + 3*T_i/T_e0)**2 * (1+K_E)**2) + \
            (1+T_i/T_e0)*(2 + K_M)**2 * (-2)*(5.6 + 3*T_i/T_e0)**-3 * 3*T_i/T_e0**2/(1+K_E)**2 + \
            (1+T_i/T_e0)*(2 + K_M)**2 * (5.6 + 3*T_i/T_e0)**-2 * (-2) * (1+K_E)**-3 * dKEdTe)
dKTdGammae = m_i * (1+T_i/T_e0)/((b_u)**2 * (5.6 + 3*T_i/T_e0)**2)*(2*(2+K_M)*dKMdGammae + (2+K_M)**2 * (-2)*(1+K_E)**-3 * dKEdGammae)
dKTdne = 0
dKGammadTe = b_u**2/(m_i * g_tu) * (-3*T_i/T_e0**2 *(1 + K_E)/((2 + K_M)**2 * (1+T_i/T_e0)) + (5.6 + 3*T_i/T_e0) * dKEdTe/( (2 + K_M)**2 * (1+T_i/T_e0)) + \
            (5.6 + 3*T_i/T_e0) * (1 + K_E) * (-2)*(2 + K_M)**-3 * dKMdTe/(1+T_i/T_e0) + \
            (5.6 + 3*T_i/T_e0) * (1 + K_E) * ( (2 + K_M)**-2) * T_i/(T_e0+T_i)**2)
dKGammadGammae = b_u**2/(m_i * g_tu) * (5.6 + 3*T_i/T_e0)/(1+T_i/T_e0) * \
                   (dKEdGammae/(2 + K_M)**2 - 2*(1 + K_E)*(2 + K_M)**-3 * dKMdGammae)
dKGammadne = 0


dKNdTe = b_u**3/(m_i * g_tu * b) * (2*(5.6 + 3*T_i/T_e0) * (-3*T_i/T_e0**2) * (1 + K_E)**2/((2 + K_M)**3 * (1+T_i/T_e0)**2) + \
            (5.6 + 3*T_i/T_e0)**2 * 2*(1 + K_E) * dKEdTe/((2 + K_M)**3 * (1+T_i/T_e0)**2) + \
            (5.6 + 3*T_i/T_e0)**2 * (1 + K_E)**2 * (-3)*(2 + K_M)**-4 * dKMdTe/(1+T_i/T_e0)**2 + \
            (5.6 + 3*T_i/T_e0)**2 * (1 + K_E)**2 * (2 + K_M)**-3 * (-2) * (1 + T_i/T_e0)**-3 * (-T_i/T_e0**2))
dKNdGammae = b_u**3/(m_i * g_tu * b) * (5.6 + 3*T_i/T_e0)/(1+T_i/T_e0)**2 * \
                (2*(1 + K_E) * dKEdGammae/((2 + K_M)**3) - 3 * (1 + K_E)**2 * (2 + K_M)**-4 * dKMdGammae)
dKNdne = 0


# Coefficient matrices
A11 = (-1/Tau_Te)*(1 - dKTdTe)
A12 = dKTdGammae/Tau_Te
A13 = dKTdne/Tau_Te
A21 = dKGammadTe/Tau_Gammae
A22 = (-1/Tau_Gammae)*(1 - dKGammadGammae)
A23 = dKGammadne/Tau_Gammae
A31 = dKNdTe/Tau_ne
A32 = dKNdGammae/Tau_ne
A33 = (-1/Tau_ne)*(1 - dKNdne)

B11 = 2*K_T * Q_u0/(P_u0**2 * Tau_Te)
B12 = -2*K_T / (P_u0**3 * Tau_Te)
B21 = -K_Gamma * P_u0**2 / (Q_u0**2 * Tau_Gammae)
B22 = 2*K_Gamma * P_u0 / (Q_u0 * Tau_Gammae)
B31 = -2 * K_N * P_u0**3/(Q_u0**3 * Tau_ne)
B32 = 3 * K_N * P_u0**2 / (Q_u0**2 * Tau_ne)

# Matrix assembly
A = np.array([[A11, A12, A13],
              [A21, A22, A23],
              [A31, A32, A33]])
B = np.array([[B11, B12],
              [B21, B22],
              [B31, B32]])
C = np.eye(3)   # outputs = states (T_e, Gamma_e, n_e) in that order


# 1) eigenvalues & eigenvectors
eigvals, V = eig(A)           # eigvals: complex
W = inv(V)                    # left modal matrix

print("Eigenvalues:")
for i, lam in enumerate(eigvals):
    print(i, lam.real, "+", lam.imag, "j")
    if lam.real < 0:
        print("  time constant tau = {:.4g} s".format(-1.0/lam.real))
    elif lam.real == 0:
        print("  marginal / zero real part")
    else:
        print("  UNSTABLE mode (positive real part)")

# 2) modal input gains (how inputs excite each mode)
B_modal = W.dot(B)   # shape (3,2)
print("\nModal excitation matrix (rows = modes):\n", B_modal)

# 3) plot step response (use explicit fig, axs to avoid creating extra axes)
u_step = np.array([1.0, 0.0])   # e.g. 1-unit step in Q_u, zero in P_u
tfinal = 3.0
nt = 1000
t = np.linspace(0, tfinal, nt)

Ainv = inv(A)
X = np.zeros((A.shape[0], nt), dtype=complex)
for k, tk in enumerate(t):
    X[:, k] = (Ainv.dot(expm(A*tk) - np.eye(A.shape[0])).dot(B)).dot(u_step)

Y = C.dot(X).real     # take real part (numerics)

fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
fig.suptitle("Step response to step in Q_u")

axs[0].plot(t, Y.T[:, 0] / e)
axs[0].set_ylabel("T_e [eV]")
axs[0].grid(True)
axs[0].legend(["T_e"])

axs[1].plot(t, Y.T[:, 1])
axs[1].set_ylabel("Gamma_e [$m^{-2} s^{-1}$]")
axs[1].grid(True)
axs[1].legend(["Gamma_e"])

axs[2].plot(t, Y.T[:, 2])
axs[2].set_ylabel("n_e [$m^{-3}$]")
axs[2].set_xlabel("time [s]")
axs[2].grid(True)
axs[2].legend(["n_e"])

fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
plt.show()
