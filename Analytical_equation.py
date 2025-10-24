# plotting different steady state equations for plasma edge analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
L = 12  # characteristic length
Te_u = 100  # upstream electron temperature in eV
Ti_u = 100  # upstream ion temperature in eV, higher than Te_u?   
ne_u = 1e19  # upstream electron density in m^-3
mi = 1.67e-27  # ion mass in kg (proton mass)
e = 1.6e-19  # elementary charge in C  
kB = 1.38e-23  # Boltzmann constant in J/K
kappa_0e = 2000   # electron parallel conductivity 
kappa_0i = 600   # ion parallel conductivity

#  2PM
'Control parameters: ne_u and q_parallel'
q_parallel = 1e6  # parallel heat flux in W/m^2
def two_point_model(ne_u, q_parallel, L, kappa_0e, mi, e,gamma=7):
    Te_t = mi/(2*e)*((4*q_parallel**2)*(7/2*q_parallel*L/kappa_0e)**(-4/7))/(gamma**2*ne_u**2*e**2)     #p242-243 in Stangeby
    ne_t = ne_u**3/q_parallel**2 * (7/2*q_parallel*L/kappa_0e)**(6/7)*gamma**2*e**3/(4*mi)
    gamma_t = ne_u**2/q_parallel * (7/2*q_parallel*L/kappa_0e)**(4/7)*gamma*e**2/(2*mi)
    return Te_t, ne_t, gamma_t

Te_t, ne_t, gamma_t = two_point_model(ne_u, q_parallel, L, kappa_0e, mi, e)

# Modelling of conduction limited regime
'Pure conduction and boundary condition at target'
def conduction_limited(Te_t, Ti_t, kappa_0e, mi, kB, gamma_e=5, gamma_i=2.5):
    c_st = (kB*(Te_t + Ti_t)/2*mi)**(1/2)
    qe_t = gamma_e*kB*ne_t*Te_t*c_st
    qi_t = gamma_i*kB*ne_t*Te_t*c_st
    def Te(s):
        return (Te_t**(7/2) + 7/2*qe_t/kappa_0e*s)**(2/7)
    def Ti(s):
        return (Ti_t**(7/2) + 7/2*qi_t/kappa_0i*s)**(2/7)

    return Te, Ti

# Onion skin method


# Plotting results
    s = np.linspace(0, 0.1, L)
    Te_profile = Te(s)
    Ti_profile = Ti(s)
    plt.figure(figsize=(10, 6))
    plt.plot(s, Te_profile, label='Electron Temperature (Te)', color='blue')
    plt.plot(s, Ti_profile, label='Ion Temperature (Ti)', color='red')
    plt.xlabel('Distance (m)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Profiles')
    plt.legend()
    plt.grid()
    plt.show()  