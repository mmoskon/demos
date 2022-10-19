# models as used in https://doi.org/10.1016/j.compbiomed.2020.104109 
# this code is adapted version of the code available at https://github.com/mmoskon/CBLBs/

import numpy as np

# a model of inverter
def not_cell(state, params):
    L_X, x, y = state
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x = params

    f = gamma_L_X * (y ** n_y)/(1 + (theta_L_X*y)**n_y )
    dL_X_dt = (f - delta_L * L_X)

    dx_dt = (eta_x * (1/(1+ (omega_x*L_X)**m_x))) - (delta_x * x) - rho_x * x # rho ... increased degradation rate

    return dL_X_dt, dx_dt

# a model of driver
def yes_cell(state, params):
    x, y = state
    gamma_x, n_y, theta_x, delta_x, rho_x = params

    dx_dt = gamma_x * (y ** n_y)/(1 + (theta_x*y)**n_y ) - (delta_x * x) - rho_x * x # rho ... increased degradation rate
    
    return dx_dt

# L_A ... intermediate
# a ... out
# b ... in
def not_cell_wrapper(state, params):
    L_A, a, b = state
    
    state_A = L_A, a, b
    params_A = params

    return not_cell(state_A, params_A)

# a ... out
# b ... in
def yes_cell_wrapper(state, params):
    a, b = state

    state_A = a, b
    params_A = params

    return yes_cell(state_A, params_A)

def not_model(T, state, params):
    L_A, a, b = state

    delta_L, gamma_L_A, n_b, theta_L_A, eta_a, omega_a, m_a, delta_a, delta_b, rho_a, rho_b = params

    state_not = L_A, a, b
    params_not = delta_L, gamma_L_A, n_b, theta_L_A, eta_a, omega_a, m_a, delta_a, rho_a
    dL_A_dt, da_dt = not_cell_wrapper(state_not, params_not)
    
    db_dt = 0

    return np.array([dL_A_dt, da_dt, db_dt])

def yes_model(T, state, params):
    a, b = state
    
    gamma_a, n_b, theta_a, delta_a, delta_b, rho_a, rho_b = params

    state_yes = a, b
    params_yes = gamma_a, n_b, theta_a, delta_a, rho_a
    da_dt = yes_cell_wrapper(state_yes, params_yes)
    db_dt = 0 

    return np.array([da_dt, db_dt])

