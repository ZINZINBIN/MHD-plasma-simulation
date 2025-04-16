import numpy as np

def compute_gradient(A: np.ndarray, dx: float):
    Ax = (np.roll(A, -1, 0) - np.roll(A, 1, 0)) / (2 * dx)
    Ay = (np.roll(A, -1, 1) - np.roll(A, 1, 1)) / (2 * dx)
    return Ax, Ay

def compute_avg_field(Ax:np.ndarray, Ay:np.ndarray):	
    Ax_avg = 0.5 * (Ax+np.roll(Ax,1,axis=0)) 
    Ay_avg = 0.5 * (Ay+np.roll(Ay,1,axis=1)) 
    return Ax_avg, Ay_avg

def compute_curl_z(A: np.ndarray, dx:float):
    Bx = (A - np.roll(A, 1, axis = 1)) / dx
    By = (-1) * (A - np.roll(A, 1, axis = 0)) / dx
    return Bx, By

def compute_div(Ax: np.ndarray, Ay: np.ndarray, dx: float):
    R = -1
    L = 1
    div_A = np.roll(Ax, R, axis = 0) - np.roll(Ax, L, axis = 0) + np.roll(Ay, R, axis = 1) - np.roll(Ay, L, axis = 1)
    div_A /= 2 * dx
    return div_A

def compute_conserved_field(field:np.ndarray, flux_x:np.ndarray, flux_y:np.ndarray, dx:float, dt:float):
    flux_x_sum = flux_x - np.roll(flux_x, 1, axis = 0)
    flux_y_sum = flux_y - np.roll(flux_y, 1, axis = 1)
    flux_sum = flux_x_sum + flux_y_sum
    flux_sum /= dx
    return field - dt * flux_sum
    
def extrapolate_space(A: np.ndarray, dx:float, Ax:np.ndarray, Ay:np.ndarray):
    # Second order extrapolation in space
    Ax_l = A - Ax * dx / 2
    Ax_r = A + Ax * dx / 2

    Ay_l = A - Ay * dx / 2
    Ay_r = A + Ay * dx / 2

    Ax_l = np.roll(Ax_l, -1, axis = 0)
    Ay_l = np.roll(Ay_l, -1, axis = 1)

    return Ax_l, Ax_r, Ay_l, Ay_r