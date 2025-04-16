import numpy as np

# Slope limit for discontinuity
def constraint_slope(f:np.ndarray, fx:np.ndarray, dx:float, axis:int, direction:int):
    fx_fd = (f - np.roll(f, direction, axis=axis))/dx

    if direction == 1:
        fx = np.maximum(0.0, np.minimum(1.0, fx_fd / (fx + 1.0e-8 * (fx == 0)))) * fx
    elif direction == -1:
        fx = np.maximum(0.0, np.minimum(1.0, (-1) * fx_fd / (fx + 1.0e-8 * (fx == 0)))) * fx
        
    return fx

def slopelimit(f:np.ndarray, fx:np.ndarray, fy:np.ndarray, dx:float):
    R = -1
    L = 1
    f_dx = constraint_slope(f, fx, dx, axis = 0, direction = L)
    f_dx = constraint_slope(f, f_dx, dx, axis = 0, direction = R)
    f_dy = constraint_slope(f, fy, dx, axis = 1, direction = L)
    f_dy = constraint_slope(f, f_dy, dx, axis = 1, direction = R)
    return f_dx, f_dy
