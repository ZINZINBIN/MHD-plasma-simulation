import numpy as np
from typing import Optional

def compute_gradient(A: np.ndarray, d: float):
    Ax = (np.roll(A, -1, 0) - np.roll(A, 1, 0)) / (2 * d)
    Ay = (np.roll(A, -1, 1) - np.roll(A, 1, 1)) / (2 * d)

    Ax[0, :] = (np.roll(A, -1, 0) - A)[0,:] / d # (A - np.roll(A, 1, 0))[0,:] / d
    Ax[-1, :] = (A - np.roll(A, 1, 0))[-1,:] / d # (np.roll(A, -1, 0) - A)[-1,:] / d

    Ay[:, 0] = (np.roll(A, -1, 1) - A)[:,0] / d # (A - np.roll(A, 1, 1))[:,0] / d
    Ay[:, -1] = (A - np.roll(A, 1, 1))[:,-1] / d # (np.roll(A, -1, 1) - A)[:,-1] / d
    
    return Ax, Ay

def compute_avg_field(Ax:np.ndarray, Ay:np.ndarray):	
	Ax_avg = 0.5 * (Ax+np.roll(Ax,1,axis=0)) 
	Ay_avg = 0.5 * (Ay+np.roll(Ay,1,axis=1)) 
	return Ax_avg, Ay_avg

def compute_curl_z(A: np.ndarray, d: float):
    Bx = (np.roll(A, -1, axis = 1) - np.roll(A, 1, axis = 1)) / (2 * d)
    By = (-1) * (np.roll(A, -1, axis = 0) - np.roll(A, 1, axis = 0)) / (2 * d)

    Bx[:,0] = (np.roll(A, -1, axis = 1) - A)[:,0] / d # (A - np.roll(A, 1, 1))[0,:] / d
    Bx[:,-1] = (A - np.roll(A, 1, axis = 1))[:,-1] / d # (np.roll(A, -1, 1) - A)[-1,:] / d

    By[0,:] = (-1) * (np.roll(A, -1, axis = 0) - A)[0,:] / d # (-1) * (A - np.roll(A, 1, 0))[:,0] / d
    By[-1,:] = (-1) * (A - np.roll(A, 1, axis = 0))[-1,:] / d # (-1) * (np.roll(A, -1, 0) - A)[:,-1] / d
    
    return Bx, By

def compute_div(Ax: np.ndarray, Ay: np.ndarray, d: float):
    R = -1
    L = 1
    div_A = np.roll(Ax, R, axis = 0) - np.roll(Ax, L, axis = 0) + np.roll(Ay, R, axis = 1) - np.roll(Ay, L, axis = 1)
    div_A /= 2 * d
    return div_A

def extrapolate_space(
    A: np.ndarray,
    d: float,
    Ax: Optional[np.ndarray] = None,
    Ay: Optional[np.ndarray] = None,
):
    if Ax is None:
        Ax, Ay = compute_gradient(A, d)

    Ax_l = A - Ax * d / 2
    Ax_r = A + Ax * d / 2

    Ay_l = A - Ay * d / 2
    Ay_r = A + Ay * d / 2

    Ax_l = np.roll(Ax_l, -1, 0)
    Ay_l = np.roll(Ay_l, -1, 1)

    return Ax_l, Ax_r, Ay_l, Ay_r

def constraint_slope(A: np.ndarray, d: float, Ax: np.ndarray, Ay: np.ndarray):
    R = -1
    L = 1
    Ax = (
        np.maximum(
            0,
            np.minimum(
                1, ((A - np.roll(A, L, axis=0)) / d) / (Ax + 1.0e-8 * (Ax == 0))
            ),
        )
        * Ax
    )
    Ax = (
        np.maximum(
            0,
            np.minimum(
                1, (-1) * ((A - np.roll(A, R, axis=0)) / d) / (Ax + 1.0e-8 * (Ax == 0))
            ),
        )
        * Ax
    )
    Ay = (
        np.maximum(
            0,
            np.minimum(
                1, ((A - np.roll(A, L, axis=1)) / d) / (Ay + 1.0e-8 * (Ay == 0))
            ),
        )
        * Ay
    )
    Ay = (
        np.maximum(
            0,
            np.minimum(
                1, (-1) * ((A - np.roll(A, R, axis=1)) / d) / (Ay + 1.0e-8 * (Ay == 0))
            ),
        )
        * Ay
    )

    return Ax, Ay
