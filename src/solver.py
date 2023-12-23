import numpy as np
import os, math, random
from typing import List, Dict, Optional
from src.compute import *
import matplotlib.pyplot as plt

class Solver:
    def __init__(
        self,
        nx: int,
        ny: int,
        t_srt: float,
        t_end: float,
        L: float = 1.0,
        is_constraint_slope: bool = True,
        is_animated: bool = False,
        verbose: int = 50,
        save_dir: Optional[str] = "./results/",
    ):
        # grid configuration
        self.nx = nx
        self.ny = ny

        self.L = L
        self.dx = L / nx

        # Setting
        self.is_constraint_slope = is_constraint_slope
        self.courant_factor = 0.64
        self.verbose = verbose
        self.save_dir = save_dir
        self.is_animated = is_animated

        self.t_srt = t_srt
        self.t_end = t_end

        # physical variables
        self.rho = np.zeros((nx, ny), dtype=np.float32)
        self.vx = np.zeros((nx, ny), dtype=np.float32)
        self.vy = np.zeros((nx, ny), dtype=np.float32)
        self.P = np.zeros((nx, ny), dtype=np.float32)
        self.Pt = np.zeros((nx, ny), dtype=np.float32)

        self.Bx = np.zeros((nx, ny), dtype=np.float32)
        self.By = np.zeros((nx, ny), dtype=np.float32)
        self.Ez = np.zeros((nx, ny), dtype=np.float32)

        self.total_energy = np.zeros((nx, ny), dtype=np.float32)
        self.internal_energy = np.zeros((nx, ny), dtype=np.float32)
        self.psi = np.zeros((nx, ny), dtype=np.float32)

        # physical constraint
        self.mu = math.pi * 4 * 10 ** (-7)
        self.gamma = 5 / 3

        self.vA = np.zeros((nx, ny), dtype=np.float32)
        self.vS = np.zeros((nx, ny), dtype=np.float32)

        self.set_init_condition()

    def set_init_condition(self):
        # Orszag-Tang vortex condition

        lin = np.linspace(0.5 * self.dx, self.L - 0.5 * self.dx, self.nx)
        
        Y,X = np.meshgrid(lin, lin)

        self.X = X
        self.Y = Y

        self.vx = (-1) * np.sin(2 * np.pi * Y)
        self.vy = np.sin(2 * np.pi * X)
        self.rho = (self.gamma**2 / (4 * np.pi)) * np.ones_like(X)
        self.P = (self.gamma / (4 * np.pi)) * np.ones_like(X)

        Az = np.cos(4 * np.pi * X) / (4 * np.pi * np.sqrt(4 * np.pi)) + np.cos(2 * np.pi * Y) / (2 * np.pi * np.sqrt(4 * np.pi))
        Bx, By = compute_curl_z(Az, self.dx)

        self.Bx = Bx
        self.By = By

        self.Pt = self.P + 0.5 * (Bx**2 + By**2)
        self.internal_energy = self.P / (self.gamma - 1) / self.rho
        self.total_energy = self.P / (self.gamma - 1) / self.rho + 0.5 * (self.vx**2 + self.vy**2) + 0.5 * (self.Bx**2 + self.By**2) / self.rho

    def compute_Alfven_speed(self, rho, Bx, By):
        self.vA = np.sqrt((Bx**2 + By**2) / (2 * rho))
        return self.vA

    def compute_sound_speed(self, rho, Bx, By, P):
        self.vS = np.sqrt(self.gamma * P / rho)
        return self.vS

    def compute_fast_magnetosonic_speed(self, vS, vA):
        self.vF = np.sqrt((vS**2 + vA**2 + np.sqrt(vS**2 + vA**2) ** 2) / 2)
        return self.vF
    
    def compute_total_energy(self, rho, vx, vy, Bx, By, P):
        self.total_energy = P / (self.gamma - 1) / rho + 0.5 * (vx ** 2 + vy ** 2) + 0.5 * (Bx ** 2 + By ** 2) / rho
        return self.total_energy
    
    def compute_pressure(self, rho, vx, vy, Bx, By, te):
        P = te - 0.5 * (vx ** 2 + vy ** 2) + 0.5 * (Bx ** 2 + By ** 2) / rho
        self.P = P * (self.gamma - 1) * rho
        return self.P

    def compute_RK_partial(
        self,
        rho: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        Bx: np.ndarray,
        By: np.ndarray,
        P: np.ndarray,
    ):
        rho_dx, rho_dy = compute_gradient(rho, self.dx)
        vx_dx, vx_dy = compute_gradient(vx, self.dx)
        vy_dx, vy_dy = compute_gradient(vy, self.dx)
        Bx_dx, Bx_dy = compute_gradient(Bx, self.dx)
        By_dx, By_dy = compute_gradient(By, self.dx)
        P_dx, P_dy = compute_gradient(P, self.dx)

        if self.is_constraint_slope:
            rho_dx, rho_dy = constraint_slope(rho, self.dx, rho_dx, rho_dy)
            vx_dx, vx_dy = constraint_slope(vx, self.dx, vx_dx, vx_dy)
            vy_dx, vy_dy = constraint_slope(vy, self.dx, vy_dx, vy_dy)
            Bx_dx, Bx_dy = constraint_slope(Bx, self.dx, Bx_dx, Bx_dy)
            By_dx, By_dy = constraint_slope(By, self.dx, By_dx, By_dy)
            P_dx, P_dy = constraint_slope(P, self.dx, P_dx, P_dy)

        drho = (-1) * (vx * rho_dx + vy * rho_dy + rho * vx_dx + rho * vy_dy)
        
        # div B used for convergence
        # JxB -> B*dB - d(0.5 * B**2)
        
        dvx = (-1) * (
            vx * vx_dx + vy * vx_dy - 1 / 4 / np.pi / rho * By * (Bx_dy - By_dx) + P_dx / rho
        )
        dvy = (-1) * (
            vx * vy_dx + vy * vy_dy - 1 / 4 / np.pi / rho * Bx * (By_dx - Bx_dy) + P_dy / rho
        )
        dBx = (-1) * (
            vx * By_dy * (-1) + vy * Bx_dy + Bx * (vx_dx + vy_dy) - Bx * vx_dx - By * vx_dy
        )
        dBy = (-1) * (
            vx * By_dx + vy * Bx_dx * (-1) + By * (vx_dx + vy_dy) - Bx * vy_dx - By * vy_dy
        )
        
        dP = (-1) * (
            vx * P_dx + vy * P_dy + self.gamma * P * vx_dx + self.gamma * P * vy_dy
        )

        return drho, dvx, dvy, dBx, dBy, dP
    
    def update_variables(self, rho, vx, vy, Bx, By, P, dt):
        
        # Runge-Kutta method 4th order
        rho_k1, vx_k1, vy_k1, Bx_k1, By_k1, P_k1 = self.compute_RK_partial(
            rho, vx, vy, Bx, By, P
        )

        rho_k2, vx_k2, vy_k2, Bx_k2, By_k2, P_k2 = self.compute_RK_partial(
            rho + 0.5 * dt * rho_k1,
            vx + 0.5 * dt * vx_k1,
            vy + 0.5 * dt * vy_k1,
            Bx + 0.5 * dt * Bx_k1,
            By + 0.5 * dt * By_k1,
            P + 0.5 * dt * P_k1,
        )

        rho_k3, vx_k3, vy_k3, Bx_k3, By_k3, P_k3 = self.compute_RK_partial(
            rho + 0.5 * dt * rho_k2,
            vx + 0.5 * dt * vx_k2,
            vy + 0.5 * dt * vy_k2,
            Bx + 0.5 * dt * Bx_k2,
            By + 0.5 * dt * By_k2,
            P + 0.5 * dt * P_k2,
        )

        rho_k4, vx_k4, vy_k4, Bx_k4, By_k4, P_k4 = self.compute_RK_partial(
            rho + dt * rho_k3,
            vx + dt * vx_k3,
            vy + dt * vy_k3,
            Bx + dt * Bx_k3,
            By + dt * By_k3,
            P + dt * P_k3,
        )

        # update physical variables
        rho_new = rho + dt * (1 / 6) * (rho_k1 + 2 * rho_k2 + 2 * rho_k3 + rho_k4)
        vx_new = vx + dt * (1 / 6) * (vx_k1 + 2 * vx_k2 + 2 * vx_k3 + vx_k4)
        vy_new = vy + dt * (1 / 6) * (vy_k1 + 2 * vy_k2 + 2 * vy_k3 + vy_k4)
        Bx_new = Bx + dt * (1 / 6) * (Bx_k1 + 2 * Bx_k2 + 2 * Bx_k3 + Bx_k4)
        By_new = By + dt * (1 / 6) * (By_k1 + 2 * By_k2 + 2 * By_k3 + By_k4)
        P_new = P + dt * (1 / 6) * (P_k1 + 2 * P_k2 + 2 * P_k3 + P_k4)
        
        return rho_new, vx_new, vy_new, Bx_new, By_new, P_new

    def solve(self):
        t = 0
        count = 0

        # initialize
        print("======================================================================")
        print("# Constraint Transport Solver: Initialize Orszag-Tang vortex condition")
        self.set_init_condition()
        
        print(
            "# Constraint Transport Solver: Iteration for solving MHD tranport equation"
        )
        
        while t < self.t_end:
            
            # compute Alfven speed, sound speed, and fast magnetosonic speed
            vA = self.compute_Alfven_speed(self.rho, self.Bx, self.By)
            vS = self.compute_sound_speed(self.rho, self.Bx, self.By, self.P)
            vF = self.compute_fast_magnetosonic_speed(vS, vA)
            
            # update time interval
            self.dt = self.courant_factor * np.min(
                self.dx / (vF + np.sqrt(self.vx**2 + self.vy**2))
            )
            
            # update physical variables
            rho_new, vx_new, vy_new, Bx_new, By_new, P_new = self.update_variables(self.rho, self.vx, self.vy, self.Bx, self.By, self.P, self.dt)
            
            self.rho = rho_new
            self.vx = vx_new
            self.vy = vy_new
            self.Bx = Bx_new
            self.By = By_new
            self.P = P_new

            # update time
            t += self.dt

            # update macroscopic variables
            self.Pt = self.P + 0.5 * (self.Bx**2 + self.By**2)
            self.internal_energy = self.P / (self.gamma - 1) / self.rho
            self.total_energy = (
                self.internal_energy
                + 0.5 * (self.vx**2 + self.vy**2)
                + 0.5 * (self.Bx**2 + self.By**2) / self.rho
            )

            if count % self.verbose == 0:
                divB = compute_div(self.Bx, self.By, self.dx)
                print(
                    "(Solver) t = {:.3f} | mean divB = {:.3f}".format(
                        t, np.mean(np.abs(divB))
                    )
                )

            # update count
            count += 1

        print("# Constraint Transport Solver: Iteration process complete")
        
        # save figure
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.plot_contourf(
            self.Pt,
            title="Total pressure",
            save_path=os.path.join(self.save_dir, "pressure.png"),
        )
        
        self.plot_contourf(
            self.total_energy,
            title="Total energy",
            save_path=os.path.join(self.save_dir, "energy.png"),
        )
        
        self.plot_contourf(
            self.internal_energy,
            title="Internal energy",
            save_path=os.path.join(self.save_dir, "energy_internal.png"),
        )
        
        self.plot_contourf(
            self.rho,
            title="Plasma density",
            save_path=os.path.join(self.save_dir, "density.png"),
        )
        
        self.plot_contourf(
            self.vx,
            title="Vx",
            save_path=os.path.join(self.save_dir, "vx.png"),
        )
        
        self.plot_contourf(
            self.vy,
            title="Vy",
            save_path=os.path.join(self.save_dir, "vy.png"),
        )
        
        self.plot_contourf(
            0.5*(self.Bx ** 2 + self.By ** 2),
            title="Magnetic pressure",
            save_path=os.path.join(self.save_dir, "pressure_magnetic.png"),
        )

    def plot_contourf(self, vector, title: str, save_path: str):
        cbar = np.linspace(np.min(vector), np.max(vector), num=128)
        plt.figure(figsize=(6, 4), dpi=160)
        plt.contourf(self.X, self.Y, vector, cbar)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)