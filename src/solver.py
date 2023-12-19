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

        xlin = np.linspace(0.5 * self.dx, self.L - 0.5 * self.dx, self.nx)
        ylin = np.linspace(0.5 * self.dx, self.L - 0.5 * self.dx, self.ny)
        
        X,Y = np.meshgrid(xlin, ylin)

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
        dvx = (-1) * (
            vx * vx_dx + vy * vx_dy - 1 / 4 / np.pi * By * (Bx_dy - By_dx) + P_dx
        )
        dvy = (-1) * (
            vx * vy_dx + vy * vy_dy - 1 / 4 / np.pi * By * (By_dx - Bx_dy) + P_dy
        )
        dBx = (-1) * (
            vx * Bx_dx + vy * Bx_dy + Bx * (vx_dx + vy_dy) - Bx * vx_dx - By * vx_dy
        )
        dBy = (-1) * (
            vx * By_dx + vy * By_dy + By * (vx_dx + vy_dy) - Bx * vy_dx - By * vy_dy
        )
        dP = (-1) * (
            vx * P_dx + vy * P_dy + self.gamma * P * vx_dx + self.gamma * P * vy_dy
        )

        return drho, dvx, dvy, dBx, dBy, dP

    def solve(self):
        t = 0
        count = 0

        # initialize
        print("======================================================================")
        print("# Constraint Transport Solver: Initialize Orszag-Tang vortex condition")
        self.set_init_condition()

        vA = self.compute_Alfven_speed(self.rho, self.Bx, self.By)
        vS = self.compute_sound_speed(self.rho, self.Bx, self.By, self.P)
        vF = self.compute_fast_magnetosonic_speed(vS, vA)

        self.dt = self.courant_factor * np.min(
            self.dx / (vF + np.sqrt(self.vx**2 + self.vy**2))
        )
        
        print(
            "# Constraint Transport Solver: Iteration for solving MHD tranport equation"
        )
        while t < self.t_end:
            # Runge-Kutta method 4th order
            rho_k1, vx_k1, vy_k1, Bx_k1, By_k1, P_k1 = self.compute_RK_partial(
                self.rho, self.vx, self.vy, self.Bx, self.By, self.P
            )

            rho_k2, vx_k2, vy_k2, Bx_k2, By_k2, P_k2 = self.compute_RK_partial(
                self.rho + 0.5 * self.dt * rho_k1,
                self.vx + 0.5 * self.dt * vx_k1,
                self.vy + 0.5 * self.dt * vy_k1,
                self.Bx + 0.5 * self.dt * Bx_k1,
                self.By + 0.5 * self.dt * By_k1,
                self.P + 0.5 * self.dt * P_k1,
            )

            rho_k3, vx_k3, vy_k3, Bx_k3, By_k3, P_k3 = self.compute_RK_partial(
                self.rho + 0.5 * self.dt * rho_k2,
                self.vx + 0.5 * self.dt * vx_k2,
                self.vy + 0.5 * self.dt * vy_k2,
                self.Bx + 0.5 * self.dt * Bx_k2,
                self.By + 0.5 * self.dt * By_k2,
                self.P + 0.5 * self.dt * P_k2,
            )

            rho_k4, vx_k4, vy_k4, Bx_k4, By_k4, P_k4 = self.compute_RK_partial(
                self.rho + self.dt * rho_k3,
                self.vx + self.dt * vx_k3,
                self.vy + self.dt * vy_k3,
                self.Bx + self.dt * Bx_k3,
                self.By + self.dt * By_k3,
                self.P + self.dt * P_k3,
            )

            # update physical variables
            self.rho += self.dt * (1 / 6) * (rho_k1 + 2 * rho_k2 + 2 * rho_k3 + rho_k4)
            self.vx += self.dt * (1 / 6) * (vx_k1 + 2 * vx_k2 + 2 * vx_k3 + vx_k4)
            self.vy += self.dt * (1 / 6) * (vy_k1 + 2 * vy_k2 + 2 * vy_k3 + vy_k4)
            self.Bx += self.dt * (1 / 6) * (Bx_k1 + 2 * Bx_k2 + 2 * Bx_k3 + Bx_k4)
            self.By += self.dt * (1 / 6) * (By_k1 + 2 * By_k2 + 2 * By_k3 + By_k4)
            self.P += self.dt * (1 / 6) * (P_k1 + 2 * P_k2 + 2 * P_k3 + P_k4)

            # update time
            t += self.dt

            # update time interval
            self.dt = self.courant_factor * np.min(
                self.dx / (vF + np.sqrt(self.vx**2 + self.vy**2))
            )

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

    def plot_contourf(self, vector, title: str, save_path: str):
        cbar = np.linspace(np.min(vector), np.max(vector), num=64)
        plt.figure(figsize=(6, 4), dpi=160)
        plt.contourf(self.X, self.Y, vector, cbar)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
