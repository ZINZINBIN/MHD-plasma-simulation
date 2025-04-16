import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional
from src.compute import (
    compute_gradient, 
    compute_avg_field, 
    compute_conserved_field, 
    extrapolate_space, 
    compute_curl_z, 
    compute_div
)
from src.slope import slopelimit
from src.util import plot_contourf, generate_contourf_gif

class Simulation:
    def __init__(
        self,
        nx: int,
        ny: int,
        t_end: float,
        L: float = 1.0,
        slopelimit: bool = True,
        animation: bool = False,
        verbose: int = 50,
        savedir: Optional[str] = "./results/",
        plot_freq: Optional[int] = 16,
        courant_factor: float = 0.5,
    ):
        # Grid configuration
        self.nx = nx
        self.ny = ny
        self.L = L
        self.dx = L / nx

        # Time integration
        self.t_end = t_end
        self.dt_min = 0.01

        # Computational stability
        self.courant_factor = courant_factor
        self.slopelimit = slopelimit

        # Figure setting
        self.verbose = verbose
        self.savedir = savedir
        self.animation = animation
        self.plot_freq = plot_freq

        # Physical variables
        self.rho = np.zeros((nx, ny), dtype=np.float32)
        self.vx = np.zeros((nx, ny), dtype=np.float32)
        self.vy = np.zeros((nx, ny), dtype=np.float32)
        self.P = np.zeros((nx, ny), dtype=np.float32)
        self.Bx = np.zeros((nx, ny), dtype=np.float32)
        self.By = np.zeros((nx, ny), dtype=np.float32)

        self.Ez = np.zeros((nx, ny), dtype=np.float32)
        self.Bxh = np.zeros((nx, ny), dtype=np.float32)
        self.Byh = np.zeros((nx, ny), dtype=np.float32)

        # Conserved quantities
        self.m = np.zeros((nx, ny), dtype=np.float32)
        self.px = np.zeros((nx, ny), dtype=np.float32)
        self.py = np.zeros((nx, ny), dtype=np.float32)
        self.en = np.zeros((nx, ny), dtype=np.float32)
        self.J = np.zeros((nx, ny), dtype=np.float32)
        self.w = np.zeros((nx, ny), dtype=np.float32)
        self.Pm = np.zeros((nx, ny), dtype = np.float32)

        self.gamma = 5/3

        # Trajectory
        self.Et = []
        self.KEt = []
        self.MEt = []
        self.dEt = []
        self.ts = []
        self.divB = []

        # Record of MHD density distribution
        self.record = []

        self.set_init_condition()

    def set_init_condition(self):

        # Orszag-Tang vortex condition
        # B0 = 1 / np.sqrt(4 * np.pi), but we use cgs unit thus we consider B0 => 1.0
        lin = np.linspace(0.5 * self.dx, self.L - 0.5 * self.dx, self.nx)
        Y,X = np.meshgrid(lin, lin)

        self.X = X
        self.Y = Y

        self.vx = (-1) * np.sin(2 * np.pi * Y / self.L)
        self.vy = np.sin(2 * np.pi * X / self.L)
        self.rho = (self.gamma**2 / (4 * np.pi)) * np.ones_like(X)

        # B0 = 1 / np.sqrt(4 * np.pi)
        Az = np.cos(4 * np.pi * X / self.L) / (4 * np.pi) + np.cos(2 * np.pi * Y / self.L) / (2 * np.pi)
        Bxh, Byh = compute_curl_z(Az, self.dx)

        self.Bxh = Bxh
        self.Byh = Byh

        Bx, By = compute_avg_field(Bxh, Byh)

        self.P = (self.gamma / (4 * np.pi)) * np.ones_like(X)
        self.Bx = Bx
        self.By = By

        en = self.P / (self.gamma - 1) + 0.5 * (self.vx**2 + self.vy**2) * self.rho + 1 / 8 / np.pi * (self.Bx**2 + self.By**2)

        self.en = en
        self.m = self.rho
        self.px = self.rho * self.vx
        self.py = self.rho * self.vy
        self.w = self.compute_vorticity(vx = self.vx, vy = self.vy)
        self.J = self.compute_current(Bx = self.Bx, By = self.By)
        self.Pm = self.compute_magnetic_energy(Bx = self.Bx, By = self.By)

    def compute_Alfven_speed(self, rho:float, Bx:float, By:float):
        vA = np.sqrt((Bx**2 + By**2) / rho / np.pi / 4)
        return vA

    def compute_sound_speed(self, rho:float, P:float, gamma:float):
        vS = np.sqrt(gamma * P / rho)
        return vS

    def compute_fast_magnetosonic_speed(self, vS:float, vA:float):
        vF = np.sqrt((vS**2 + vA**2 + np.sqrt(vS**2 + vA**2) ** 2) / 2)
        return vF

    def compute_total_energy(self, rho:float, vx:float, vy:float, Bx:float, By:float, P:float, gamma:float):
        E_total = P/(gamma - 1) + 0.5 * (vx ** 2 + vy ** 2) * rho + 1 / np.pi / 8 * (Bx ** 2 + By ** 2)
        return E_total

    def compute_kinetic_energy(self, rho:float, vx:float, vy:float, P:float, gamma:float):
        KE = 0.5 * (vx**2 + vy**2) * rho + P / (gamma - 1)
        return KE

    def compute_magnetic_energy(self, Bx, By):
        ME = 1 / np.pi / 8 * (Bx**2 + By**2)
        return ME

    def compute_dissipation_energy(self, w:float, j:float, vis_kinetic:float, vis_magnetic:float):
        dE = vis_kinetic * w ** 2 + vis_magnetic * j ** 2
        return dE

    def compute_vorticity(self, vx:float, vy:float):
        _, vx_dy = compute_gradient(vx, self.dx)
        vy_dx, _ = compute_gradient(vy, self.dx)
        w = vy_dx - vx_dy 
        return w

    def compute_current(self, Bx:float, By:float):
        _, Bx_dy = compute_gradient(Bx, self.dx)
        By_dx, _ = compute_gradient(By, self.dx)
        Jz = (Bx_dy - By_dx) / 4 / np.pi
        return Jz

    def compute_magnetic_pressure(self, Bx:float, By:float):
        Pm = 1 / 8 / np.pi * (Bx ** 2 + By ** 2)
        return Pm

    def compute_Rusanov_Flux(self, rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, Bx_L, Bx_R, By_L, By_R):

        en_L = P_L / (self.gamma - 1) + 0.5 * rho_L * (vx_L ** 2 + vy_L ** 2) + 1 / 8 / np.pi * (Bx_L ** 2 + By_L ** 2)
        en_R = P_R / (self.gamma - 1) + 0.5 * rho_R * (vx_R ** 2 + vy_R ** 2) + 1 / 8 / np.pi * (Bx_R ** 2 + By_R ** 2)

        rho_avg  = 0.5*(rho_L + rho_R)
        momx_avg = 0.5*(rho_L * vx_L + rho_R * vx_R)
        momy_avg = 0.5*(rho_L * vy_L + rho_R * vy_R)
        en_avg   = 0.5*(en_L + en_R)
        Bx_avg   = 0.5*(Bx_L + Bx_R)
        By_avg   = 0.5*(By_L + By_R)        
        P_avg = (self.gamma - 1) * (en_avg - 0.5 * (momx_avg ** 2 + momy_avg ** 2) / rho_avg - 1 / 8 / np.pi * (Bx_avg ** 2 + By_avg ** 2))

        # compute flux of physical quantities
        flux_m = momx_avg
        flux_px = momx_avg ** 2 / rho_avg + P_avg + 1 / 8 / np.pi * By_avg ** 2 - 1 / 8 / np.pi * Bx_avg ** 2
        flux_py = momx_avg * momy_avg / rho_avg - 1 / 4 / np.pi * Bx_avg * By_avg
        flux_en = (en_avg + P_avg) * momx_avg / rho_avg - 1 / 4 / np.pi * Bx_avg * (Bx_avg * momx_avg + By_avg * momy_avg) / rho_avg
        flux_By = (By_avg * momx_avg - Bx_avg * momy_avg) / rho_avg

        # compute wave speeds
        c0_L = np.sqrt(self.gamma*P_L/rho_L)
        c0_R = np.sqrt(self.gamma*P_R/rho_R)

        ca_L = np.sqrt((Bx_L**2+By_L**2)/rho_L/4/np.pi)
        ca_R = np.sqrt((Bx_R**2+By_R**2)/rho_R/4/np.pi)
        cf_L = np.sqrt(0.5*(c0_L**2+ca_L**2) + 0.5*np.sqrt((c0_L**2+ca_L**2)**2) )
        cf_R = np.sqrt(0.5*(c0_R**2+ca_R**2) + 0.5*np.sqrt((c0_R**2+ca_R**2)**2) )
        C_L = cf_L+ np.abs(vx_L)
        C_R = cf_R + np.abs(vx_R)
        C = np.maximum(C_L, C_R)

        # add stabilizing diffusive term
        flux_m  += C * 0.5 * (rho_R - rho_L)
        flux_px += C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
        flux_py += C * 0.5 * (rho_R * vy_R - rho_L * vy_L)
        flux_en += C * 0.5 * ( en_R - en_L )
        flux_By += C * 0.5 * ( By_R - By_L )

        return flux_m, flux_px, flux_py, flux_en, flux_By

    def compute_constrained_transport(self, bx:np.ndarray, by:np.ndarray, flux_by_X:np.ndarray, flux_bx_Y:np.ndarray, dx:float, dt:float):
        Ez = 0.25 * (-flux_by_X - np.roll(flux_by_X,-1,axis=1) + flux_bx_Y + np.roll(flux_bx_Y,-1,axis=0))
        dbx, dby = compute_curl_z(-Ez, dx)

        bx = bx + dt * dbx
        by = by + dt * dby

        return Ez, bx, by

    def solve(self):

        t = 0
        count = 0

        print("======================================================================")
        print("# Constraint Transport Solver: Initialize Orszag-Tang vortex condition")

        self.plot_snapshot("init", t = 0)

        while t < self.t_end:

            # Primitive variables
            rho = self.rho
            vx = self.vx
            vy = self.vy
            P = self.P
            Bx = self.Bx
            By = self.By

            # Ez, Bx, By
            Ez = self.Ez
            bx = self.Bxh
            by = self.Byh

            # Conserved variables
            m = self.m
            px = self.px
            py = self.py
            en = self.en

            # Compute wave velocities
            vA = self.compute_Alfven_speed(rho, Bx, By)
            vS = self.compute_sound_speed(rho, P, self.gamma)
            vF = self.compute_fast_magnetosonic_speed(vS, vA)

            # time difference based on CFL condition
            dt = self.courant_factor * np.min(self.dx / (vF + np.sqrt(vx**2 + vy**2)))

            if dt > self.dt_min:
                dt = self.dt_min

            # compute gradients of variables
            rho_dx, rho_dy = compute_gradient(rho, self.dx)
            vx_dx, vx_dy = compute_gradient(vx, self.dx)
            vy_dx, vy_dy = compute_gradient(vy, self.dx)
            Bx_dx, Bx_dy = compute_gradient(Bx, self.dx)
            By_dx, By_dy = compute_gradient(By, self.dx)
            P_dx, P_dy = compute_gradient(P, self.dx)

            # Slopelimit to handle discontinuity
            if self.slopelimit:
                rho_dx, rho_dy = slopelimit(rho, rho_dx, rho_dy, self.dx)
                vx_dx, vx_dy = slopelimit(vx, vx_dx, vx_dy, self.dx)
                vy_dx, vy_dy = slopelimit(vy, vy_dx, vy_dy, self.dx)
                Bx_dx, Bx_dy = slopelimit(Bx, Bx_dx, Bx_dy, self.dx)
                By_dx, By_dy = slopelimit(By, By_dx, By_dy, self.dx)
                P_dx, P_dy = slopelimit(P, P_dx, P_dy, self.dx)

            # Extrapolation for half-time : this can be improved through high-order time integration method (RK method)
            rho_h = self.rho - 0.5 * dt * (vx * rho_dx + vx_dx * rho + vy * rho_dy + vy_dy * rho)
            vx_h = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1/rho) * P_dx - 1 / 4 / np.pi / rho * By * By_dx + 1 / 4 / np.pi / rho * By * Bx_dy)
            vy_h = vy - 0.5 * dt * (vy * vy_dy + vx * vy_dx + (1/rho) * P_dy + 1 / 4 / np.pi / rho * Bx * Bx_dy - 1 / 4 / np.pi / rho * Bx * By_dx)
            Bx_h = Bx - 0.5 * dt * (-By * vx_dy + Bx * vy_dy + vy * Bx_dy - vx * By_dy)
            By_h = By - 0.5 * dt * (By * vx_dx - Bx * vy_dx - vy * Bx_dx + vx * By_dx)
            P_h = P - 0.5 * dt * (self.gamma * P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)

            rho_xl, rho_xr, rho_yl, rho_yr = extrapolate_space(rho_h, self.dx, rho_dx, rho_dy)
            vx_xl, vx_xr, vx_yl, vx_yr = extrapolate_space(vx_h, self.dx, vx_dx, vx_dy)
            vy_xl, vy_xr, vy_yl, vy_yr = extrapolate_space(vy_h, self.dx, vy_dx, vy_dy)
            Bx_xl, Bx_xr, Bx_yl, Bx_yr = extrapolate_space(Bx_h, self.dx, Bx_dx, Bx_dy) 
            By_xl, By_xr, By_yl, By_yr = extrapolate_space(By_h, self.dx, By_dx, By_dy)
            P_xl, P_xr, P_yl, P_yr = extrapolate_space(P_h, self.dx, P_dx, P_dy)

            # compute conservative flux
            flux_m_x, flux_px_x, flux_py_x, flux_en_x, flux_by_x = self.compute_Rusanov_Flux(rho_xl, rho_xr, vx_xl, vx_xr, vy_xl, vy_xr, P_xl, P_xr, Bx_xl, Bx_xr, By_xl, By_xr)
            flux_m_y, flux_py_y, flux_px_y, flux_en_y, flux_bx_y = self.compute_Rusanov_Flux(rho_yl, rho_yr, vy_yl, vy_yr, vx_yl, vx_yr, P_yl, P_yr, By_yl, By_yr, Bx_yl, Bx_yr)

            # compute conserved quantities
            m = compute_conserved_field(m, flux_m_x, flux_m_y, self.dx, dt)
            px = compute_conserved_field(px, flux_px_x, flux_px_y, self.dx, dt)
            py = compute_conserved_field(py, flux_py_x, flux_py_y, self.dx, dt)
            en = compute_conserved_field(en, flux_en_x, flux_en_y, self.dx, dt)

            J = self.compute_current(Bx, By)
            w = self.compute_vorticity(vx, vy)
            Pm = self.compute_magnetic_energy(Bx,By)

            Ez, bx, by = self.compute_constrained_transport(bx, by, flux_by_x, flux_bx_y, self.dx, dt)
            Bx, By = compute_avg_field(bx, by)

            # update variables
            self.Ez = Ez
            self.Bxh = bx
            self.Byh = by

            self.J = J
            self.w = w
            self.Pm = Pm
            self.px = px
            self.py = py
            self.m = m
            self.en = en

            self.vx = px / rho
            self.vy = py / rho
            self.rho = m
            self.P = (self.gamma - 1) * (en - 0.5 * (px**2 + py**2) / rho - 1 / 8 / np.pi * (Bx**2 + By**2))
            self.Bx = Bx
            self.By = By

            # check divergence free
            divB = compute_div(Bx, By, self.dx)

            # update time
            t += dt
            count += 1

            # Save trajectories
            self.ts.append(t)
            self.divB.append(np.mean(np.abs(divB)))
            self.KEt.append(np.mean(self.compute_kinetic_energy(rho, vx, vy, P, self.gamma)))
            self.MEt.append(np.mean(self.compute_magnetic_energy(Bx, By)))
            self.Et.append(np.mean(self.compute_total_energy(rho,vx,vy,Bx,By,P, self.gamma)))

            if count % self.verbose == 0:
                print(
                    "t = {:.3f} | divB = {:.3f} | E = {:.3f} | P = {:.3f} | rho = {:.3f}".format(
                        t, np.mean(np.abs(divB)), np.mean(self.en), np.mean(P), np.mean(self.rho)
                    )
                )
                self.record.append(rho)

            # break condition
            if t >= self.t_end:
                break

        self.plot_snapshot(None, t=t)
        self.plot_energy_evolution()
        self.plot_div_B()
        
        if self.animation:
            generate_contourf_gif(self.record, self.savedir, "density_evolution.gif", r"$\rho (x,y,t)$", xmin = 0, xmax = self.L, plot_freq = self.plot_freq)
        
    def plot_snapshot(self, tag:Optional[str], t:float):

        if tag is not None:
            plot_contourf(self.P, self.savedir, "pressure_{}.png".format(tag), "pressure at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.Pm, self.savedir, "pressure_magnetic_{}.png".format(tag), "magnetic pressure at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.en, self.savedir, "energy_{}.png".format(tag), "total energy at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.rho, self.savedir, "density_{}.png".format(tag), "density at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.vx, self.savedir, "vx_{}.png".format(tag), "vx at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.vy, self.savedir, "vy_{}.png".format(tag), "vy at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.Ez, self.savedir, "Ez_{}.png".format(tag), "$E_z$ at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.w, self.savedir, "vorticity_{}.png".format(tag), r"$\nabla \times v(x,y)$ at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.J, self.savedir, "current_{}.png".format(tag), "$J(x,y)$ at t = {:.3f}".format(t), dpi = 160)

        else:
            plot_contourf(self.P, self.savedir, "pressure.png", "pressure at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.Pm, self.savedir, "pressure_magnetic.png", "magnetic pressure at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.en, self.savedir, "energy.png", "total energy at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.rho, self.savedir, "density.png", "density at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.vx, self.savedir, "vx.png", "vx at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.vy, self.savedir, "vy.png", "vy at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.Ez, self.savedir, "Ez.png", "$E_z$ at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.w, self.savedir, "vorticity.png", r"$\nabla \times v(x,y)$ at t = {:.3f}".format(t), dpi = 160)
            plot_contourf(self.J, self.savedir, "current.png", "$J(x,y)$ at t = {:.3f}".format(t), dpi = 160)
        
    def plot_energy_evolution(self):
        # plot the energy dissipation
        plt.figure(figsize=(5, 4))
        plt.plot(self.ts, self.Et, "r-", label="Total")
        plt.plot(self.ts, self.KEt, "b-", label="Kinetic")
        plt.plot(self.ts, self.MEt, "k-", label="Magnetic")
        plt.xlabel("t")
        plt.ylabel("E")
        plt.title("Energy dissipation")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.savedir, "energy_dissipation.png"), dpi=160)

    def plot_div_B(self):
        plt.figure(figsize=(5, 4))
        plt.plot(self.ts, self.divB, "r-", label="Total")
        plt.xlabel("t")
        plt.ylabel(r"$|\nabla \cdot B|$")
        plt.title(r"$|\nabla \cdot B|$")
        plt.tight_layout()
        plt.savefig(os.path.join(self.savedir, "divB.png"), dpi=160)
