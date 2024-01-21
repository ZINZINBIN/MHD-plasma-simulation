import numpy as np
import os, math, random
from typing import List, Dict, Optional
from src.compute import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import matplotlib.animation as animation

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
        plot_freq :Optional[int] = 16,
    ):
        # grid configuration
        self.nx = nx
        self.ny = ny

        self.L = L
        self.dx = L / nx

        # Setting
        self.is_constraint_slope = is_constraint_slope
        self.courant_factor = 0.5
        self.verbose = verbose
        self.save_dir = save_dir
        self.is_animated = is_animated
        self.plot_freq = plot_freq

        self.t_srt = t_srt
        self.t_end = t_end

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
        self.total_energy = np.zeros((nx, ny), dtype=np.float32)

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

        Az = np.cos(4 * np.pi * X) / (4 * np.pi * np.sqrt(4 * np.pi)) + np.cos(2 * np.pi * Y) / (2 * np.pi * np.sqrt(4 * np.pi))
        Bxh, Byh = compute_curl_z(Az, self.dx)

        self.Bxh = Bxh
        self.Byh = Byh
        
        Bx, By = compute_avg_field(Bxh, Byh)
        
        self.P = (self.gamma / (4 * np.pi)) * np.ones_like(X) + 0.5 * (Bx**2 + By**2)
        self.Bx = Bx
        self.By = By

        self.total_energy = (self.P -  0.5 * (self.Bx**2 + self.By**2)) / (self.gamma - 1) + 0.5 * (self.vx**2 + self.vy**2) * self.rho + 0.5 * (self.Bx**2 + self.By**2)

        self.m = self.rho * self.dx ** 2
        self.px = self.rho * self.vx * self.dx ** 2
        self.py = self.rho * self.vy * self.dx ** 2
        
    def compute_Alfven_speed(self, rho, Bx, By):
        self.vA = np.sqrt((Bx**2 + By**2) / rho)
        return self.vA

    def compute_sound_speed(self, rho, Bx, By, P):
        self.vS = np.sqrt(self.gamma * (P - 0.5 * (Bx ** 2 + By ** 2)) / rho)
        return self.vS

    def compute_fast_magnetosonic_speed(self, vS, vA):
        self.vF = np.sqrt((vS**2 + vA**2 + np.sqrt(vS**2 + vA**2) ** 2) / 2)
        return self.vF
    
    def compute_total_energy(self, rho, vx, vy, Bx, By, P):
        self.total_energy = (P - 0.5 * (Bx ** 2 + By ** 2))/(self.gamma - 1) + 0.5 * (vx ** 2 + vy ** 2) * rho + 0.5 * (Bx ** 2 + By ** 2)
        return self.total_energy
    
    def compute_thermal_pressure(self, rho, vx, vy, Bx, By, te):
        P = te - 0.5 * (vx ** 2 + vy ** 2) * rho - 0.5 * (Bx ** 2 + By ** 2)
        return P * (self.gamma - 1)
    
    def compute_pressure(self, rho, vx, vy, Bx, By, te):
        P = te - 0.5 * (vx ** 2 + vy ** 2) * rho - 0.5 * (Bx ** 2 + By ** 2)
        self.P = P * (self.gamma - 1) + 0.5 * (Bx ** 2 + By ** 2)
        return P * (self.gamma - 1) + 0.5 * (Bx ** 2 + By ** 2)

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
            vx * vx_dx + vy * vx_dy + P_dx / rho - 2 * 1 / rho * Bx * Bx_dx - 1 / rho * By * Bx_dy - 1 / rho * Bx * By_dy 
        )
        dvy = (-1) * (
            vx * vy_dx + vy * vy_dy + P_dy / rho - 2 * 1 / rho * By * By_dy - 1 / rho * Bx * By_dx - 1 / rho * By * Bx_dx
        )
        dBx = (-1) * (
            vx * By_dy * (-1) + vy * Bx_dy + Bx * vy_dy - By * vx_dy
        )
        dBy = (-1) * (
            vx * By_dx + vy * Bx_dx * (-1) + By * vx_dx - Bx * vy_dx 
        )
         
        dP = (-1) * (
            self.gamma * (P-0.5*(Bx**2+By**2)) * (vx_dx + vy_dy) + By ** 2 * vx_dx + Bx ** 2 * vy_dy +
            (self.gamma - 2) * (Bx*vx + By*vy) * (Bx_dx + By_dy) + 
            vx * P_dx + vy * P_dy - Bx*By*vx_dy - By*Bx * vy_dx
        )
    
        return drho, dvx, dvy, dBx, dBy, dP
    
    def extrapolate_half_time(self, rho, vx, vy, Bx, By, P, dt):
        
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

    def compute_Flux(self, rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, Bx_L, Bx_R, By_L, By_R):
        
        en_L = (P_L - 0.5*(Bx_L**2+By_L**2))/ (self.gamma - 1) + 0.5 * rho_L * (vx_L ** 2 + vy_L ** 2) + 0.5 * (Bx_L ** 2 + By_L ** 2)
        en_R = (P_R - 0.5*(Bx_R**2+By_R**2)) / (self.gamma - 1) + 0.5 * rho_R * (vx_R ** 2 + vy_R ** 2) + 0.5 * (Bx_R ** 2 + By_R ** 2)
        
        rho_avg  = 0.5*(rho_L + rho_R)
        momx_avg = 0.5*(rho_L * vx_L + rho_R * vx_R)
        momy_avg = 0.5*(rho_L * vy_L + rho_R * vy_R)
        en_avg   = 0.5*(en_L + en_R)
        Bx_avg   = 0.5*(Bx_L + Bx_R)
        By_avg   = 0.5*(By_L + By_R)        
        P_avg = (self.gamma - 1) * (en_avg - 0.5 * (momx_avg ** 2 + momy_avg ** 2) / rho_avg - 0.5 * (Bx_avg ** 2 + By_avg ** 2)) + 0.5 * (Bx_avg ** 2 + By_avg ** 2)
        
        # compute flux of physical quantities
        flux_m = momx_avg
        flux_px = momx_avg ** 2 / rho_avg + P_avg + 0.5 * (Bx_avg ** 2 + By_avg ** 2) - Bx_avg ** 2
        flux_py = momx_avg * momy_avg / rho_avg - Bx_avg * By_avg
        flux_en = (en_avg + P_avg) * momx_avg / rho_avg - Bx_avg * (Bx_avg * momx_avg + By_avg * momy_avg) / rho_avg
        flux_By = (By_avg * momx_avg - Bx_avg * momy_avg) / rho_avg
        
        # compute wave speeds   
        c0_L = np.sqrt(self.gamma*((P_L-0.5*(Bx_L**2+By_L**2))/rho_L))
        c0_R = np.sqrt(self.gamma*((P_R-0.5*(Bx_R**2+By_R**2))/rho_R))
        
        ca_L = np.sqrt((Bx_L**2+By_L**2)/rho_L)
        ca_R = np.sqrt((Bx_R**2+By_R**2)/rho_R)
        cf_L = np.sqrt(0.5*(c0_L**2+ca_L**2) + 0.5*np.sqrt((c0_L**2+ca_L**2)**2) )
        cf_R = np.sqrt(0.5*(c0_R**2+ca_R**2) + 0.5*np.sqrt((c0_R**2+ca_R**2)**2) )
        C_L = cf_L+ np.abs(vx_L)
        C_R = cf_R + np.abs(vx_R)
        C = np.maximum(C_L, C_R)
	
        # add stabilizing diffusive term
        flux_m -= C * 0.5 * (rho_L - rho_R)
        flux_px -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
        flux_py -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
        flux_en -= C * 0.5 * ( en_L - en_R )
        flux_By -= C * 0.5 * ( By_L - By_R )
        
        return flux_m, flux_px, flux_py, flux_en, flux_By
    
    def update_variables_by_flux(self, F:np.ndarray, flux_F_X:np.ndarray, flux_F_Y:np.ndarray, dx:float, dt:float):
        F += - dt * dx * flux_F_X
        F +=   dt * dx * np.roll(flux_F_X,1,axis=0)
        F += - dt * dx * flux_F_Y
        F +=   dt * dx * np.roll(flux_F_Y,1,axis=1)
        return F
    
    def compute_constrained_transport(self, bx:np.ndarray, by:np.ndarray, flux_by_X:np.ndarray, flux_bx_Y:np.ndarray, dx:float, dt:float):
        R = -1   
        L = 1  
    
        Ez = 0.25 * ( -flux_by_X - np.roll(flux_by_X,R,axis=1) + flux_bx_Y + np.roll(flux_bx_Y,R,axis=0) )
        dbx, dby = compute_curl_z(-Ez, dx)
        
        bx += dt * dbx
        by += dt * dby
        
        self.Ez = Ez
        
        return bx, by

    def solve(self):
        t = 0
        count = 0

        # initialize
        print("======================================================================")
        print("# Constraint Transport Solver: Initialize Orszag-Tang vortex condition")
        self.set_init_condition()
        
        # save figure
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.plot_contourf(
            self.P,
            title="Initial total pressure",
            save_path=os.path.join(self.save_dir, "pressure_init.png"),
        )
        
        self.plot_contourf(
            self.total_energy,
            title="Initial total energy",
            save_path=os.path.join(self.save_dir, "energy_init.png"),
        )
        
        self.plot_contourf(
            self.vx,
            title="Initial Vx",
            save_path=os.path.join(self.save_dir, "vx_init.png"),
        )
        
        self.plot_contourf(
            self.vy,
            title="Initial Vy",
            save_path=os.path.join(self.save_dir, "vy_init.png"),
        )
        
        self.plot_contourf(
            0.5*(self.Bx ** 2 + self.By ** 2),
            title="Initial magnetic pressure",
            save_path=os.path.join(self.save_dir, "pressure_magnetic_init.png"),
        )

        # Predictive modeling for MHD constraint tranpsort
        print(
            "# Constraint Transport Solver: Iteration for solving MHD tranport equation"
        )
        
        if self.is_animated:
            rho_list = []
            P_list = []
            rho_list.append(self.rho)
            P_list.append(self.P)
            
        else:
            rho_list = None
            P_list = None
        
        while t < self.t_end:
                        
            # compute Alfven speed, sound speed, and fast magnetosonic speed
            vA = self.compute_Alfven_speed(self.rho, self.Bx, self.By)
            vS = self.compute_sound_speed(self.rho, self.Bx, self.By, self.P)
            vF = self.compute_fast_magnetosonic_speed(vS, vA)
            
            # update time interval
            self.dt = self.courant_factor * np.min(self.dx / (vF + np.sqrt(self.vx**2 + self.vy**2)))
            
            # compute gradients of variables
            rho_dx, rho_dy = compute_gradient(self.rho, self.dx)
            vx_dx, vx_dy = compute_gradient(self.vx, self.dx)
            vy_dx, vy_dy = compute_gradient(self.vy, self.dx)
            Bx_dx, Bx_dy = compute_gradient(self.Bx, self.dx)
            By_dx, By_dy = compute_gradient(self.By, self.dx)
            P_dx, P_dy = compute_gradient(self.P, self.dx)
            
            if self.is_constraint_slope:
                rho_dx, rho_dy = constraint_slope(self.rho, self.dx, rho_dx, rho_dy)
                vx_dx, vx_dy = constraint_slope(self.vx, self.dx, vx_dx, vx_dy)
                vy_dx, vy_dy = constraint_slope(self.vy, self.dx, vy_dx, vy_dy)
                Bx_dx, Bx_dy = constraint_slope(self.Bx, self.dx, Bx_dx, Bx_dy)
                By_dx, By_dy = constraint_slope(self.By, self.dx, By_dx, By_dy)
                P_dx, P_dy = constraint_slope(self.P, self.dx, P_dx, P_dy)
            
            # update physical variables : corner-centered values with RK 4th order approximation
            rho, vx, vy, Bx, By, P = self.extrapolate_half_time(self.rho, self.vx, self.vy, self.Bx, self.By, self.P, self.dt * 0.5)
            
            # extrapolate corner-centered frame to face-centered frame
            rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_space(self.rho, self.dx, rho_dx, rho_dy)
            vx_XL,  vx_XR,  vx_YL,  vx_YR  = extrapolate_space(self.vx,  self.dx, vx_dx,  vx_dy)
            vy_XL,  vy_XR,  vy_YL,  vy_YR  = extrapolate_space(self.vy,  self.dx, vy_dx,  vy_dy)
            P_XL,   P_XR,   P_YL,   P_YR   = extrapolate_space(self.P,   self.dx, P_dx,   P_dy)
            Bx_XL,  Bx_XR,  Bx_YL,  Bx_YR  = extrapolate_space(self.Bx,  self.dx, Bx_dx,  Bx_dy)
            By_XL,  By_XR,  By_YL,  By_YR  = extrapolate_space(self.By,  self.dx, By_dx,  By_dy)
              
            # compute conservative flux
            flux_m_x, flux_px_x, flux_py_x, flux_en_x, flux_by_x = self.compute_Flux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, Bx_XL, Bx_XR, By_XL, By_XR)
            flux_m_y, flux_py_y, flux_px_y, flux_en_y, flux_bx_y = self.compute_Flux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, By_YL, By_YR, Bx_YL, Bx_YR)
            
            # update via flux
            self.m = self.update_variables_by_flux(self.m, flux_m_x, flux_m_y, self.dx, self.dt)
            self.px = self.update_variables_by_flux(self.px, flux_px_x, flux_px_y, self.dx, self.dt)
            self.py = self.update_variables_by_flux(self.py, flux_py_x, flux_py_y, self.dx, self.dt)
            self.total_energy = self.update_variables_by_flux(self.total_energy, flux_en_x, flux_en_y, self.dx, self.dt)
            
            bx, by = self.compute_constrained_transport(self.Bxh, self.Byh, flux_by_x, flux_bx_y, self.dx, self.dt)
            Bx, By = compute_avg_field(bx,by)
            
            # update physical variables from conserved quantities
            self.rho = self.m / self.dx ** 2
            self.vx = self.px / self.rho
            self.vy = self.py / self.rho
            
            self.Bxh = bx
            self.Byh = by
            self.Bx = Bx
            self.By = By    
            self.P = self.compute_pressure(self.rho, self.vx, self.vy, self.Bx, self.By, self.total_energy)
            
            # update time
            t += self.dt

            # update macroscopic variables
            total_energy = self.compute_total_energy(self.rho, self.vx, self.vy, self.Bx, self.By, self.P)
            thermal_P = self.compute_thermal_pressure(self.rho, self.vx, self.vy, self.Bx, self.By, total_energy)
            
            if count % self.verbose == 0:
                divB = compute_div(self.Bx, self.By, self.dx)
                print(
                    "(Solver) t = {:.3f} | mean divB = {:.3f} | mean E = {:.3f} | mean P = {:.3f} | mean rho = {:.3f}".format(
                        t, np.mean(np.abs(divB)), np.mean(self.total_energy), np.mean(thermal_P), np.mean(self.rho)
                    )
                )

            # update count
            count += 1
            
            if rho_list is not None:
                rho_list.append(self.rho)
                P_list.append(self.P)
            
        print("# Constraint Transport Solver: Iteration process complete")
        
        # save figure
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.plot_contourf(
            self.P,
            title="Total pressure",
            save_path=os.path.join(self.save_dir, "pressure.png"),
        )
        
        thermal_P = self.compute_thermal_pressure(self.rho, self.vx, self.vy, self.Bx, self.By, total_energy)
        
        self.plot_contourf(
            thermal_P,
            title="Thermal pressure",
            save_path=os.path.join(self.save_dir, "thermal_pressure.png"),
        )
        
        self.plot_contourf(
            total_energy,
            title="Total energy",
            save_path=os.path.join(self.save_dir, "energy.png"),
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

        if self.is_animated:
            print("# Generate animation file")
            fig, ax = plt.subplots(1,1,figsize = (6,4), facecolor = 'white', dpi=160)
            
            def _plot(idx : int, ax:Axes, data_list):
                ax.cla()
                
                data = data_list[idx]
                cbar = np.linspace(0.01, 0.5, num = 128)
                ax.contourf(self.X,self.Y, data, cbar)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            
            replay = lambda idx : _plot(idx, ax, rho_list)
            idx_max = len(rho_list) - 1
            indices = [i for i in range(idx_max)]
            ani = animation.FuncAnimation(fig, replay, frames = indices)
            writergif = animation.PillowWriter(fps = self.plot_freq, bitrate = False)
            ani.save(os.path.join(self.save_dir, "animation.gif"), writergif)
            print("# Complete")

    def plot_contourf(self, vector, title: str, save_path: str):
        
        cbar = np.linspace(np.min(vector), np.max(vector), num = 64)
        plt.figure(figsize=(6, 4), dpi=160)
        plt.contourf(self.X, self.Y, vector, cbar)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)