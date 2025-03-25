# MHD simulation
## Introduction
This is a github repository for 2-dimensional Magnetohydrodynamic (MHD) simulation for Orszag-Tang vortex. The Orszag-Tang vortex is a supersonic 2D MHD turbulence model problem and is given by the initial condition $B = -sin(y) \hat{x} + sin(2x) \hat{y}$ and $v = -sin(y) \hat{x} + sin(x) \hat{y}$. This condition evolves into highly nonlinear turbulence in 2-dimensional systems. This code simulates the evolution of the MHD turbulence with <a href = "https://en.wikipedia.org/wiki/Finite_volume_method">Finite Volume method </a>. The time integration utilized in this code is <a href = "https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods">explicit Runge-Kunta 4th order method</a>. 

## Simulation
The below figure represents the simluation result of MHD turbulence. 
<div>
    <p float = 'left'>
        <img src="./results/density.png"  width="45%">
        <img src="./results/energy.png"  width="45%">
    </p>
</div>

The energy with respect to time given as below figure shows the energy dissipation.

<div>
    <p float = 'left'>
        <img src="./results/energy_dissipation.png"  width="60%">
    </p>
</div>

## How to run
```
    python3 main.py --num_mesh {int} --t_srt {float} --t_end {float}
                    --L {float} --use_constraint_slope {boolean} --use_animation {boolean}
                    --verbose {int} --plot_freq {int} --save_dir {path}
                    --courant_factor {float}
```

## Reference
There are good sources to be referred for understanding the MHD simulation and details on finite volume method. 
- Philip Mocz, Constrained Transport Magnetohydrodynamics Simulation: https://levelup.gitconnected.com/create-your-own-constrained-transport-magnetohydrodynamics-simulation-with-python-276f787f537d
- Philip Mocz, Fininte volume code: https://github.com/pmocz/finitevolume-python
- crewsdw, Magnetohydrodynamics2D: https://github.com/crewsdw/Magnetohydrodynamics2D