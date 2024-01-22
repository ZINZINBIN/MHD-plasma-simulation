# MHD simulation
## Introduction
This code is 2D MHD simulation of Orszag-Tang vortex with Finite Volume Method. Based on Philip Mocz' code, RK4 explicit method is additinoally applied. 

## Simulation Result
<div>
    <p float = 'left'>
        <img src="/results/density.png"  width="360" height="240">
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
- Philip Mocz, Constrained Transport Magnetohydrodynamics Simulation: https://levelup.gitconnected.com/create-your-own-constrained-transport-magnetohydrodynamics-simulation-with-python-276f787f537d
- Philip Mocz, Fininte volume code: https://github.com/pmocz/finitevolume-python
- crewsdw, Magnetohydrodynamics2D: https://github.com/crewsdw/Magnetohydrodynamics2D