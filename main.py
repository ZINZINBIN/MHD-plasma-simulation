import argparse
from src.sim import Simulation

def parsing():
    parser = argparse.ArgumentParser(description="MHD simulation code: Constraint transport case")
    parser.add_argument("--num_mesh", type=int, default=100)
    parser.add_argument("--t_end", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--courant_factor", type=float, default=0.2)
    parser.add_argument("--slopelimit", type=bool, default=True)
    parser.add_argument("--use_animation", type=bool, default=True)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--plot_freq", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./results/")
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()
    sim = Simulation(
        nx=args["num_mesh"],
        ny=args["num_mesh"],
        t_end=args["t_end"],
        L=args["L"],
        slopelimit=args["slopelimit"],
        animation=args["use_animation"],
        verbose=args["verbose"],
        savedir=args["savedir"],
        plot_freq=args["plot_freq"],
        courant_factor=args["courant_factor"],
    )

    sim.solve()