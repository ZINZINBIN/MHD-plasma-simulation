import argparse
from src.solver import Solver

def parsing():
    parser = argparse.ArgumentParser(description="MHD simulation code: Constraint transport case")
    parser.add_argument("--num_mesh", type=int, default=200)
    parser.add_argument("--t_srt", type=float, default=0)
    parser.add_argument("--t_end", type=float, default=0.5)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--courant_factor", type=float, default=0.4)
    parser.add_argument("--use_constraint_slope", type=bool, default=True)
    parser.add_argument("--use_animation", type=bool, default=False)
    parser.add_argument("--verbose", type=int, default=16)
    parser.add_argument("--plot_freq", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./results/")
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    
    args = parsing()
    solver = Solver(
        nx = args['num_mesh'],
        ny = args['num_mesh'],
        t_srt = args['t_srt'],
        t_end = args['t_end'],
        L = args['L'],
        is_constraint_slope = args['use_constraint_slope'],
        is_animated = args['use_animation'],
        verbose = args['verbose'],
        save_dir = args['save_dir'],
        plot_freq = args['plot_freq'],
        courant_factor = args['courant_factor'],
    )
    
    solver.solve()