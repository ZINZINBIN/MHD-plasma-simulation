import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

def plot_contourf(field:np.ndarray, savedir:str, filename:str, title:str, dpi:int = 160):
    # check directory
    filepath = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor="white", dpi=dpi)
    ax.imshow(field, cmap="jet")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi)
    plt.close(fig)

def generate_contourf_gif(record:List, savedir:str, filename:str, title:str, xmin:float, xmax:float, plot_freq:int = 32):
    # check directory
    filepath = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)

    T = len(record)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor="white")
    ax.cla()
    ax.imshow(record[0], cmap="jet")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()

    def _update(idx):
    
        ax.cla()
        ax.imshow(record[idx], cmap="jet")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.tight_layout()

    ani = animation.FuncAnimation(fig, _update, frames=T, interval = 1000// plot_freq, blit=False)

    # Save animation
    ani.save(filepath, writer=animation.PillowWriter(fps=plot_freq, bitrate = False))
    plt.close(fig)