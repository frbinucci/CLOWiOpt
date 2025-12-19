#  The MIT License (MIT)
#  Copyright © 2025 University of Perugia
#
#  Author:
#
#  - Francesco Binucci (francesco.binucci@cnit.it)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from pathlib import Path

import numpy as np
import yaml

def generate_network(path_loss_matrix,t_sim):

    channel_gain_matrix = np.zeros((t_sim,path_loss_matrix.shape[0],path_loss_matrix.shape[1]))
    i=0
    for row in path_loss_matrix:
        j=0
        for cell in row:
            if cell!=0:
                sigma = 1 / np.sqrt(2)
                pl_linear = cell
                channel = sigma * np.abs(np.random.randn(1, t_sim) + 1j * np.random.randn(1, t_sim))
                channel = np.sqrt(pl_linear) * channel.flatten()
                channel_gain_matrix[:,i,j]= channel
            j+=1
        i+=1

    return channel_gain_matrix

def load_cfg(cfg_path: str) -> dict:
    cfg_path = Path(cfg_path).resolve()
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["_base_dir"] = cfg_path.parent  # utile se vuoi path relativi al config
    return cfg

def build_matrices(cfg: dict):
    N_nodes = cfg["network"]["N_nodes"]
    N_users = cfg["network"]["N_users"]

    adj = np.zeros((N_nodes, N_nodes))
    bw_m = np.zeros((N_nodes, N_nodes))

    for (i, j, pl) in cfg["network"]["links"]:
        adj[int(i), int(j)] = float(pl)

    bandwidth = float(cfg["network"]["Bw"])
    for i in range(N_nodes):
        for j in range(N_nodes):
            bw_m[i, j] = bandwidth * (adj[i, j] > 0)

    return N_nodes, N_users, adj, bw_m


def print_percentage_bar(iteration, total, bar_length=50):
    """
    Prints a percentage bar.

    Args:
        iteration (int): Current iteration.
        total (int): Total iterations.
        bar_length (int): Length of the percentage bar.
    """
    progress = iteration / total
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    percentage = progress * 100
    print(f"\r|{bar}| {percentage:.2f}%", end="")
    if iteration == total:
        print()  # Move to the next line when complete