
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

import os
import sys
import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

# from segmentation_models_pytorch.utils.functional import precision  # unused
# from scipy.io import savemat  # unused

import utils.simulation_results as sr
sys.modules["simulation_results"] = sr

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
rcParams["font.size"] = 14
rcParams["legend.fontsize"] = "medium"
rcParams["axes.grid"] = True


def _as_path_list(paths):
    """Normalize a path input (str/pathlike/list/tuple) to a list[str]."""
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        return [str(paths)]
    return [str(p) for p in paths]


def _build_eta_and_trans_end_arrays(dirs, etas, trans_end_common=14000):
    """
    Build:
      eta_array:       shape (P, E)
      trans_end_array: shape (P, E) filled with trans_end_common
    where P = len(dirs), E = len(etas)
    """
    dirs = _as_path_list(dirs)
    etas = np.array(etas, dtype=float).reshape(1, -1)  # (1, E)
    eta_array = np.repeat(etas, repeats=len(dirs), axis=0)  # (P, E)
    trans_end_array = np.full((len(dirs), eta_array.shape[1]), int(trans_end_common), dtype=int)
    return eta_array, trans_end_array


def _build_window_array(dirs, windows):
    """
    window_array should be indexable as window_array[p] and iterable.
    Shape: (P, W)
    """
    dirs = _as_path_list(dirs)
    windows = np.array(windows, dtype=int).reshape(1, -1)  # (1, W)
    return np.repeat(windows, repeats=len(dirs), axis=0)  # (P, W)


def plot_time_to_reliability(path_to_load, eta_array, window_array, t_sim, **kwargs):
    labels = kwargs.get("labels", None)
    output_dir = kwargs.get("output_dir", "./matlab_plotting")
    plot_per_rel = kwargs.get("plot_per_rel", True)
    constraint = kwargs.get("constraint", 1e-1)
    theta1 = kwargs.get("theta1", 5e-1)
    n_rel = kwargs.get("n_rel", 7)
    M = kwargs.get("M", 1.5)

    enable_label_plotting = kwargs.get("enable_label_plotting", False)
    enable_bound_plot = kwargs.get("bound", False)
    gamma = kwargs.get("gamma", 1)
    frame_flag = kwargs.get("frame_flag", True)

    path_to_load = _as_path_list(path_to_load)
    if labels is None:
        labels = [os.path.basename(os.path.normpath(p)) or str(p) for p in path_to_load]

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    avg_reliability = np.zeros((int(n_rel), int(t_sim)))

    upper_bound = None  # will be set from last loaded sim
    window = None

    for p, path in enumerate(path_to_load):
        for r in range(1, n_rel + 1):
            for eta in eta_array[p]:
                for window in window_array[p]:
                    sim_path = fr"{path}/simulation_{float(eta)}_window_size_{int(window)}_{r}.pkl"
                    try:
                        with open(sim_path, "rb") as input_file:
                            e = pickle.load(input_file)
                    except FileNotFoundError:
                        print(f"[WARN] Missing file: {sim_path}")
                        continue

                    gamma = e.gamma
                    if e.theta1 is not None:
                        theta1 = e.theta1
                    constraint = e.alpha

                    if enable_bound_plot:
                        gamma = e.gamma
                        theta1 = e.theta1

                    if frame_flag:
                        reliability_array = np.mean(e.average_reliability_over_multiple_decisions, axis=0)
                    else:
                        reliability_array = np.mean(e.average_reliability, axis=0)

                    F = int(t_sim / window)

                    # Avoid divide-by-zero in bound (start from 1)
                    raxis = np.arange(1, F + 1, dtype=float)

                    reliability_array = np.repeat(reliability_array, int(window))

                    upper_bound = np.divide(M - theta1, gamma * raxis)
                    avg_reliability[r - 1, :] = reliability_array[:t_sim]

                    if plot_per_rel:
                        if enable_label_plotting:
                            plt.plot(reliability_array[:t_sim], label=labels[p])
                        else:
                            plt.plot(reliability_array[:t_sim])

                        plt.plot(np.repeat(upper_bound + constraint, int(window))[:t_sim], linestyle="dashed")

    if output_dir is not None:
        np.save(f"{output_dir}/reliability_plot.npy", avg_reliability)

    if not plot_per_rel and upper_bound is not None and window is not None:
        plt.plot(np.repeat(upper_bound + constraint, int(window))[:t_sim], linestyle="dashed")
        avg = np.mean(avg_reliability, axis=0)
        std = np.std(avg_reliability, axis=0)
        plt.plot(avg, linewidth=2)
        plt.fill_between(np.linspace(start=1, stop=t_sim, num=t_sim), avg - std, avg + std, alpha=0.6)

    plt.hlines(xmin=0, xmax=t_sim, y=constraint, linestyles="dashed", linewidth=2)
    plt.xlabel("Time Slot Index [t]", fontsize=14)
    plt.ylim([constraint - 0.02, constraint + 0.02])
    plt.xlim([0, t_sim])
    plt.ylabel("FNR", fontsize=14)
    if enable_label_plotting or plot_per_rel:
        plt.legend()
    plt.show()


def plot_queues(path_array, eta_array, window_array, t_sim):
    path_array = _as_path_list(path_array)

    for p, path_to_load in enumerate(path_array):
        for eta in eta_array[p]:
            for window in window_array[p]:
                sim_path = fr"{path_to_load}/simulation_{float(eta)}_window_size_{int(window)}_1.pkl"
                try:
                    with open(sim_path, "rb") as input_file:
                        e = pickle.load(input_file)
                except FileNotFoundError:
                    print(f"[WARN] Missing file: {sim_path}")
                    continue

                queue_tracker = e.queue_tracker
                plt.plot(np.mean(np.sum(queue_tracker, axis=1), axis=1), label=rf"$\eta = {eta}$")

    plt.xlabel("Time Slot Index [t]", fontsize=14)
    plt.ylabel("Virtual Queues", fontsize=14)
    plt.legend()
    plt.show()


def plot_virtual_queues(path_to_load, eta_array, window_length, t_sim):
    path_to_load = _as_path_list(path_to_load)

    for p, path in enumerate(path_to_load):
        for eta in eta_array[p]:
            sim_path = fr"{path}/simulation_{float(eta)}_window_size_{int(window_length)}_1.pkl"
            try:
                with open(sim_path, "rb") as input_file:
                    e = pickle.load(input_file)
            except FileNotFoundError:
                print(f"[WARN] Missing file: {sim_path}")
                continue

            queue_tracker = e.fnr_outage_queues
            plt.plot(np.mean(queue_tracker, axis=1)[0:t_sim], label=rf"$\eta = {eta}$")

    plt.xlabel("Time Slot Index [t]", fontsize=14)
    plt.ylabel("Average virtual queues", fontsize=14)
    plt.legend()
    plt.show()


def plot_energy_precision_trade_off(path_to_load, eta_array, window_size, t_sim, trans_end=0, **kwargs):
    matlab_out = kwargs.get("matlab_out", "./matlab")
    realization_number = kwargs.get("realization_number", 1)
    start_rel = kwargs.get("start_rel", 5)
    labels = kwargs.get("labels", None)

    path_to_load = _as_path_list(path_to_load)
    if labels is None:
        labels = [os.path.basename(os.path.normpath(p)) or str(p) for p in path_to_load]

    if matlab_out is not None and not os.path.exists(matlab_out):
        os.makedirs(matlab_out, exist_ok=True)

    trans_end_array = trans_end  # can be scalar or (P,E)

    # if scalar, broadcast it
    if np.isscalar(trans_end_array):
        trans_end_array = np.full((len(path_to_load), eta_array.shape[1]), int(trans_end_array), dtype=int)

    for p, path in enumerate(path_to_load):
        for window in window_size[p]:
            energy_array = np.zeros(eta_array.shape[1], dtype=float)
            precision_array = np.zeros(eta_array.shape[1], dtype=float)

            for j, eta in enumerate(eta_array[p]):
                for r in range(start_rel, start_rel + realization_number):
                    sim_path = fr"{path}/simulation_{float(eta)}_window_size_{int(window)}_{r}.pkl"
                    try:
                        with open(sim_path, "rb") as input_file:
                            e = pickle.load(input_file)
                    except FileNotFoundError:
                        print(f"[WARN] Missing file: {sim_path}")
                        continue

                    precision_tracker = e.prediction_set_cardinality
                    energy_tracker = e.power_consumption_over_time

                    te = int(trans_end_array[p, j])
                    precision_tracker = precision_tracker[te:, :]
                    precision_tracker = precision_tracker[precision_tracker >= 0]

                    if precision_tracker.size > 0:
                        precision_array[j] += (1 - np.mean(precision_tracker))
                    energy_array[j] += np.mean(np.sum(np.sum(energy_tracker[te:, :, :], axis=2), axis=1))

                precision_array[j] /= realization_number
                energy_array[j] /= realization_number

            plt.plot(energy_array * 0.05 * 1000, precision_array, marker="o", label=rf"{labels[p]} $S={int(window)}$")

            if matlab_out is not None:
                np.save(f"./{matlab_out}/energy_array_{int(window)}_{labels[p]}.npy", energy_array * 1e3)
                np.save(f"./{matlab_out}/precision_array_{int(window)}_{labels[p]}.npy", precision_array)

    plt.legend()
    plt.xlabel("System Power Consumption [mW]")
    plt.ylabel("Precision")
    plt.show()


def _interactive_fallback():
    t_sim = 15000
    window_array = np.array([[10], [10]])
    trans_end_array = np.array([
        [14000, 14000, 14000, 14000, 14000, 14000, 13100],
        [14000, 14000, 14000, 14000, 14000, 14000, 14000],
    ])
    eta_array = np.array([
        [0.25, 0.3, 0.4, 0.42, 0.6, 0.6, 0.8],
        [0.2,  0.3, 0.4, 0.5,  0.6, 0.6, 0.8],
    ])

    print("Selection Menu")
    print("1 - Time to reliability")
    print("2 - Precision/Energy trade-off")
    print("4 - Plot queues")
    print("5 - Plot deciding node rates")

    choice = int(input("What do you want to plot? "))
    matlab_out = "./matlab"

    path = ["./sim_res/WiOPTSimCLO15k/", "./sim_res/WiOPTSimLyapunov15k/"]

    if choice == 1:
        method = int(input("Choose the method (1 - CLO, 2 - LO): "))
        if method == 1:
            path_to_load = f"{path[0]}/multiple_realizations"
        else:
            path_to_load = f"{path[1]}/multiple_realizations"

        n_rel = int(input("Select the number of realizations to plot: "))
        eta_array_local = np.array([[0.8]])

        plot_time_to_reliability(
            [path_to_load],
            eta_array_local,
            window_array,
            t_sim,
            matlab_out=matlab_out,
            trans_end=0,
            n_rel=n_rel,
            B=1,
            b=1,
            m=0,
            M=(1 + 0.5),
            plot_per_rel=True,
            output_dir="./matlab_plotting",
            frame_flag=True,
        )

    elif choice == 2:
        plot_energy_precision_trade_off(
            path,
            eta_array,
            window_array,
            t_sim,
            trans_end=trans_end_array,
            start_rel=1,
            realization_number=1,
        )

    else:
        print("Not yet implemented...")



def main():
    if len(sys.argv) == 1:
        _interactive_fallback()
        return

    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    subparsers = parser.add_subparsers(dest="cmd")

    # Shared args helper
    def add_shared(subp):
        subp.add_argument("-d", "--dirs", nargs="+", required=True, help="One or more directories containing simulations.")
        subp.add_argument("-e", "--etas", nargs="+", type=float, required=True, help="List of eta values (space-separated).")
        subp.add_argument("--trans-end", type=int, default=14000, help="Common trans_end value (default: 14000).")
        subp.add_argument("-w", "--windows", nargs="+", type=int, default=[10], help="Window sizes (default: 10).")
        subp.add_argument("--t-sim", type=int, default=15000, help="Simulation horizon (default: 15000).")
        subp.add_argument("--labels", nargs="*", default=None, help="Optional labels for each directory (same length as --dirs).")

    # time-to-reliability
    p1 = subparsers.add_parser("time", help="Plot time-to-reliability (FNR over time).")
    add_shared(p1)
    p1.add_argument("--n-rel", type=int, default=7, help="Number of realizations (default: 7).")
    p1.add_argument("--plot-per-rel", action="store_true", help="Plot each realization (instead of mean/std).")
    p1.add_argument("--output-dir", default="./matlab_plotting", help="Output dir for numpy arrays.")
    p1.add_argument("--frame-flag", action="store_false", help="Use average_reliability_over_multiple_decisions.")

    # energy/precision tradeoff
    p2 = subparsers.add_parser("tradeoff", help="Plot energy vs precision trade-off.")
    add_shared(p2)
    p2.add_argument("--matlab-out", default="./matlab", help="Dir for saving .npy arrays.")
    p2.add_argument("--start-rel", type=int, default=1, help="First realization index (default: 1).")
    p2.add_argument("--realizations", type=int, default=1, help="Number of realizations to average (default: 1).")

    # queues
    p3 = subparsers.add_parser("queues", help="Plot virtual queues.")
    add_shared(p3)

    args = parser.parse_args()

    dirs = _as_path_list(args.dirs)
    if args.labels is not None and len(args.labels) not in (0, len(dirs)):
        raise ValueError("If provided, --labels must have the same length as --dirs.")

    eta_array, trans_end_array = _build_eta_and_trans_end_arrays(dirs, args.etas, args.trans_end)
    window_array = _build_window_array(dirs, args.windows)

    if args.cmd == "time":
        plot_time_to_reliability(
            dirs,
            eta_array,
            window_array,
            args.t_sim,
            n_rel=args.n_rel,
            plot_per_rel=args.plot_per_rel,
            output_dir=args.output_dir,
            frame_flag=args.frame_flag,
            enable_label_plotting=True,
            labels=args.labels,
        )

    elif args.cmd == "tradeoff":
        plot_energy_precision_trade_off(
            dirs,
            eta_array,
            window_array,
            args.t_sim,
            trans_end=trans_end_array,
            matlab_out=args.matlab_out,
            start_rel=args.start_rel,
            realization_number=args.realizations,
            labels=args.labels,
        )

    elif args.cmd == "queues":
        plot_queues(dirs, eta_array, window_array, args.t_sim)

    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

