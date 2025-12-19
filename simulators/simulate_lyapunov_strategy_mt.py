import argparse
import os.path
import pickle
import sys
from collections import deque

#import torchsummary

import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from optimizers.LO_Optimizer import solve_instantaneous_opt_problem
from utils.simulation_results import Simulation
from utils.system_status import SystemStatus
import time

import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import yaml

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


if __name__=='__main__':



    ENCODER_MAP = {0: 'timm-mobilenetv3_small_minimal_100', 2: 'resnet18', 3: 'resnet50'}
    ENCODER_PFR_PREDICTOR_MAP = {0: 'timm-mobilenetv3_small_minimal_100', 2: 'mobileone_s0', 3: 'mobileone_s0'}

    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'
    ACTIVATION = None

    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument("--config", type=str, default="configs/CLO-config.yaml")
    parser.add_argument("--data_output_dir",type=str,default='./simulation_results',help='Directory in which save sim results')
    parser.add_argument("--eta",type=float,default=0,help='The value of the Penalty Parameter for the prediction-set cardinality')
    parser.add_argument("--V",type=float,default=1e1,help='The value of the Penalty Parameter associated to the energy consumption')
    parser.add_argument("--window_length",type=int,default=1,help='Window length for adaptive_crc')
    parser.add_argument("--t_sim",type=int,default=int(5e3),help='Simulation duration')
    parser.add_argument("--tau",type=float,default=50e-3,help='Time-slot duration')
    parser.add_argument("--error_probability_constraint",type=float,default=0,help='Error Prob Constraint')
    parser.add_argument("--prediction_set_cardinality_constraint",type=float,default=3,help='Prediction Set Cardinality constraint')
    parser.add_argument("--energy_constraint",type=float)
    parser.add_argument("--simulation_index",type=int,default=0,help='Simulation Index (for average simulations)')
    parser.add_argument("--enable_plotting",type=bool,default=False,help='Enable virtual queue dynamical plotting')
    parser.add_argument("--number_of_realizations",type=int,default=1,help='Number of sim realizations')
    parser.add_argument("--gamma",type=float,default=5e-1,help="OCRC step size")
    parser.add_argument("--alpha",type=float,default=0.12,help="Reliability Constraint")
    parser.add_argument("--fixed_threshold",type=float,default=None,help='Threshold')
    parser.add_argument("--min_arrival_rate",type=float,default=0.4,help="Minimum arrival rate")
    parser.add_argument("--max_arrival_rate",type=float,default=0.8,help="Maximum arrival rate")
    parser.add_argument("--arrival_rate_refresh_rate",type=float,default=100,help="Arrival rate stationary time")
    parser.add_argument("--init_seed",type=int,default=1,help="Simulation seed")
    parser.add_argument("--init_sim_index",type=int,default=0,help='Init sim. index')


    #Network Constraints
    parser.add_argument("--transmission_constraints",type=int,default=1,help='Transmission Constraints (common for all the users)')
    parser.add_argument("--computational_constraints",type=int,default=3,help='Computational Constraints (common for all the nodes)')


    #Lyapunov Optimization Parameteres
    parser.add_argument("--fnr_step_sizes",type=float,default=0.5,help='FNR step size')
    parser.add_argument("--fnr_virtual_queues_init",type=float,default=0.5,help='FNR virtual queues init')


    #Learning task features
    parser.add_argument('--data_unit_bits',type=int,default=32,help='Bit per pixel (images encoding)')
    parser.add_argument('--image_size',type=int,default=3*256**2,help='Images size (n_ch,width,height)')


    #Transmission parameters
    parser.add_argument("--pmax",type=float,default=3.5,help='Maximum transmit power')
    parser.add_argument("--Bw",type=float,default=20e6,help='Transmission Bandwidth')

    parser.add_argument("--N0",type=float,default=3.98e-21,help='Noise Power Spectral Density')


    args, unknown = parser.parse_known_args()

    t_sim = args.t_sim
    energy_constraint = args.energy_constraint
    V = args.V
    eta = args.eta
    tau = args.tau
    a_crc_refresh_freq = args.window_length
    data_output_dir = args.data_output_dir

    n_realizations = args.number_of_realizations
    arrival_rate_refresh_rate = args.arrival_rate_refresh_rate
    min_a = args.min_arrival_rate
    max_a = args.max_arrival_rate
    init_seed = args.init_seed
    init_sim_index = args.init_sim_index

    np.random.seed(init_seed)
    torch.manual_seed(init_seed)

    gamma = args.fnr_step_sizes
    init_queues = args.fnr_virtual_queues_init

    data_unit_bits = args.data_unit_bits
    image_size = args.image_size


    print(f'Torch seed: {torch.initial_seed()}')



    for n_rel in range(1,n_realizations+1):

        if not os.path.exists(data_output_dir):
            os.makedirs(data_output_dir)

        N_nodes = 4
        N_users = 3
        adj_matrix = np.zeros((N_nodes,N_nodes))
        bandwidth_matrix = np.zeros((N_nodes,N_nodes))

        adj_matrix[0,3] = 1e-9
        adj_matrix[1,3] = 1e-9
        adj_matrix[2,3] = 1e-9



        channel_gain_matrix = generate_network(adj_matrix,t_sim)

        #Optimization_setting
        tx_power_array = args.pmax*np.ones(N_nodes)
        router_queues_matrix = np.zeros((t_sim,N_nodes*N_users))
        users_queues_matrix = np.zeros((t_sim,N_nodes))
        data_unit_bits = np.array([data_unit_bits,data_unit_bits,data_unit_bits])*image_size
        N0 = args.N0
        user_indexes = np.array([0,1,2])
        arrival_rates = min_a*np.ones(N_users)

        transmission_constraints_matrix = args.transmission_constraints*np.ones((N_nodes,N_nodes))
        computational_constraints_matrix = args.computational_constraints*np.ones(N_nodes)
        bandwidth = args.Bw

        #Learning Models
        learning_models = {}
        cardinality_predictors = {}
        depth_mapping = np.array([0,1,2,3])
        models_association = {0:0,1:0,2:0,3:3}
        for n_layers in depth_mapping:
            if not models_association[n_layers]==None:
                network = smp.Unet(
                    encoder_name=ENCODER_MAP[models_association[n_layers]],
                    encoder_weights=ENCODER_WEIGHTS,
                    in_channels=3,
                    classes=1,
                )
                network.load_state_dict(torch.load(
                    f'utils/learning_models/D{models_association[n_layers]}/segmentation_network'))
                network.eval()
                network = network.to('cuda')
            else:
                network=None
            learning_models[n_layers] = network
            if not models_association[n_layers]==None:
                learning_models[n_layers] = network
                #summary(network, torch.zeros(1,1, 28, 28))
                cardinality_predictor = smp.PSPNet(
                    encoder_name=ENCODER_PFR_PREDICTOR_MAP[models_association[n_layers]],
                    encoder_weights=ENCODER_WEIGHTS,
                    in_channels=3,
                    classes=1,
                )
                cardinality_predictor.load_state_dict(torch.load(
                    f'utils/precision_predictors/D{models_association[n_layers]}/precision_predictor'))
                cardinality_predictor = cardinality_predictor.to('cuda')
                cardinality_predictor.eval()
            else:
                cardinality_predictor = None
            cardinality_predictors[n_layers] = cardinality_predictor

        system_status = SystemStatus(t_sim=t_sim,N_users=N_users,window_size=a_crc_refresh_freq)

        system_status.cardinality_surrogate = cardinality_predictors
        system_status.learning_models = learning_models

        for i in range(0,bandwidth_matrix.shape[0]):
            for j in range(0,bandwidth_matrix.shape[1]):
                bandwidth_matrix[i,j]=bandwidth*(adj_matrix[i,j]>0)
                transmission_constraints_matrix[i,j] = transmission_constraints_matrix[i,j]*(adj_matrix[i,j]>0)

        node_queues = np.zeros((N_nodes,N_users))

        system_status.bandwidth_matrix = bandwidth_matrix
        system_status.N0 = N0
        system_status.N_nodes = N_nodes
        system_status.p_max_array = tx_power_array
        system_status.N_users = N_users
        system_status.channel_status_matrix = channel_gain_matrix[0,:]**2
        system_status.node_queues = node_queues
        system_status.V =V
        system_status.bandwidth_matrix = bandwidth_matrix
        system_status.eta = eta
        system_status.tau = tau
        system_status.computational_constraints = computational_constraints_matrix
        system_status.transmission_constraints = transmission_constraints_matrix
        system_status.data_unit_bits = data_unit_bits
        system_status.user_indexes = user_indexes
        #system_status.cardinality_surrogate = cardinality_predictors
        system_status.learning_models = learning_models
        system_status.gamma = 1*np.ones(N_users)
        system_status.reliability_constraints = args.alpha*np.ones(N_users)
        system_status.alpha_array = 5e-1*np.ones(N_users)
        system_status.dynamic_error_tracker = np.zeros(N_users)
        system_status.dynamic_prediction_counter = np.zeros(N_users)

        system_status.accuracy_step_size = 5*np.ones(3)


        system_status.enable_crc = False
        system_status.enable_prediction_set_cardinality_constraint = False
        system_status.place_holder = 0

        data_units_queues = {}
        for i in range(N_nodes):
            if i in range(N_users):
                data_units_queues.update({(i,i):list()})
            else:
                for k in range(N_users):
                    if i not in range(N_users):
                        data_units_queues.update({(i,k):list()})

        system_status.data_units_queues = data_units_queues

        I_opt_tracker = np.zeros((t_sim,N_nodes,N_users))
        P_opt_tracker = np.zeros((t_sim,N_nodes,N_nodes))
        R_opt_tracker = np.zeros((t_sim,N_nodes,N_nodes,N_users))
        alpha_tracker = np.zeros((t_sim,N_users))
        q_hat_tracker = np.zeros((t_sim,N_users))
        queue_tracker = np.zeros((t_sim,N_nodes,N_users))
        prediction_set_cardinality_tracker = np.zeros((t_sim,N_users))
        energy_virtual_queue_tracker = np.zeros((t_sim,N_users))

        prediction_set_cardinality_virtual_queue_tracker = np.zeros((t_sim,N_users))
        accuracy_virtual_queue_tracker = np.zeros((t_sim,N_users))
        error_probability_tracker = np.zeros((t_sim,N_users))
        real_time_variance_tracker = np.zeros((t_sim,N_users))



        system_status.init_actual_queues(np.random.binomial(1,arrival_rates))
        overall_trak = 0
        made_decisions = 0

        mobile_mean_reliability_tracker = np.zeros((t_sim,N_nodes,N_users))
        reliability_tracker = np.zeros((t_sim,N_users))
        fnr_virtual_queues = args.fnr_virtual_queues_init*np.ones(N_users)
        std_virtual_queue_tracker =  np.zeros((t_sim,N_users))
        std_virtual_queues  = np.zeros(N_users)
        fnr_step_sizes = args.fnr_step_sizes*np.ones(N_users)

        fnr_virtual_queue_tracker = np.zeros((t_sim,N_users))
        print(f"Simulating eta ={eta}...")
        print(fnr_virtual_queues)
        system_status.init_queues()
        state = True

        model_list = ['D1','D2','D3']

        print(f'Data output dir :{data_output_dir}, eta = {eta}, S = {a_crc_refresh_freq}')
        start_time = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
        print(f'Simulation start: {start_time}')

        theta_grid = np.linspace(start=0,stop=10,num=10)
        #theta_opt = theta_grid[4]
        if args.enable_plotting == True:

            plt.ion()

            # Create figure and axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

            line1, = ax1.plot([], [], 'b-')  # 'b-' for blue solid line
            line2, = ax2.plot([], [], 'r-')  # 'r-' for red solid line

            # Initialize data for both subplots
            x_data, y1_data, y2_data = [], [], []

            ax1.set_title('Node Queues')
            ax2.set_title("FNR virtual queues")

            ax1.grid()
            ax2.grid()


            hl, = plt.plot([], [])

            data_points = deque(maxlen=t_sim)

        print(os.getcwd())
        theta_array = np.load(f'utils/LUTs/theta_lut.npy')
        if not args.fixed_threshold == None:
            theta_array = np.array([theta_array[args.fixed_threshold]])
            system_status.alpha_array = theta_array[0] * np.ones(N_users)
        else:
            print("Threshold Optimization Enabled!")

        for t in range(0,t_sim):
            print_percentage_bar(t, t_sim)

            if t % arrival_rate_refresh_rate == 0:
                r = np.random.rand()
                if r < 0.5:
                    arrival_rates = min_a*np.ones(N_users)
                else:
                    arrival_rates = max_a*np.ones(N_users)

            best_value = np.inf
            for theta in theta_array:
                R_star,I_star,P_star,flag,value = solve_instantaneous_opt_problem(system_status,fnr_virtual_queues,fnr_step_sizes,theta,model_list=['D0','D0','D0','D3'])
                if value<best_value:
                    best_value = value
                    R_opt = R_star
                    I_opt = I_star
                    P_opt = P_star
                    theta_opt = theta
            system_status.alpha_array = theta_opt*np.ones(N_users)

            arrivals = np.random.binomial(1,arrival_rates)

            if flag==False:
                if (t + 1) % a_crc_refresh_freq == 0:
                    system_status.update_crc_param=True



                prediction_set_cardinality, tmp_error_tracker = system_status.adaptive_crc(I_opt)

                if (t+1)%a_crc_refresh_freq==0:
                    for k in range(N_users):
                        fnr_virtual_queues[k] = np.maximum(fnr_virtual_queues[k] + fnr_step_sizes[k] * (tmp_error_tracker[k] - system_status.reliability_constraints[k]), 0)
                #
                # if (t+1)%a_crc_refresh_freq==0:
                #     print(f'Avg FNR = {np.mean(system_status.average_reliability_per_frame_over_multiple_predictions[:,system_status.frame_index-1])}')


                system_status.overall_energy_consumption = np.sum(P_opt)
                system_status.update(R_opt,I_opt,
                                     arrivals,
                                     channel_gain_matrix[t,:]**2)

                prediction_set_cardinality_tracker[t,:] = prediction_set_cardinality
                I_opt_tracker[t,:] = I_opt

                for i in range(N_nodes):
                    for j in range(N_nodes):
                        if np.sum(R_opt[i,j,:])>transmission_constraints_matrix[i,j]:
                            print(f'Node {i} is transmitting {np.sum(R_opt[i,j,:])} DUs towards node {j}')

                for k in range(N_users):
                    reliability_tracker[t,k] = system_status.slot_fnr[k]


                made_decisions+=np.sum(np.sum(I_opt))


                fnr_virtual_queue_tracker[t,:] = fnr_virtual_queues
                P_opt_tracker[t,:] = P_opt
                R_opt_tracker[t,:] = R_opt
                queue_tracker[t,:] = system_status.node_queues
                alpha_tracker[t,:] = system_status.alpha_array
                q_hat_tracker[t,:] = system_status.q_hat_array
                energy_virtual_queue_tracker[t,:] = system_status.energy_virtual_queue
                prediction_set_cardinality_virtual_queue_tracker[t,:] = system_status.prediction_set_cardinality_virtual_queue




                if args.enable_plotting == True:
                    x = t
                    y1 = system_status.node_queues
                    y2 = fnr_virtual_queues
                    x_data.append(t)
                    y1_data.append(np.mean(y1))
                    y2_data.append(np.mean(y2))

                    line1.set_xdata(x_data)
                    line1.set_ydata(y1_data)

                    # Rescale axes if needed
                    ax1.relim()  # Recompute limits based on the new data
                    ax1.autoscale_view()  # Rescale the view

                    line2.set_xdata(x_data)
                    line2.set_ydata(y2_data)

                    ax2.relim()
                    ax2.autoscale_view()

                    # Update the figure
                    fig.canvas.draw()
                    fig.canvas.flush_events()


        if args.enable_plotting == True:
            plt.show()

        simulation=Simulation(R=R_opt_tracker,
                              I=I_opt_tracker,
                              power_consumption_over_time=P_opt_tracker,
                              reliability=system_status.reliability_over_time,
                              t_sim=t_sim,
                              V=V,
                              reliability_mask=system_status.prediction_mask,
                              network_topology=channel_gain_matrix,
                              eta=eta,
                              prediction_set_cardinality=prediction_set_cardinality_tracker,
                              queue_tracker=queue_tracker,
                              alpha_tracker=alpha_tracker,
                              accuracy_virtual_queue_tracker=accuracy_virtual_queue_tracker,
                              error_probability_tracker=error_probability_tracker,
                              real_time_variance_tracker=real_time_variance_tracker,
                              q_hat_tracker=q_hat_tracker,
                              prediction_set_cardinality_virtual_queue_tracker = prediction_set_cardinality_virtual_queue_tracker,
                              energy_virtual_queue_tracker=energy_virtual_queue_tracker,
                              reliability_per_frame = reliability_tracker,
                              average_reliability = system_status.average_reliability_per_frame,
                              fnr_virtual_queue=fnr_virtual_queue_tracker,
                              average_reliability_over_multiple_decisions=system_status.average_reliability_per_frame_over_multiple_predictions,
                              alpha=args.alpha,
                              theta1=args.fixed_threshold,
                              gamma=args.gamma,
                              frame_prediction_mask = system_status.frame_prediction_mask
                              )

        stop_time = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
        print(f'Simulation over: {stop_time}')

        if system_status.enable_prediction_set_cardinality_control==True:
            out_dir = f'{data_output_dir}/simulation_{str(float(eta))}_window_size_{a_crc_refresh_freq}_{n_rel+init_sim_index}.pkl'

        with open(out_dir, 'wb') as out_file:
            pickle.dump(simulation, out_file, pickle.HIGHEST_PROTOCOL)














