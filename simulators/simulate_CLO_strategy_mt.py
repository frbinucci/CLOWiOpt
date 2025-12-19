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
from optimizers.optimization_solver import solve_instantaneous_opt_problem
from utils.simulation_results import Simulation
from utils.system_status import SystemStatus
import time

import warnings

# Disable all warnings
warnings.filterwarnings("ignore")



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

    np.random.seed(512)
    torch.manual_seed(512)
    parser = argparse.ArgumentParser(description='Optional app description')

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
    parser.add_argument("--enable_plotting",type=bool,default=True,help='Enable virtual queue dynamical plotting')
    parser.add_argument("--number_of_realizations",type=int,default=1,help='Number of sim realizations')
    parser.add_argument("--gamma",type=float,default=5e-1,help="OCRC step size")
    parser.add_argument("--alpha",type=float,default=0.12,help="Reliability Constraint")
    parser.add_argument("--fixed_threshold",type=float,default=5e-1,help='Threshold')
    parser.add_argument("--min_arrival_rate",type=float,default=0.4,help="Minimum arrival rate")
    parser.add_argument("--max_arrival_rate",type=float,default=0.8,help="Maximum arrival rate")
    parser.add_argument("--arrival_rate_refresh_rate",type=float,default=100,help="Arrival rate stationary time")
    parser.add_argument("--maximum_transmit_power",type=float,default=3.5,help="Maximum TX power per node")
    parser.add_argument("--path_loss",type=float,default=1e-9,help="Path loss for each link")


    args, unknown = parser.parse_known_args()

    t_sim = args.t_sim
    energy_constraint = args.energy_constraint
    V = args.V
    eta = args.eta
    tau = args.tau
    a_crc_refresh_freq = args.window_length
    data_output_dir = args.data_output_dir
    p_err = args.error_probability_constraint
    n_realizations = args.number_of_realizations
    arrival_rate_refresh_rate = args.arrival_rate_refresh_rate
    min_a = args.min_arrival_rate
    max_a = args.max_arrival_rate
    p_max = args.maximum_transmit_power
    path_loss = args.path_loss

    print(f'Torch seed: {torch.initial_seed()}')



    for n_rel in range(1,n_realizations+1):

        if not os.path.exists(data_output_dir):
            os.makedirs(data_output_dir)

        N_nodes = 4
        N_users = 3
        adj_matrix = np.zeros((N_nodes,N_nodes))
        bandwidth_matrix = np.zeros((N_nodes,N_nodes))

        adj_matrix[0,3] = path_loss
        adj_matrix[1,3] = path_loss
        adj_matrix[2,3] = path_loss

        channel_gain_matrix = generate_network(adj_matrix,t_sim)
        #Optimization_setting
        tx_power_array = p_max*np.ones(N_nodes)
        router_queues_matrix = np.zeros((t_sim,N_nodes*N_users))
        users_queues_matrix = np.zeros((t_sim,N_nodes))
        data_unit_bits = np.array([32,32,32])*(256*256*3)
        N0 = 3.98e-21
        user_indexes = np.array([0,1,2])
        arrival_rates = min_a*np.ones(N_users)

        transmission_constraints_matrix = 3*np.ones((N_nodes,N_nodes))
        computational_constraints_matrix = 3*np.ones(N_nodes)
        bandwidth = 20e6

        #Learning Models
        learning_models = {}
        cardinality_predictors = {}
        deciding_nodes = np.array([0,1,2,3])#,4,5,6])
        node_mapping = {0: 0, 1: 0, 2: 0, 3: 1}
        models_association = {0:0,1:0,2:0,3:3}#,4:2,5:2,6:3}
        for n_layers in deciding_nodes:
            if not models_association[n_layers]==None:
                network = smp.Unet(
                    encoder_name=ENCODER_MAP[models_association[n_layers]],
                    encoder_weights=ENCODER_WEIGHTS,
                    in_channels=3,
                    classes=1,
                )
                network.load_state_dict(torch.load(
                    f'./learning_models/D{models_association[n_layers]}/segmentation_network'))
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
                    f'./precision_predictors/D{models_association[n_layers]}/precision_predictor'))
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
        system_status.gamma = args.gamma*np.ones(N_users)
        system_status.reliability_constraints = args.alpha*np.ones(N_users)
        system_status.alpha_array = args.fixed_threshold*np.ones(N_users)
        system_status.dynamic_error_tracker = np.zeros(N_users)
        system_status.dynamic_prediction_counter = np.zeros(N_users)

        system_status.enable_accuracy_control = False
        system_status.enable_maximum_accuracy = False
        system_status.enable_prediction_set_cardinality_control = True
        system_status.enable_crc = True
        system_status.enable_prediction_set_cardinality_constraint = False
        system_status.enable_minimum_p_set_control = False

        system_status.energy_step_size = 1
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
        std_virtual_queue_tracker =  np.zeros((t_sim,N_users))
        std_virtual_queues  = np.zeros(N_users)

        print(f"Simulating eta ={eta}...")
        print("\n")
        print(f"Reliability Constraint: {args.alpha}")
        print(f'Theta start: {args.fixed_threshold}')
        print(f'Gamma: {args.gamma}')
        print(f'V: {args.V}')
        print("\n")
        system_status.init_queues()
        state = True

        model_list = ['D1','D2','D3']

        print(f'Data output dir :{data_output_dir}, eta = {eta}, S = {a_crc_refresh_freq}')
        start_time = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
        print(f'Simulation start: {start_time}')

        for t in range(0,t_sim):
            print_percentage_bar(t, t_sim)

            if t % arrival_rate_refresh_rate == 0:
                r = np.random.rand()
                if r < 0.5:
                    arrival_rates = min_a*np.ones(N_users)
                else:
                    arrival_rates = max_a*np.ones(N_users)

            R_opt,I_opt,P_opt,flag,value = solve_instantaneous_opt_problem(system_status)

            arrivals = np.random.binomial(1,arrival_rates)

            if flag==False:
                if (t + 1) % a_crc_refresh_freq == 0:
                    system_status.update_crc_param=True



                prediction_set_cardinality, tmp_error_tracker = system_status.adaptive_crc(I_opt)

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

                P_opt_tracker[t,:] = P_opt
                R_opt_tracker[t,:] = R_opt
                queue_tracker[t,:] = system_status.node_queues
                alpha_tracker[t,:] = system_status.alpha_array
                q_hat_tracker[t,:] = system_status.q_hat_array
                energy_virtual_queue_tracker[t,:] = system_status.energy_virtual_queue
                prediction_set_cardinality_virtual_queue_tracker[t,:] = system_status.prediction_set_cardinality_virtual_queue

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
                              average_reliability_over_multiple_decisions=system_status.average_reliability_per_frame_over_multiple_predictions,
                              alpha=args.alpha,
                              theta1=args.fixed_threshold,
                              gamma=args.gamma,
                              frame_prediction_mask = system_status.frame_prediction_mask,
                              N_users=N_users,
                              N_nodes=N_nodes,
                              deciding_nodes = deciding_nodes,
                              depth_mapping = node_mapping,
                              relibaility_min=system_status.reliability_min,
                              alpha_max=system_status.alpha_max
                              )

        stop_time = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
        print(f'\nSimulation over: {stop_time}')

        if system_status.enable_prediction_set_cardinality_control==True:
            out_dir = f'{data_output_dir}/simulation_{str(float(eta))}_window_size_{a_crc_refresh_freq}_{n_rel}.pkl'

        with open(out_dir, 'wb') as out_file:
            pickle.dump(simulation, out_file, pickle.HIGHEST_PROTOCOL)














