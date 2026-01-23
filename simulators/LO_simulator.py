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
from simulators.system_status import SystemStatus
from tools import *
import time

import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

from pathlib import Path

if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument("--config", type=str, default="configs/CLO-config.yaml")
    parser.add_argument("--data_output_dir",type=str,default=None,help='Directory in which save sim results')
    parser.add_argument("--eta",type=float,default=None,help='The value of the Penalty Parameter for the prediction-set cardinality')
    parser.add_argument("--V",type=float,default=None,help='The value of the Penalty Parameter associated to the energy consumption')
    parser.add_argument("--window_length",type=int,default=None,help='Window length for adaptive_crc')
    parser.add_argument("--t_sim",type=int,default=None,help='Simulation duration')
    parser.add_argument("--tau",type=float,default=None,help='Time-slot duration')
    parser.add_argument("--simulation_index",type=int,default=0,help='Simulation Index (for average simulations)')
    parser.add_argument("--enable_plotting",type=bool,default=False,help='Enable virtual queue dynamical plotting')
    parser.add_argument("--number_of_realizations",type=int,default=1,help='Number of sim realizations')
    parser.add_argument("--gamma",type=float,default=None,help="OCRC step size")
    parser.add_argument("--alpha",type=float,default=None,help="Reliability Constraint")
    parser.add_argument("--fixed_threshold",type=float,default=None,help='Threshold')
    parser.add_argument("--min_arrival_rate",type=float,default=None,help="Minimum arrival rate")
    parser.add_argument("--max_arrival_rate",type=float,default=None,help="Maximum arrival rate")
    parser.add_argument("--arrival_rate_refresh_rate",type=float,default=None,help="Arrival rate stationary time")
    parser.add_argument("--init_seed",type=int,default=None,help="Simulation seed")
    parser.add_argument("--init_sim_index",type=int,default=None,help='Init sim. index')

    parser.add_argument("--dataset_path",type=str,default=None,help='Dataset path')



    #Network Constraints
    parser.add_argument("--transmission_constraints",type=int,default=1,help='Transmission Constraints (common for all the users)')
    parser.add_argument("--computational_constraints",type=int,default=3,help='Computational Constraints (common for all the nodes)')
    parser.add_argument("--user_indexes",type=list,default=None,help='User Indexes')


    #Lyapunov Optimization Parameteres
    parser.add_argument("--fnr_step_sizes",type=float,default=None,help='FNR step size')
    parser.add_argument("--fnr_virtual_queues_init",type=float,default=None,help='FNR virtual queues init')


    #Learning task features
    parser.add_argument('--data_unit_bits',type=int,default=None,help='Bit per pixel (images encoding)')
    parser.add_argument('--image_size',type=int,default=None,help='Images size (n_ch,width,height)')


    #Transmission parameters
    parser.add_argument("--pmax",type=float,default=None,help='Maximum transmit power')
    parser.add_argument("--Bw",type=float,default=None,help='Transmission Bandwidth')

    parser.add_argument("--N0",type=float,default=None,help='Noise Power Spectral Density')

    args, unknown = parser.parse_known_args()
    cfg = load_cfg(args.config)

    # override semplici (se forniti)
    if args.eta is not None: cfg["optimizer"]["eta"] = args.eta
    if args.V is not None: cfg["optimizer"]["V"] = args.V
    if args.t_sim is not None: cfg["simulation"]["t_sim"] = args.t_sim
    if args.tau is not None: cfg["simulation"]["tau"] = args.tau
    if args.window_length is not None: cfg["reliability"]["window_length"] = args.window_length
    if args.number_of_realizations is not None: cfg["simulation"]["n_realizations"] = args.number_of_realizations
    if args.min_arrival_rate is not None: cfg["traffic"]["min_arrival_rate"] = args.min_arrival_rate
    if args.max_arrival_rate is not None: cfg["traffic"]["max_arrival_rate"] = args.max_arrival_rate
    if args.gamma is not None: cfg["reliability"]["gamma"] = args.gamma
    if args.alpha is not None: cfg["reliability"]["alpha"] = args.alpha
    if args.init_seed is not None: cfg["simulation"]["seed"] = args.init_seed
    if args.init_sim_index is not None: cfg["simulation"]["init_sim_index"] = args.init_sim_index
    if args.fnr_virtual_queues_init is not None: cfg["reliability"]["fnr_virtual_queues_init"] = args.fnr_virtual_queues_init

    if args.fnr_step_sizes is not None: cfg["reliability"]["fnr_step_sizes"] = args.fnr_step_sizes

    if args.data_unit_bits is not None: cfg["task"]["data_unit_bits"] = args.data_unit_bits
    if args.image_size is not None: cfg["task"]["image_size"] = args.image_size

    if args.N0 is not None: cfg["network"]["N0"] = args.N0
    if args.pmax is not None: cfg["network"]["pmax"] = args.pmax
    if args.Bw is not None: cfg["network"]["Bw"] = args.Bw
    if args.transmission_constraints is not None: cfg["network"]["transmission_constraints"] = args.transmission_constraints
    if args.computational_constraints is not None: cfg["network"]["computational_constraints"] = args.computational_constraints




    if args.fixed_threshold is not None: cfg["threshold"]["fixed_index"] = args.fixed_threshold
    if args.data_output_dir is not None: cfg["paths"]["data_output_dir"] = args.data_output_dir

    if args.user_indexes is not None: cfg["network"]["user_indexes"] = args.user_indexes

    if args.dataset_path is not None: cfg["path"]["dataset_path"] = args.dataset_path


    V = cfg["optimizer"]["V"]
    eta = cfg["optimizer"]["eta"]
    tau = cfg["simulation"]["tau"]
    a_crc_refresh_freq = cfg["reliability"]["window_length"]
    data_output_dir = cfg["paths"]["data_output_dir"]

    n_realizations = cfg["simulation"]["n_realizations"]
    t_sim = cfg["simulation"]["t_sim"]
    arrival_rate_refresh_rate = cfg["reliability"]["window_length"]
    min_a = cfg["traffic"]["min_arrival_rate"]
    max_a = cfg["traffic"]["max_arrival_rate"]
    init_seed = cfg["simulation"]["seed"]
    init_sim_index = cfg["simulation"]["init_sim_index"]

    np.random.seed(init_seed)
    torch.manual_seed(init_seed)

    gamma = cfg["reliability"]["gamma"]
    init_queues = cfg["reliability"]["fnr_virtual_queues_init"]

    data_unit_bits = cfg["task"]["data_unit_bits"]
    image_size = cfg["task"]["image_size"]

    p_max = cfg["network"]["pmax"]
    N0 = cfg["network"]["N0"]
    bandwidth = cfg["network"]["Bw"]
    user_indexes = cfg["network"]["user_indexes"]

    ENCODER_MAP = {int(k): v for k, v in cfg["models"]["encoder_map"].items()}
    ENCODER_PFR_PREDICTOR_MAP = {int(k): v for k, v in cfg["models"]["encoder_pfr_predictor_map"].items()}
    ENCODER_WEIGHTS = cfg["models"]["encoder_weights"]
    DEVICE = cfg["models"]["device"]

    depth_mapping = np.array(cfg["models"]["depth_mapping"])
    models_association = {int(k): v for k, v in cfg["models"]["models_association"].items()}
    model_list = cfg["models"]["model_list"]

    fnr_step_sizes = cfg["reliability"]["fnr_step_sizes"]
    alpha = cfg["reliability"]["alpha"]

    dataset_path = cfg["paths"]["dataset_path"]

    for n_rel in range(1,n_realizations+1):

        if not os.path.exists(data_output_dir):
            os.makedirs(data_output_dir)

        N_nodes,N_users,adj_matrix,bandwidth_matrix = build_matrices(cfg)


        channel_gain_matrix = generate_network(adj_matrix,t_sim)

        #Optimization_setting
        tx_power_array = p_max*np.ones(N_nodes)
        router_queues_matrix = np.zeros((t_sim,N_nodes*N_users))
        users_queues_matrix = np.zeros((t_sim,N_nodes))
        data_unit_bits = np.array([data_unit_bits,data_unit_bits,data_unit_bits])*image_size
        user_indexes = np.array([0,1,2])
        arrival_rates = min_a*np.ones(N_users)

        transmission_constraints_matrix = cfg["network"]["transmission_constraints"]*np.ones((N_nodes,N_nodes))
        computational_constraints_matrix = cfg["network"]["computational_constraints"]*np.ones(N_nodes)

        #Learning Models
        learning_models = {}
        cardinality_predictors = {}
        #depth_mapping = np.array([0,1,2,3])
        #models_association = {0:0,1:0,2:0,3:3}

        seg_tpl = cfg["paths"]["segmentation_ckpt_tpl"]
        pred_tpl = cfg["paths"]["predictor_ckpt_tpl"]

        for n_layers in depth_mapping:
            ckpt_seg = seg_tpl.format(d=models_association[n_layers])
            ckpt_pred = pred_tpl.format(d=models_association[n_layers])
            if not models_association[n_layers]==None:
                network = smp.Unet(
                    encoder_name=ENCODER_MAP[models_association[n_layers]],
                    encoder_weights=ENCODER_WEIGHTS,
                    in_channels=3,
                    classes=1,
                )
                network.load_state_dict(torch.load(
                    ckpt_seg))
                network.eval()
                network = network.to('cuda')
            else:
                network=None
            learning_models[n_layers] = network
            if not models_association[n_layers]==None:
                learning_models[n_layers] = network
                cardinality_predictor = smp.PSPNet(
                    encoder_name=ENCODER_PFR_PREDICTOR_MAP[models_association[n_layers]],
                    encoder_weights=ENCODER_WEIGHTS,
                    in_channels=3,
                    classes=1,
                )
                cardinality_predictor.load_state_dict(torch.load(ckpt_pred))
                cardinality_predictor = cardinality_predictor.to('cuda')
                cardinality_predictor.eval()
            else:
                cardinality_predictor = None
            cardinality_predictors[n_layers] = cardinality_predictor

        system_status = SystemStatus(t_sim=t_sim,N_users=N_users,window_size=a_crc_refresh_freq,dataset_path=dataset_path)

        system_status.cardinality_surrogate = cardinality_predictors
        system_status.learning_models = learning_models

        for i in range(0,bandwidth_matrix.shape[0]):
            for j in range(0,bandwidth_matrix.shape[1]):
        #        bandwidth_matrix[i,j]=bandwidth*(adj_matrix[i,j]>0)
                transmission_constraints_matrix[i,j] = transmission_constraints_matrix[i,j]*(adj_matrix[i,j]>0)

        print(bandwidth_matrix)
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
        system_status.learning_models = learning_models
        system_status.gamma = fnr_step_sizes*np.ones(N_users)
        system_status.reliability_constraints = alpha*np.ones(N_users)
        system_status.alpha_array = init_queues*np.ones(N_users)
        system_status.dynamic_error_tracker = np.zeros(N_users)
        system_status.dynamic_prediction_counter = np.zeros(N_users)


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
        fnr_virtual_queues = init_queues*np.ones(N_users)
        std_virtual_queue_tracker =  np.zeros((t_sim,N_users))
        std_virtual_queues  = np.zeros(N_users)
        fnr_step_sizes = fnr_step_sizes*np.ones(N_users)

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

        #Saving simulation
        out_dir = f'{data_output_dir}/simulation_{str(float(eta))}_window_size_{a_crc_refresh_freq}_{n_rel+init_sim_index}.pkl'
        with open(out_dir, 'wb') as out_file:
            pickle.dump(simulation, out_file, pickle.HIGHEST_PROTOCOL)














