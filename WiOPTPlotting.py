import os, sys

import pickle
from collections import Counter
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import rcParams
from segmentation_models_pytorch.utils.functional import precision
from scipy.io import savemat
#from utils.simulation_results import Simulation
import utils.simulation_results as sr
sys.modules["simulation_results"] = sr
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.size'] = 14
rcParams['legend.fontsize'] = "medium"
rcParams['axes.grid'] = True

def plot_time_to_reliability(path_to_load,eta_array,window_array,t_sim,**kwargs):
    label_array = ['LO','CLO']
    output_dir = kwargs.get('output_dir','./matlab_plotting')
    plot_per_rel = kwargs.get('plot_per_rel',False)
    constraint = kwargs.get('constraint',1e-1)
    theta1 = kwargs.get('theta1',5e-1)
    n_rel = kwargs.get('n_rel',7)

    M = kwargs.get('M',1)

    enable_label_plotting = kwargs.get('enable_label_plotting',False)
    enable_bound_plot=kwargs.get('bound',False)
    gamma = kwargs.get('gamma',1)
    trans_end = kwargs.get('trans_end',0)
    frame_flag = kwargs.get('frame_flag',True)



    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    i=0

    #mus = np.arange(40, 100, 10)
    #cmap = plt.cm.Set1

    color_array = ['tab:blue','tab:green','tab:orange']

    avg_outage_prob = 0
    avg_overshoot = 0

    avg_reliability = np.zeros((int(n_rel),int(t_sim)))

    alpha_min = +np.inf
    alpha_min_avg = 0
    if not isinstance(path_to_load, list):
        path_to_load = list(path_to_load)
    p=0
    for path in path_to_load:
        for r in range(1,n_rel+1):
            for eta in eta_array[p]:
                i=0
                for window in window_array[p]:
                    with open(fr"{path}/simulation_{float(eta)}_window_size_{window}_{r}.pkl", "rb") as input_file:
                        e = pickle.load(input_file)
                        gamma = e.gamma
                        if e.theta1 is not None:
                            theta1 = e.theta1

                        constraint = e.alpha

                        if enable_bound_plot==True:
                            gamma = e.gamma
                            theta1 = e.theta1

                        if frame_flag==True:

                            reliability_array = np.mean(e.average_reliability_over_multiple_decisions,axis=0)
                        else:
                            reliability_array = np.mean(e.average_reliability,axis=0)


                        F = int(t_sim/window)

                        if frame_flag==True:
                            raxis=np.linspace(start=0,stop=F-1,num=F)

                        reliability_array = np.repeat(reliability_array,window)


                        upper_bound = np.divide(M  - theta1, gamma * raxis)
                        avg_reliability[r - 1, :] = reliability_array

                        if plot_per_rel == True:
                            if enable_label_plotting==True:
                                plt.plot(reliability_array,label=label_array[p])#,color=color,linewidth=2)
                            else:
                                plt.plot(reliability_array)


                            plt.plot(np.repeat(upper_bound + constraint, window), linestyle='dashed')  # ,color=color)
                        else:
                            avg_reliability[r - 1, :] = reliability_array
                    i+=1
        p+=1

    if(output_dir!=None):
        np.save(f'{output_dir}/reliability_plot.npy',avg_reliability)

    if plot_per_rel==False:
        plt.plot(np.repeat(upper_bound + constraint, window), linestyle='dashed')  # ,color=color)
        avg = np.mean(avg_reliability,axis=0)
        std = np.std(avg_reliability,axis=0)
        plt.plot(avg,linewidth=2)
        plt.fill_between(np.linspace(start=1, stop=t_sim, num=t_sim), avg - std,
                         avg + std, color='g', alpha=0.6)

    plt.hlines(xmin=0,xmax=t_sim,y=constraint,linestyles='dashed',linewidth=2,color='b')
    plt.xlabel("Time Slot Index [t]",fontsize=14)
    plt.ylim([constraint-0.02,constraint+0.02])
    plt.xlim([0,t_sim])

    plt.ylabel("FNR",fontsize=14)
    plt.legend()
    plt.show()



def plot_queues(path_array,eta_array,window_array,t_sim):

    p=0
    for path_to_load in path_array:
        i = 0
        for eta in eta_array[p]:
            for window in window_array[p]:
                with open(fr"{path_to_load}/simulation_{float(eta)}_window_size_{window}_1.pkl", "rb") as input_file:
                    e = pickle.load(input_file)
                    queue_tracker = e.queue_tracker

                    plt.plot(np.mean(np.sum(queue_tracker,axis=1),axis=1),label=rf'$\eta = {eta}$')
                i+=1
        p+=1
    plt.xlabel("Time Slot Index [t]",fontsize=14)
    plt.ylabel("Virtual Queues",fontsize=14)
    plt.legend()
    plt.show()

def plot_virtual_queues(path_to_load,eta_array,window_length,t_sim):

    xx = [5,1]
    p=0
    for path in path_to_load:
        i = 0
        for eta in eta_array:
            with open(fr"{path}/simulation_{float(eta)}_window_size_{window_length}_1.pkl", "rb") as input_file:
                e = pickle.load(input_file)
                queue_tracker = e.fnr_outage_queues

                plt.plot(np.mean(queue_tracker,axis=1)[0:t_sim])

            i+=1
        p+=1
    plt.xlabel("Time Slot Index [t]",fontsize=14)
    plt.ylabel("Average virtual queues",fontsize=14)
    #plt.xlim([0,1000])
    plt.legend()
    plt.show()

def plot_energy_precision_trade_off(path_to_load,eta_array,window_size,t_sim,trans_end=0,**kwargs):

    matlab_out = kwargs.get('matlab_out','./matlab')

    realization_number = kwargs.get('realization_number',1)
    start_rel = kwargs.get('start_rel',5)

    if matlab_out is not None:
        if not os.path.exists(matlab_out):
            os.makedirs(matlab_out)



    trans_end_array = trans_end


    label_array = ['CLO','LO']
    p=0
    for path in path_to_load:
        i = 0
        for window in window_size[p]:
            energy_array = np.zeros(eta_array.shape[1])
            precision_array = np.zeros(eta_array.shape[1])
            j = 0
            for eta in eta_array[p]:
                precision_array[j]=0
                energy_array[j]=0
                for r in range(start_rel,start_rel+realization_number):
                    with open(fr"{path}/simulation_{float(eta)}_window_size_{int(window)}_{r}.pkl", "rb") as input_file:
                        e = pickle.load(input_file)
                        precision_tracker = e.prediction_set_cardinality
                        energy_tracker = e.power_consumption_over_time

                        precision_tracker = precision_tracker[trans_end_array[p,j]:,:]



                        precision_tracker = precision_tracker[precision_tracker>=0]


                        precision_array[j] += (1-np.mean(precision_tracker))
                        energy_array[j] += np.mean(np.sum(np.sum(energy_tracker[trans_end_array[p,j]:,:,:],axis=2),axis=1))
                precision_array[j]/=realization_number
                energy_array[j]/=realization_number
                j+=1


            plt.plot(energy_array*0.05*1000,precision_array,marker='o', label=rf'{label_array[p]} $S={window}$')
            if matlab_out is not None:
                np.save(f'./{matlab_out}/energy_array_{window}_{label_array[p]}.npy', energy_array * 1e3)
                np.save(f'./{matlab_out}/precision_array_{window}_{label_array[p]}.npy', precision_array)
            i+=1

        p+=1
    plt.legend()
    plt.xlabel('System Power Consumption [mW]')
    plt.ylabel('Precision')
    plt.show()



if __name__=="__main__":

    t_sim = 15000
    window_array = np.array([[10],[10]])
    trans_end_array = np.array([[14000,14000,14000,14000,14000,14000,13100],[14000,14000,14000,14000,14000,14000,14000]])
    eta_array = np.array([[0.25,0.3,0.4,0.42,0.6,0.6,0.8],[0.2,0.3,0.4,0.5,0.6,0.6,0.8]])

    print("Selection Menu")
    print("1 - Time to reliability")
    print("2 - Precision/Energy trade-off")
    print("4 - Plot queues")
    print("5 - Plot deciding node rates")

    choice = int(input("What do you want to plot? "))
    matlab_out = './matlab'

    path = ['./sim_res/WiOPTSimCLO15k/','./sim_res/WiOPTSimLyapunov15k/']

    if choice==1:
        method = int(input("Choose the method (1 - CLO, 2 - LO): "))
        if method == 1:
            path_to_load = f'{path[0]}/multiple_realizations'
        else:
            path_to_load = f'{path[1]}/multiple_realizations'
        n_rel = int(input("Select the number of realizations to plot: "))
        eta_array = np.array([[0.8]])
        plot_time_to_reliability([path_to_load], eta_array, window_array, t_sim,matlab_out=matlab_out,trans_end=0,n_rel=n_rel,B=1,b=1,m=(0),M=(1+0.5),plot_per_rel=True,output_dir='./matlab_plotting',frame_flag=True)
    elif choice==2:
        plot_energy_precision_trade_off(path, eta_array, window_array, t_sim,trans_end=trans_end_array,start_rel=1,realization_number=1)
    else:
        print("Not yet implemented...")

