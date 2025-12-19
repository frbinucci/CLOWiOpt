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

import cvxpy as cp
import numpy as np
import torch
from cvxpy import Problem, Minimize


def ptx(bandwidth_matrix,channel_status_matrix,N0,data_unit_bits,tau,i,j,k):
    return ((bandwidth_matrix[i,j]*N0)/channel_status_matrix[i,j])*(2**(data_unit_bits[k]/(tau*bandwidth_matrix[i,j]))-1)


def solve_instantaneous_opt_problem(system_status,virtual_queues,step_sizes,theta_it,path='./utils/LUTs',model_list = ['D0','D0','D0','D3'],optimization_type='nn'):

    N_nodes = system_status.N_nodes
    N_users = system_status.N_users
    N0 = system_status.N0
    channel_status_matrix = system_status.channel_status_matrix
    data_unit_bits = system_status.data_unit_bits

    theta_array = np.load(f'{path}/theta_lut.npy')

    std_fnr_lut = {}
    avg_fnr_lut = {}
    avg_fpr_lut = {}


    lut_dict = {}
    model_index = 0
    for model_name in model_list:
        if not model_name==None:
            std_lut=np.load(f'{path}/std_fnr_lut_{model_name}.npy')
            avg_lut=np.load(f'{path}/mean_fnr_lut_{model_name}.npy')
            avg_fpr = np.load(f'{path}/mean_fpr_lut_{model_name}.npy')
            theta_index = 0
            for theta in theta_array:
                std_fnr_lut[(model_index,theta)]=std_lut[theta_index]
                avg_fpr_lut[(model_index,theta)]=avg_fpr[theta_index]
                avg_fnr_lut[(model_index,theta)]=avg_lut[theta_index]
                theta_index+=1
        model_index+=1



    transmission_constraints = cp.Parameter(system_status.transmission_constraints.shape,name='transmission_constraints')
    transmission_constraints.value = system_status.transmission_constraints

    computational_constraints = system_status.computational_constraints
    eta = system_status.eta
    V = system_status.V
    cardinality_surrogate = system_status.cardinality_surrogate
    tau = system_status.tau
    bandwidth_matrix = cp.Parameter(system_status.bandwidth_matrix.shape,name='bandwidth_matrix')
    bandwidth_matrix.value = system_status.bandwidth_matrix
    p_max = cp.Parameter(len(system_status.p_max_array),name='p_max')
    p_max.value = system_status.p_max_array

    node_queues = system_status.virtual_node_queues

    differential_backlog = {}


    for i in range(0,N_nodes):
        differential_backlog[i] = np.zeros((N_nodes,N_users))
        for j in range(0,N_nodes):
            for k in range(0,N_users):
                if bandwidth_matrix.value[i,j]>0:
                    differential_backlog[i][j,k]=(node_queues[i,k]-node_queues[j,k])

    R = {}
    for i in range(N_nodes):
        R[i] = cp.Variable((N_nodes,N_users))


    p_tot = 0


    I = cp.Variable((N_nodes,N_users))
    constraints = list()

    #Objective function
    cost = 0


    for i in range(N_nodes):
        for j in range(N_nodes):
            if bandwidth_matrix.value[i, j] > 0:
                for k in range(N_users):
                    p_tot+=cp.sum(V*R[i][j,k]*ptx(bandwidth_matrix.value,channel_status_matrix, N0, data_unit_bits, tau, i, j,k))

    cost+=p_tot

    for i in range(0,N_nodes):
        for j in range(0,N_nodes):
            if bandwidth_matrix.value[i,j]>0:
                for k in range(0,N_users):
                    cost+=cp.sum(-R[i][j,k]*differential_backlog[i][j,k])

    for i in range(0,N_nodes):
        for j in range(0,N_users):
            cost+=cp.sum(-I[i,j]*node_queues[i,j])


    if optimization_type=='lut':
        for i in range(0,N_nodes):
            for k in range(0,N_users):
                if (i,k) in system_status.data_units_queues:
                    if system_status.data_units_queues[(i, k)]:
                        if (i, theta_it) in avg_fpr_lut:
                            cost+=cp.sum(I[i,k]*eta*avg_fpr_lut[(i,theta_it)])
    else:
        for i in range(0,N_nodes):
            for k in range(0,N_users):
                if (i,k) in system_status.data_units_queues:
                    if system_status.data_units_queues[(i, k)]:
                        if not system_status.cardinality_surrogate[i]==None:
                            data,labels = system_status.data_units_queues[(i, k)][0]
                            data = data.to('cuda')
                            labels = labels.to('cuda')
                            alpha = torch.unsqueeze(torch.tensor(theta_it),axis=0).to('cuda')
                            predicted_mask = system_status.cardinality_surrogate[i](data)
                            predicted_mask = predicted_mask>alpha
                            predicted_card = system_status.calculate_fpr(predicted_mask.squeeze(),labels.squeeze())
                            cost+=cp.sum(I[i,k]*predicted_card*eta)

    for i in range(0,N_nodes):
        for k in range(0,N_users):
            if (i,k) in system_status.data_units_queues:
                if system_status.data_units_queues[(i, k)]:
                    if (i,theta_it) in avg_fnr_lut:
                        cost+=cp.sum(I[i,k]*step_sizes[k]*virtual_queues[k]*avg_fnr_lut[(i,theta_it)])



    #Computational Constraints
    constraints.append(cp.sum(cp.abs(I),axis=1)<=computational_constraints)
    #constraints.append(cp.sum(cp.abs(I),axis=0)<=1)


    #Maximum power constraint
    for i in range(N_nodes):
        ptot = 0
        for j in range(N_nodes):
            if bandwidth_matrix.value[i, j] > 0:
                for k in range(N_users):
                    ptot+=cp.sum(R[i][j,k]*ptx(bandwidth_matrix.value,channel_status_matrix, N0, data_unit_bits, tau, i, j,k))
        constraints.append(ptot<=p_max[i])

    #Link transmission constraints
    for i in range(0,N_nodes):
        tx_constraint = 0
        for j in range(0,N_nodes):
            if bandwidth_matrix.value[i, j] > 0:
                for k in range(0,N_users):
                    tx_constraint +=R[i][j,k]
                constraints.append(tx_constraint<=transmission_constraints[i,j])


    # #constraints.append(cp.sum(cp.sum(cp.abs(R),axis=0),axis=0)<=N_nodes*N_users)
    constraints.append(0<=I)
    constraints.append(I<=1)
    #constraints.append(I<=node_queues)


    for i in range(N_nodes):
        for k in range(N_users):
            queue_feasibility_constraint = 0
            for j in range(N_nodes):
                if bandwidth_matrix.value[i, j] > 0:
                    queue_feasibility_constraint+=R[i][j,k]
            constraints.append(queue_feasibility_constraint+I[i,k]<=node_queues[i,k]-system_status.place_holder)

    #
    for i in range(0,N_nodes):
        for j in range(0,N_nodes):
            if bandwidth_matrix.value[i,j]>0:
                for k in range(0,N_users):
                    constraints.append(R[i][j,k]>=0)
                    constraints.append(R[i][j,k]<=1)
            else:
                for k in range(0,N_users):
                    constraints.append(R[i][j, k] == 0)



    prob = Problem(Minimize(cost),constraints)

    flag = False
    try:
        prob.solve(solver=cp.MOSEK,verbose=False)
    except NameError:
        print(NameError)
        flag=True
        print("Optimization_ERROR!")


    R_matrix = np.zeros((N_nodes,N_nodes,N_users))
    P = np.zeros((N_nodes,N_nodes))

    if flag==False:
        I_opt = I.value
        for i in range(N_nodes):
            for j in range(N_nodes):
                transmitted_bits = 0
                if bandwidth_matrix.value[i,j]>0:
                    for k in range(N_users):
                        R_matrix[i,j,k]=R[i][j,k].value
                        if R_matrix[i,j,k]==None:
                            R_matrix[i,j,k]=0
                        transmitted_bits += R_matrix[i,j,k]*data_unit_bits[k]
                        P[i,j] += R_matrix[i,j,k]*ptx(bandwidth_matrix.value,channel_status_matrix, N0, data_unit_bits, tau, i, j,k)
    else:
        I_opt = np.zeros((N_nodes,N_users))


    #print(np.max(R_matrix))
    return np.round(R_matrix).astype(int),np.round(I_opt),P,flag,prob.value





