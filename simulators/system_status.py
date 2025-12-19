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

import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.dataset import CarSegmentationDataset


class SystemStatus:
    def __init__(self,**kwargs):


        self.t_sim = kwargs.get('t_sim',500)
        self.N0 = kwargs.get('N0',3.98e-21)
        self.N_nodes = kwargs.get('N_nodes',None)
        self.N_users = kwargs.get('N_users',None)
        self.channel_status_matrix = kwargs.get('channel_status_matrix',None)
        self.user_queues = kwargs.get('user_queues',None)
        self.router_queues = kwargs.get('router_queues',None)
        self.data_unit_bits = kwargs.get('data_unit_bits',None)
        self.transmission_constraints = kwargs.get('transmission_constranints',None)
        self.computational_constraints = kwargs.get('computational_constraints',None)
        self.eta = kwargs.get('eta',None)
        self.V = kwargs.get('V',None)
        self.tau = kwargs.get('tau',None)
        self.bandwidth_matrix = kwargs.get('bandiwidth_matrix',None)
        self.p_max = kwargs.get('p_max',None)
        self.node_queues = kwargs.get('node_queues',None)
        self.cardinality_surrogate = kwargs.get('cardinality_surrogate',None)
        self.user_indexes = kwargs.get('user_indexes',None)
        self.data_units_queues = kwargs.get('data_unit_queues',list())
        self.alpha_array = kwargs.get('alpha_array',None)
        self.update_crc_param = kwargs.get('update_crc_param',False)
        self.dynamic_error_tracker = kwargs.get('dynamic_error_tracker',None)
        self.dynamic_prediction_counter = kwargs.get('dynamic_predition_counter',None)
        self.reliability_constraints = kwargs.get('reliability_constraint',None)
        self.window_size = kwargs.get('window_size',100)

        self.fnr_track_window = kwargs.get('fnr_track_window',1000)
        self.fnr_tracker = np.zeros((self.fnr_track_window,self.N_users))
        self.dynamic_prediction_mask = np.zeros((self.fnr_track_window,self.N_users))

        self.prediction_mask = np.zeros((self.N_users,self.t_sim))
        self.reliability_over_time = np.zeros((self.N_users,self.t_sim))
        self.gamma = kwargs.get('gamma',None)
        self.learning_models = kwargs.get('learning_models_old', None)
        self.time_slot_index = 0
        self.dataset_iterator = kwargs.get('dataset_iterator',None)
        self.dataset_loader = kwargs.get('dataset_loader',None)
        self.error_probability_over_time = -1*np.ones((self.N_users,self.t_sim))
        self.accuracy_virtual_queue = np.zeros((self.N_users))
        self.prediction_set_cardinality_virtual_queue = np.zeros((self.N_users))
        self.accuracy_constraint = kwargs.get('accuracy_constraint',None)
        self.accuracy_step_size = kwargs.get('accuracy_step_size',None)

        self.enable_accuracy_control = kwargs.get('enable_accuracy_control',True)
        self.enable_prediction_set_cardinality_control = kwargs.get('enable_prediction_set_cardinality_control',False)
        self.error_counter = np.zeros(self.N_users)
        self.prediction_counter = np.zeros(self.N_users)
        self.error_probability = np.zeros(self.N_users)
        self.crc_assisted_prediction = kwargs.get('enable_self_crc_assisted_prediction',False)
        self.non_conformity_scores_list = list()
        self.error_probability_window_size = kwargs.get('error_probability_window_size',100)
        self.error_probability_per_frame = kwargs.get('error_probability_frame',np.zeros(self.N_users))
        self.real_time_variance = np.zeros((self.N_users))
        self.real_time_average_list = list()
        self.window_error = np.zeros((self.N_users,self.t_sim))-1
        self.real_time_average = np.zeros((self.N_users))
        self.p_err_window_size = kwargs.get('p_err_window_size',10)
        self.q_hat_array = np.zeros(self.N_users)
        self.avg_fpr_mean = np.zeros(self.N_users)
        self.avg_fpr_frame_mean = np.zeros(self.N_users)
        self.enable_prediction_set_cardinality_constraint = kwargs.get('enable_prediction_set_cardinality_constraint',False)

        self.enable_crc = kwargs.get('enable_crc',True)
        self.prediction_set_cardinality_constraint = kwargs.get('prediction_set_cardinality_constraint',[1,1,1])

        self.prediction_set_cardinality_step_size = kwargs.get('prediction_set_cardinality_step_size',[1,1,1])
        self.energy_virtual_queue = 0
        self.energy_constraint = kwargs.get('energy_constraint',None)
        self.overall_energy_consumption = 0
        self.enable_minimum_p_set_control = kwargs.get('enable_minimum_p_set_control',True)
        self.energy_step_size = 100
        self.task_type=kwargs.get('task_type','image_segmentation')

        self.prediction_per_frame = kwargs.get('prediction_per_frame',50)
        self.alpha_max = np.zeros(self.N_users)

        self.n_frames = int(self.t_sim/self.window_size)
        self.frame_index = 0
        self.reliability_per_frame = np.zeros((self.N_users,self.n_frames))
        self.average_reliability_per_frame = np.zeros((self.N_users,self.n_frames))
        self.avg_mean = np.zeros(self.N_users)
        self.avg_frame_mean = np.zeros(self.N_users)
        self.average_reliability_per_frame_over_multiple_predictions = np.zeros((self.N_users,self.n_frames))
        self.precision_tracker=np.zeros(self.N_users)
        self.slot_precision=np.zeros(self.N_users)
        self.slot_fnr = np.zeros(self.N_users)
        self.frame_prediction_mask = np.zeros((self.N_users,self.n_frames))
        self.alpha_min = np.ones(self.N_users)*np.inf
        self.reliability_min = +np.ones(self.N_users)*np.inf
        self.prob_logit = kwargs.get('prob_logit',True)


        self.enable_reliability_tracking = True

        self.virtual_node_queues = kwargs.get('virtual_node_queues',None)
        self.place_holder = kwargs.get('place_holder',0)

        self.dataset_path = kwargs.get('dataset_path',None)

        for n in range(self.N_users):
            self.non_conformity_scores_list.append(list())
        for n in range(self.N_users):
            self.real_time_average_list.append(list())


    def update(self,R,I,A,channel_status_matrix):
        if self.dataset_loader == None:
            self.init_iterator()
        #Updating queues
        for i in range(self.N_nodes):
            for k in range(self.N_users):
                arrivals = 0
                ki = k
                if i in range(self.N_users):
                    arrivals = A[i]
                    ki = i
                    #I[i,ki]=0
                self.node_queues[i,ki] = max(self.node_queues[i,ki]-np.sum(R[i,:,ki])-I[i,ki],0)+arrivals+np.sum(R[:,i,ki])
                self.virtual_node_queues[i,ki] = self.node_queues[i,ki] + self.place_holder
                if i in range(self.N_users):
                    break

        i=0
        for arrival in A:
            if arrival == 1:
                try:
                    inputs, labels = next(self.dataset_iterator)
                except:
                    self.init_iterator()
                    inputs,labels= next(self.dataset_iterator)
                self.data_units_queues[(i,i)].append((inputs,labels))
            i+=1

        #Updating actual queues
        i=0
        for row in R:
            j=0
            break_flag = False
            for cell in row:
                k=0
                for element in cell:
                    ki = k
                    if i in range(self.N_users):
                        ki=i
                        break_flag = True
                    #print(f'R{(i,j,ki)}= {element}')
                    if R[i,j,ki]==1 and self.data_units_queues[(i,ki)]:
                        next_data_unit = self.data_units_queues[(i,ki)].pop(0)
                        self.data_units_queues[(j,ki)].append(next_data_unit)
                    if break_flag==True:
                        break
                    k+=1
                j+=1
            i+=1

        self.channel_status_matrix = channel_status_matrix
        self.time_slot_index+=1

    def init_actual_queues(self,new_data_units):
        self.init_iterator()
        i=0
        for arrival in new_data_units:
            if arrival == 1:
                try:
                    input, labels = next(self.dataset_iterator)
                except:
                    self.init_iterator()
                    inputs,labels= next(self.dataset_iterator)
                self.data_units_queues[(i,i)].append((input,labels))
                self.node_queues[i,i]=1
            i+=1

    def adaptive_crc(self,I):
        i=0
        self.slot_fnr = np.zeros(self.N_users)
        for row in I:
            k=0
            break_flag = False
            for cell in row:
                ki = k
                if i in range(self.N_users):
                    ki = i
                    break_flag = True
                    self.slot_precision[ki] = -1
                if I[i,ki]==1 and self.data_units_queues[(i,ki)]:
                    next_data_unit,next_data_label = self.data_units_queues[(i,ki)].pop(0)
                    #Inference phase + Conformal Prediction
                    output = self.learning_models[i](next_data_unit.to('cuda'))

                    self.prediction_counter[ki]+=1
                    #cal_smx = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy().squeeze()

                    if self.enable_crc==True or self.enable_reliability_tracking==True:
                        if self.task_type=='image_classification':
                            non_conformity_score = 1 - cal_smx[next_data_label[0]]
                            self.non_conformity_scores_list[ki].append(non_conformity_score)

                            quantile_array = np.array(self.non_conformity_scores_list[ki])

                            self.alpha_array[ki] = max(0, self.alpha_array[ki])
                            self.alpha_array[ki] = min(1, self.alpha_array[ki])
                            current_threshold = np.quantile(quantile_array, 1 - self.alpha_array[ki])
                            self.q_hat_array[ki]=1-current_threshold
                        elif self.task_type=='image_segmentation':

                            if self.prob_logit==True:
                                y_pred_logits = torch.sigmoid(self.learning_models[i](next_data_unit.to('cuda')))
                                threshold = self.alpha_array[k]
                            else:
                                y_pred_logits = self.learning_models[i](next_data_unit.to('cuda'))
                                threshold = np.log(self.alpha_array[k]/(1-self.alpha_array[k]))

                            y = next_data_label.to('cuda')

                            y_pred_binary = y_pred_logits > threshold

                            fnr =  self.calculate_fnr(y_pred_binary.squeeze(),y.squeeze())

                            fpr = self.calculate_fpr(y_pred_binary.squeeze(),y.squeeze())

                            self.dynamic_error_tracker[ki] = ((self.dynamic_prediction_counter[ki])/(self.dynamic_prediction_counter[ki]+1))*self.dynamic_error_tracker[ki]+1/(self.dynamic_prediction_counter[ki]+1)*fnr

                            self.slot_fnr[ki] = fnr
                            self.fnr_tracker[self.time_slot_index%self.fnr_track_window,ki] = fnr
                            self.dynamic_prediction_mask[self.time_slot_index%self.fnr_track_window,ki] = 1
                            self.slot_precision[ki] = fpr
                            self.prediction_mask[ki,self.time_slot_index] = 1
                            self.frame_prediction_mask[ki,self.frame_index] +=1
                            self.precision_tracker[ki] =  ((self.dynamic_prediction_counter[ki])/(self.dynamic_prediction_counter[ki]+1))*self.precision_tracker[ki]+1/(self.dynamic_prediction_counter[ki]+1)*fpr
                            self.dynamic_prediction_counter[ki]+=1
                k+=1
                if break_flag==True:
                    break
            i+=1

        for j in range(self.N_users):
                if self.dynamic_prediction_counter[j]>0:
                    self.avg_mean[j] = ((self.prediction_counter[j]-1)/(self.prediction_counter[j]))*self.avg_mean[j]+1/(self.prediction_counter[j])*self.dynamic_error_tracker[j]
                    self.avg_fpr_mean[j] = ((self.prediction_counter[j]-1)/(self.prediction_counter[j]))*self.avg_fpr_mean[j]+1/(self.prediction_counter[j])*self.precision_tracker[j]

                self.average_reliability_per_frame[j,self.frame_index] = self.avg_mean[j]

        tmp_error_tracker = None
        if self.update_crc_param == True:
            tmp_error_tracker = np.copy(self.dynamic_error_tracker)
            for j in range(self.N_users):
                if self.dynamic_prediction_counter[j]>0:
                    if self.enable_crc==True:
                            if self.alpha_array[j]>self.alpha_max[j]:
                                self.alpha_max[j]=self.alpha_array[j]
                            self.alpha_array[j] = self.alpha_array[j] + self.gamma[j] * (self.reliability_constraints[j]-self.dynamic_error_tracker[j])
                    self.avg_frame_mean[j] = ((self.frame_index)/(self.frame_index+1))*self.avg_frame_mean[j]+1/(self.frame_index+1)*self.dynamic_error_tracker[j]
                    self.avg_fpr_frame_mean[j] = ((self.frame_index)/(self.frame_index+1))*self.avg_fpr_mean[j]+1/(self.frame_index+1)*self.precision_tracker[j]
                    self.reliability_per_frame[j, self.frame_index] = self.dynamic_error_tracker[j]
                self.average_reliability_per_frame_over_multiple_predictions[j,self.frame_index] = self.avg_frame_mean[j]

                self.dynamic_error_tracker[j] = 0
                self.precision_tracker[j] = 0
                self.dynamic_prediction_counter[j] = 0
            self.frame_index += 1
            self.update_crc_param=False



            # for j in range(self.N_users):
            #         self.alpha_array[j] = self.alpha_array[j] + self.gamma[j] * (
            #                     self.reliability_constraints[j] - self.dynamic_error_tracker[
            #                 j] / self.prediction_per_frame)
            #         self.dynamic_error_tracker[j] = 0
            #         self.dynamic_prediction_counter[j] = 0

        return self.slot_precision,tmp_error_tracker


    def init_iterator(self):
        if self.dataset_loader==None:

            validation_set = CarSegmentationDataset(f'{self.dataset_path}/images/validation/data',
                                                                                                     f'{self.dataset_path}/images/validation/labels')
            test_set =  CarSegmentationDataset(f'{self.dataset_path}/images/test/data',
                                                                                                     f'{self.dataset_path}/images/test/labels')
            train_dev_sets = torch.utils.data.ConcatDataset([validation_set, test_set])
            print(f'Dataset length {len(train_dev_sets)}')
            self.dataset_loader = torch.utils.data.DataLoader(train_dev_sets,batch_size=1)
        self.dataset_iterator = iter(self.dataset_loader)

    def print_queue_status(self):
        for i in range(self.N_nodes):
            for k in range(self.N_users):
                ki=k
                break_flag = False
                if i in range(self.N_users):
                    ki = i
                    break_flag=True
                print(f"({i},{ki}), Virtual queue = {self.node_queues[i,ki]}, Actual queue = {len(self.data_units_queues[(i,ki)])}")
                if break_flag==True:
                    break

    def init_queues(self):
        self.virtual_node_queues = self.node_queues+self.place_holder


    def calculate_fpr(self,prediction, ground_truth):
        # Ensure prediction and ground_truth are on the same device
        device = prediction.device

        # Calculate False Positives (FP) and True Negatives (TN) for each sample
        fp = ((prediction == 1) & (ground_truth == 0)).sum(dim=0).sum(dim=0).to(device)
        tp = ((prediction == 1) & (ground_truth == 1)).sum(dim=0).sum(dim=0).to(device)
        tn = ((prediction == 0) & (ground_truth == 0)).sum(dim=0).sum(dim=0).to(device)
        fn = ((prediction == 0) & (ground_truth == 1)).sum(dim=0).sum(dim=0).to(device)

        # Calculate FPR for each sample, avoiding division by zero
        fpr = torch.zeros_like(fp, dtype=torch.float32, device=device)
        valid = (fn+tp+1) > 0
        fpr[valid] = fp[valid].float() / (fn[valid]+tp[valid]+1).float()


        valid = valid.cpu().detach().numpy()

        # Convert the FPR result to a numpy array
        fpr_numpy = fpr.cpu().numpy()
        fpr_numpy[valid] = np.minimum(fpr_numpy[valid],1)
        return fpr_numpy

    def calculate_fnr(self,prediction, ground_truth):
        # Ensure prediction and ground_truth are on the same device
        device = prediction.device

        # Calculate False Positives (FP) and True Negatives (TN) for each sample
        fp = ((prediction == 1) & (ground_truth == 0)).sum(dim=0).sum(dim=0).to(device)
        tp = ((prediction == 1) & (ground_truth == 1)).sum(dim=0).sum(dim=0).to(device)
        tn = ((prediction == 0) & (ground_truth == 0)).sum(dim=0).sum(dim=0).to(device)
        fn = ((prediction == 0) & (ground_truth == 1)).sum(dim=0).sum(dim=0).to(device)

        # Calculate FPR for each sample, avoiding division by zero
        fpr = torch.zeros_like(fp, dtype=torch.float32, device=device)
        fnr = torch.zeros_like(fn, dtype=torch.float32, device=device)
        valid = (fn+tp) > 0
        fnr[valid] = fn[valid].float() / (fn[valid] + tp[valid]).float()

        # Convert the FPR result to a numpy array
        fpr_numpy = fnr.cpu().numpy()

        return fpr_numpy

    def compute_window_fnr(self):
        avg_reliability = np.zeros((self.N_users, self.fnr_track_window))
        if self.time_slot_index-self.fnr_track_window>0:
            for t in range(self.fnr_track_window):
                for u in range(self.N_users):
                    avg_reliability[u, t] = np.sum(self.fnr_tracker[:, u]) / np.sum(
                        self.dynamic_prediction_mask[:, u])

        return avg_reliability


