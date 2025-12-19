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

import numpy as np


class Simulation:
    def __init__(self,**kwargs):


        #Simulation Parameters
        self.gamma = kwargs.get('gamma',None)
        self.alpha = kwargs.get('alpha',None)
        self.theta1 = kwargs.get('theta1',None)
        self.m = kwargs.get('m',0)
        self.M = kwargs.get('M',0)

        #Network Configuration Settings
        self.N_users = kwargs.get('N_users',3)
        self.N_nodes = kwargs.get('N_nodes',0)
        self.node_mapping = kwargs.get('node_mapping',None)
        self.deciding_nodes = kwargs.get('deciding_nodes',None)


        self.V = kwargs.get('V',None)
        self.t_sim = kwargs.get('t_sim',None)
        self.eta = kwargs.get('eta',None)
        self.path_loss_array = kwargs.get('path_loss_array',None)
        self.network_topology = kwargs.get('network_topology',None)
        self.device_indexes = kwargs.get('device_indexes',[0,1,2])
        self.prediction_set_cardinality = kwargs.get('prediction_set_cardinality',None)
        self.alpha_tracker = kwargs.get('alpha_tracker',None)
        self.frame_prediction_mask = kwargs.get('frame_prediction_mask',None)

        self.offloading_decisions_over_time=kwargs.get('R',None)
        self.deciding_node_over_time = kwargs.get('I',None)
        self.reliability_over_time = kwargs.get('reliability',None)
        self.reliability_mask = kwargs.get('reliability_mask',None)
        self.power_consumption_over_time = kwargs.get('power_consumption_over_time',None)
        self.accuracy_virtual_queue_tracker = kwargs.get('accuracy_virtual_queue_tracker',None)
        self.q_hat_tracker = kwargs.get('q_hat_tracker',None)
        self.prediction_set_cardinality_virtual_queue_tracker = kwargs.get('prediction_set_cardinality_virtual_queue_tracker',None)

        self.mobile_mean_reliability = np.zeros((len(self.device_indexes),self.t_sim))
        self.queue_tracker = kwargs.get('queue_tracker',None)
        self.error_probability_tracker = kwargs.get('error_probability_tracker',None)
        self.real_time_variance_tracker = kwargs.get('real_time_variance_tracker',None)
        self.energy_virtual_queue_tracker = kwargs.get('energy_virtual_queue_tracker',None)
        self.reliability_per_frame = kwargs.get('reliability_per_frame',None)
        self.average_reliability = kwargs.get('average_reliability',None)
        self.fnr_virtual_queue = kwargs.get('fnr_virtual_queue',None)
        self.std_virtual_queues = kwargs.get('std_virtual_queues',None)
        self.average_reliability_over_multiple_decisions = kwargs.get('average_reliability_over_multiple_decisions',None)
        self.reliability_min = kwargs.get('relibaility_min',None)
        self.alpha_max = kwargs.get('alpha_max',None)

        self.fnr_outage_queues = kwargs.get('fnr_outage_queues',None)
        self.Z_tracker = kwargs.get('Z_tracker',None)
        self.latency_tracker = kwargs.get('latency_tracker',None)
        self.latency_constraint = kwargs.get('latency_constraint',None)
        self.tau = kwargs.get('tau',None)
        self.arrival_rate = kwargs.get('arrival_rate',None)



