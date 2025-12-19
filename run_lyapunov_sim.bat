@echo off
set folder=%1
set realization_number=10
set eta_list=0.8
for %%b in (%eta_list%) do (
       python ./simulators/simulate_lyapunov_strategy_mt.py --data_output_dir %1 --t_sim 15000 --number_of_realizations 10 --eta %%b --window_length 10 --alpha 0.14 --V 10 --init_seed 512 --init_sim_index 0
)
