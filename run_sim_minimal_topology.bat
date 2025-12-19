 @echo off
set list= 1 1.2 1.4 1.6 1.8 2 2.5 3
set window_list= 1
set folder = %1
for %%a in (%list%) do (
    for %%b in (%window_list%) do (
        python simulate_CLO_strategy_mt.py --eta %%a --window_length %%b --data_output_dir %1 --t_sim 15000
    )
)