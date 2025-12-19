@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM -------------------------------------------------
REM Usage:
REM   %~nx0 OUTPUT_DIR [SCRIPT] [REALIZATIONS] ETA1 [ETA2 ...]
REM   %~nx0 OUTPUT_DIR [SCRIPT] [REALIZATIONS] ETA1,ETA2,ETA3
REM
REM Examples:
REM   %~nx0 results
REM   %~nx0 results ./simulators/simulate_lyapunov_strategy_mt.py 10 0.6 0.8 1.0
REM   %~nx0 results ./simulators/simulate_lyapunov_strategy_mt.py 20 0.6,0.8,1.0
REM   %~nx0 results .\simulators\simulate_lyapunov_strategy_mt.py 5 0.9
REM -------------------------------------------------

set "folder=%~1"
if "%folder%"=="" (
  echo Usage: %~nx0 OUTPUT_DIR [SCRIPT] [REALIZATIONS] ETA1 [ETA2 ...]
  exit /b 1
)

set "script=%~2"
if "%script%"=="" set "script=./simulators/simulate_lyapunov_strategy_mt.py"

set "realization_number=%~3"
if "%realization_number%"=="" set "realization_number=10"

REM Remaining args (4+) are etas
shift
shift
shift

set "eta_list="
:collect_etas
if "%~1"=="" goto etas_done
set "eta_list=!eta_list! %~1"
shift
goto collect_etas

:etas_done
if "!eta_list!"=="" set "eta_list=0.8"

REM Support comma-separated list passed as a single token
set "eta_list=!eta_list:,= !"

for %%b in (!eta_list!) do (
  echo Running: script="!script!" eta=%%b realizations=!realization_number! output="!folder!"
  python "!script!" ^
    --data_output_dir "!folder!" ^
    --t_sim 15000 ^
    --number_of_realizations !realization_number! ^
    --eta %%b ^
    --window_length 10 ^
    --alpha 0.14 ^
    --V 10 ^
    --init_seed 512 ^
    --init_sim_index 0
)

endlocal
