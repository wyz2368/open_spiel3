#!/bin/bash

#SBATCH --job-name=dqn_do
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=14g
#SBATCH --time=05-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python psro_v2_example.py --game_name=leduc_poker --n_players=2 --meta_strategy_method=nash --oracle_type=DQN --gpsro_iterations=150 --number_training_episodes=10000 --sbatch_run=True --root_result_folder=dqn_do_prd

