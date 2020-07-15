#!/bin/bash

#SBATCH --job-name=egta_kuhn_poker_dqn
##SBATCH --job-name=egta_kuhn_poker_pg
##SBATCH --job-name=egta_kuhn_poker_ars
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=7g
#SBATCH --time=01-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard


module load python3.6-anaconda/5.2.0
##cd  $(dirname '${SLURM_SUBMIT_DIR}')
cd ${SLURM_SUBMIT_DIR}
##python psro_v2_example.py --oracle_type=BR --quiesce=False --gpsro_iterations=150 --number_training_episodes=100000 --sbatch_run=True
##python psro_v2_example.py --oracle_type=PG --quiesce=False --gpsro_iterations=150 --number_training_episodes=100000 --sbatch_run=True
python futu.py