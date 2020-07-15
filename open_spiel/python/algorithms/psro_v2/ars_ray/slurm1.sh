#!/bin/bash

#SBATCH --job-name=attacks
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --cpus-per-task=5
#SBATCH --nodes=3
#SBATCH --tasks-per-node 1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=00:10:00
#SBATCH --account=wellman1
#SBATCH --partition=standard


worker_num=2

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password --redis-shard-ports=6380 --node-manager-port=12345 --object-manager-port=12346 & # Starting the head
sleep 20


for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password --node-manager-port=12345 --object-manager-port=12346 & # Starting the workers
  sleep 20
done


python -u valid_ars_ray.py $redis_password 15
