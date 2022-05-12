#!/bin/sh

#SBATCH -A uppmax2020-2-2 # project number for accounting.
#SBATCH -p core -n 4 # resource, 4 cpu cores
#SBATCH -M snowy # cluster name
#SBATCH -t 10:00:00 # time reserved for your job, not the exact time your job will run. If the job takes longer time, you need to increase the time. format: #hour:min:sec#
#SBATCH -J m2-arabic # job name
#SBATCH --gres=gpu:1 # reserve one GPU for your job


module load python/3.6.8
source /home/krisfarr/thesis/venv/bin/activate

python /home/krisfarr/thesis/Trainer_implementation/main.py --language arabic --model_checkpoint mbert --training_lingual multi --validation_lingual mono
python /home/krisfarr/thesis/Trainer_implementation/main.py --language arabic --model_checkpoint mbert --training_lingual multi --validation_lingual multi