#!/bin/bash -l

#SBATCH --job-name=whisper-tiny   # create a short name for your job
#SBATCH --nodes=1          #161
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=400GB               # total memory per node 
#SBATCH --time=0-24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --ntasks-per-node=1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                    #default
#SBATCH --account=EUHPC_A01_006       #SBATCH --account=EUHPC_A01_006
#SBATCH --output=/leonardo_work/EUHPC_A01_006/experiments_whisper/logs/whisper-large/sbatch_logs/sbatch-%J.log
#SBATCH --exclude=lrdn1765,lrdn3032,lrdn2751,lrdn0959,lrdn1072,lrdn2031,lrdn0970,lrdn1753,lrdn1636,lrdn3146,lrdn2631
#SBATCH --requeue
# Example how to launch the job: 
# sbatch start_script_whisper.sh scripts/whisper_large_multinode.sh

module purge

# Automatically resubmit the job unless it is finished
#if [ ! -f "finished_whisper" ] ; then
#	sbatch --dependency=afterany:$SLURM_JOBID start_script_whisper.sh scripts/whisper_large_multinode.sh
#else
#	exit 0
#fi

pwd
addr=$(/bin/hostname -s)
export MASTER_ADDR=$addr
export NPROC_PER_NODE=4 # We use this to calculate distributed_world_size (total nr of GPUs) in train_script.sh.
# export MASTER_PORT=14938 # MASTER_PORT env variable is overwritten in the nodes anyway by SLURM/HPC system


# debugging flags (optional)
# export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1

# I experiencing low bandwidth when using NCCL with GPUs, the following variable
# might help increase bandwidth: NCCL_CROSS_NIC=1. See the following link here for
# more details.
# export NCCL_CROSS_NIC=1
# export NCCL_IB_GID_INDEX=3  
export NCCL_IB_TIMEOUT=22
# export CUDA_DEVICE_MAX_CONNECTIONS=1

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

TRAIN_SCRIPT=$1
PROJECT="/leonardo_work/EUHPC_A01_006/experiments_whisper"
export CHECKPOINT_DIR="/leonardo_work/EUHPC_A01_006/models/whisper-tiny"
export CONFIG_DIR="${PROJECT}/configs"
CONTAINER_PATH="/leonardo_work/EUHPC_A01_006/containers/whisper-flash-new"
LOGGING=$PROJECT/logs/whisper-tiny # Make sure to create logs/ before running this script

echo "MASTER_ADDR" $MASTER_ADDR
echo "MASTER_PORT" $MASTER_PORT
echo "NPROC_PER_NODE" $NPROC_PER_NODE
echo "SLURM_JOB_NAME" $SLURM_JOB_NAME
echo "SLURM_JOB_ID" $SLURM_JOB_ID
echo "SLURM_JOB_NODELIST" $SLURM_JOB_NODELIST
echo "SLURM_JOB_NUM_NODES" $SLURM_JOB_NUM_NODES
echo "SLURM_LOCALID" $SLURM_LOCALID
echo "SLURM_NODEID" $SLURM_NODEID
echo "SLURM_PROCID" $SLURM_PROCID
echo "SLURM_GPUS" $SLURM_GPUS
echo "SLURM_GPUS_PER_NODE" $SLURM_GPUS_PER_NODE
echo "CHECKPOINT_DIR" $CHECKPOINT_DIR

 srun error handling:
 --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
 --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    --label \
    --output=$LOGGING/${SLURM_JOB_NAME}_${DATETIME}.log
    "


cmd="srun $SRUN_ARGS \
      singularity exec --nv -B /leonardo_work/EUHPC_A01_006,/leonardo_scratch/large/userexternal/lvesterb,/leonardo_scratch/large/userexternal/lvesterb $CONTAINER_PATH bash $PROJECT/$TRAIN_SCRIPT"

echo "Executing:"
echo $cmd

$cmd

# touch finished_whisper

set +x
exit 0
