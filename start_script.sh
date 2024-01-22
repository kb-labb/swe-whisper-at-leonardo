#!/bin/bash -l

##SBATCH --job-name=w2v   # create a short name for your job
#SBATCH --nodes=1           #161
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256GB               # total memory per node 
#SBATCH --time=0-00:18:00          # total run time limit (HH:MM:SS)
#SBATCH --ntasks-per-node=1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg                    #default
#SBATCH --account=EUHPC_D01_040
#SBATCH --output=logs/sbatch-%J.log


module purge
module load singularity

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

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

TRAIN_SCRIPT=$1
LABBIS=$2 #labbis is you! pass your name to the start_script command
PROJECT="/leonardo_work/EUHPC_D01_040/kb-leonardo/${LABBIS}/swe-whisper-at-leonardo"
export CHECKPOINT_DIR="/leonardo_work/EUHPC_D01_040/kb-leonardo/${LABBIS}/swe-whisper-at-leonardo/checkpoints"
export CONFIG_DIR="/leonardo_work/EUHPC_D01_040/kb-leonardo/${LABBIS}/swe-whisper-at-leonardo/configs"
TARGET_DIR="/leonardo_work/EUHPC_D01_040/kb-leonardo/${LABBIS}/swe-whisper-at-leonardo/scripts"
CONTAINER_PATH="/leonardo_work/EUHPC_D01_040/containers/whisper-sandbox"
LOGGING=$PROJECT/logs # Make sure to create logs/ before running this script

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
echo "WORK" $WORK

cmd="srun -l --output=$LOGGING/${SLURM_JOB_NAME}_${DATETIME}.log \
      singularity exec --nv -B $WORK $CONTAINER_PATH bash $TARGET_DIR/$TRAIN_SCRIPT"

echo "Executing:"
echo $cmd

$cmd

set +x
exit 0
