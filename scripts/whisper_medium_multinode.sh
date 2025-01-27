/bin/hostname -s
echo pwd

echo "Running whisper fine-tuning with torchrun"
echo "GPUS_PER_NODE" $GPUS_PER_NODE
echo "SLURM_JOB_NUM_NODES" $SLURM_JOB_NUM_NODES
echo "MASTER_ADDR" $MASTER_ADDR
echo "MASTER_PORT" $MASTER_PORT
echo "LOCAL_RANK" $LOCAL_RANK
echo "SLURM_PROCID" $SLURM_PROCID
echo "SLURM_JOB_NODELIST" $SLURM_JOB_NODELIST
echo "SLURM_NODEID" $SLURM_NODEID
echo "CHECKPOINT_DIR" $CHECKPOINT_DIR

WORLD_SIZE=$(($NPROC_PER_NODE*$SLURM_JOB_NUM_NODES))
# node id of main process in each node. 0*4 = 0, 1*4 = 4, 2*4 = 8, 3*4 = 12, etc...
NODE_ID=$(($SLURM_NODEID*$NPROC_PER_NODE)) 
echo "WORLD_SIZE" $WORLD_SIZE
###       --model_name_or_path="/leonardo_work/EUHPC_A01_006/models/whisper-tiny" \
#--smdb_dataset="/leonardo_work/EUHPC_A01_006/data/processed/without_timestamps/smdb_processed/train/" \
#--youtube_dataset="/leonardo_work/EUHPC_A01_006/data/processed/without_timestamps/youtube_processed/train/" \
#--resume_from_checkpoint="output_nov29_small_buffer10_workers6/checkpoint-6000" \
#--save_steps="100" \
#--dataset_name="/leonardo_scratch/large/userexternal/jsikora0/all_exhausted_training_test/" \

#        --deepspeed=$CONFIG_DIR"/ds_config_justyna.json" \
#--resume_from_checkpoint="output_nov29_small_buffer10_workers6/checkpoint-6000" \
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/run_speech_recognition_seq2seq_streaming_bpe_previous.py \
    --deepspeed=$CONFIG_DIR"/ds_config_medium.json" \
    --model_name_or_path="/leonardo_work/EUHPC_A01_006/models/whisper-medium" \
    --node_id=$SLURM_NODEID \
    --proc_id=$SLURM_PROCID \
    --dataset_name="/leonardo_scratch/large/userexternal/jsikora0/interleave_small/interleave_small_stage1" \
    --language="swedish" \
    --train_split_name="train" \
    --max_steps="150000" \
    --max_eval_samples="2048" \
    --dataloader_num_workers="12" \
    --cache_dir=/leonardo_scratch/large/userexternal/jsikora0/cache \
    --cache_dir_tokenizer=/leonardo_scratch/large/userexternal/jsikora0/cache_tokenizer \
    --output_dir="outputs/2025-01-24_medium-stage1_lr5e-5" \
    --per_device_train_batch_size="8" \
    --gradient_accumulation_steps="4" \
    --per_device_eval_batch_size="8" \
    --logging_steps="10" \
    --learning_rate="5e-5" \
    --warmup_steps="5000" \
    --eval_strategy="steps" \
    --eval_steps="750" \
    --save_steps="750" \
    --save_strategy="steps" \
    --max_length="448" \
    --generation_max_length="448" \
    --max_duration_in_seconds="30" \
    --text_column_name="input_features" \
    --audio_column_name="input_features" \
    --freeze_feature_encoder="False" \
    --report_to="tensorboard" \
    --metric_for_best_model="wer" \
    --greater_is_better="False" \
    --dispatch_batches="False" \
    --gradient_checkpointing=False \
    --ignore_data_skip=True \
    --adam_epsilon="1e-6" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.01" \
    --lr_scheduler_type="linear" \
    --bpe_dropout="0.2" \
    --fp16 \
    --do_train \
    --do_eval \
    --streaming \
    --shuffle_buffer_size="50" \
    --predict_with_generate \
    --stamps_probs="0.7" \
    --prompt_probability="0.7" \
    --remove_unused_columns="False" \
    --activation_dropout="0.1" \
    --seed="789" \

        
        
# --overwrite_output_dir
#--train_with_timestamps="True" \

## This is not enough to train with adafactor, need --adafactor=True
# --optim="adafactor" \


# Removed args:
# 	--load_best_model_at_end \

echo "Finished"
