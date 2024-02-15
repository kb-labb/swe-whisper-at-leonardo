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

WORLD_SIZE=$(($NPROC_PER_NODE*$SLURM_JOB_NUM_NODES))
# node id of main process in each node. 0*4 = 0, 1*4 = 4, 2*4 = 8, 3*4 = 12, etc...
NODE_ID=$(($SLURM_NODEID*$NPROC_PER_NODE)) 
echo "WORLD_SIZE" $WORLD_SIZE


torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
	scripts/run_speech_recognition_seq2seq.py \
        --deepspeed=$CONFIG_DIR"/ds_config.json" \
	--model_name_or_path="/leonardo_work/EUHPC_D01_040/models/whisper-large-v3" \
        --dataset_name="/leonardo_work/EUHPC_D01_040/data/google___fleurs" \
        --language="swedish" \
        --train_split_name="train" \
        --eval_split_name="test" \
        --max_steps="50" \
        --output_dir="output" \
        --per_device_train_batch_size="64" \
        --gradient_accumulation_steps="1" \
	--per_device_eval_batch_size="1" \
        --logging_steps="1" \
        --learning_rate="1e-5" \
	--warmup_steps="1000" \
        --evaluation_strategy="steps" \
        --eval_steps="1000" \
        --save_strategy="steps" \
        --save_steps="1000" \
        --generation_max_length="225" \
        --length_column_name="input_length" \
        --max_duration_in_seconds="30" \
        --text_column_name="transcription" \
        --freeze_feature_encoder="False" \
        --report_to="tensorboard" \
        --metric_for_best_model="wer" \
        --greater_is_better="False" \
	--optim="adafactor" \
        --gradient_checkpointing=True \
	--load_best_model_at_end \
        --fp16 \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --predict_with_generate 

echo "Finished"
