/bin/hostname -s
echo pwd

echo "Running whisper fine-tuning"
echo "GPUS_PER_NODE" $GPUS_PER_NODE
echo "SLURM_JOB_NUM_NODES" $SLURM_JOB_NUM_NODES
echo "MASTER_ADDR" $MASTER_ADDR
echo "MASTER_PORT" $MASTER_PORT

WORLD_SIZE=$(($NPROC_PER_NODE*$SLURM_JOB_NUM_NODES))
# node id of main process in each node. 0*4 = 0, 1*4 = 4, 2*4 = 8, 3*4 = 12, etc...
NODE_ID=$(($SLURM_NODEID*$NPROC_PER_NODE)) 
echo "WORLD_SIZE" $WORLD_SIZE

python  scripts/run_speech_recognition_seq2seq.py \
	--model_name_or_path="/leonardo_work/EUHPC_D01_040/models/whisper-tiny" \
	--dataset_name="/leonardo_work/EUHPC_D01_040/data/google___fleurs" \
	--language="swedish" \
	--train_split_name="train" \
	--eval_split_name="test" \
	--max_steps="5000" \
	--output_dir="output" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="32" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
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
	--load_best_model_at_end \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate 


#### For multinode training we need something more like the below examples

# python -m torch.distributed.launch \ 
#     --use_env \
#     --nproc_per_node=$GPUS_PER_NODE \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=0 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     $(which fairseq-hydra-train) \
#     task.data=/leonardo_work/EUHPC_D01_040/data/p4/manifest/ \
#     distributed_training.distributed_world_size=8 \
#     +optimization.update_freq='[4]' \
#     checkpoint.save_dir=/leonardo_work/EUHPC_D01_040/kb-leonardo/faton/swe-wav2vec/checkpoints/ \
#     --config-dir /leonardo_work/EUHPC_D01_040/kb-leonardo/faton/swe-wav2vec/ \
#     --config-name wav2vec2_large_additional_pretrain

# fairseq-hydra-train \
#     task.data=/leonardo_work/EUHPC_D01_040/data/p4/manifest/ \
#     checkpoint.save_dir=/leonardo_work/EUHPC_D01_040/kb-leonardo/faton/swe-wav2vec/checkpoints/ \
#     --config-dir /leonardo_work/EUHPC_D01_040/kb-leonardo/faton/swe-wav2vec/ \
#     --config-name wav2vec2_large_additional_pretrain \
#     distributed_training.distributed_world_size=${WORLD_SIZE} \
#     distributed_training.nprocs_per_node=${GPUS_PER_NODE} \
#     distributed_training.distributed_init_method="tcp://${MASTER_ADDR}:${MASTER_PORT}" \
#     distributed_training.distributed_rank=$NODE_ID


echo "Finished"
