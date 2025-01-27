/bin/hostname -s
echo pwd

echo "Running whisper fine-tuning with torchrun"
echo "GPUS_PER_NODE" $GPUS_PER_NODE
echo "SLURM_JOB_NUM_NODES" $SLURM_JOB_NUM_NODES
echo "MASTER_ADDR" $MASTER_ADDR
echo "MASTER_PORT" $MASTER_PORT
echo "CINECA_SCRATCH" $CINECA_SCRATCH

WORLD_SIZE=$(($NPROC_PER_NODE*$SLURM_JOB_NUM_NODES))
# node id of main process in each node. 0*4 = 0, 1*4 = 4, 2*4 = 8, 3*4 = 12, etc...
NODE_ID=$(($SLURM_NODEID*$NPROC_PER_NODE)) 
echo "WORLD_SIZE" $WORLD_SIZE

#--cache_dir="/leonardo_work/EUHPC_A01_006/new_cache" \
#--cache_dir=/leonardo_scratch/large/userexternal/lvesterb \

#--smdb_dataset=/leonardo_work/EUHPC_A01_006/data/big_parquets/smdb \
#--svt_dataset=/leonardo_work/EUHPC_A01_006/data/big_parquets/svt   \
#--riksdag_dataset=/leonardo_work/EUHPC_A01_006/data/rixvox/all_parquets \
#--youtube_dataset=/leonardo_work/EUHPC_A01_006/data/big_parquets/youtube \
#--multi_dataset \
#--youtube_dataset=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages/stage1/youtube \

python scripts/interleave/interleave.py \
    --model_name_or_path="openai/whisper-tiny" \
	--test_dataset_name="/leonardo_work/EUHPC_A01_006/data/parquets_for_whisper/smdb_2/XA_tv4_tv4_2021-10-14_070000_080000.parquet" \
	--train_dataset_name="/leonardo_work/EUHPC_A01_006/data/parquets_for_whisper/smdb_2/XA_tv4_tv4_2021-10-14_080000_090000.parquet" \
	--language="swedish" \
	--train_split_name="train" \
	--eval_split_name="eval" \
	--max_steps="5000" \
	--output_dir="output"\
	--save_dir="/leonardo_scratch/large/userexternal/jsikora0/interleave_large_stage1/" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="32" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--cache_dir=/leonardo_scratch/large/userexternal/jsikora0/ \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="text_whisper" \
	--freeze_feature_encoder="False" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	--gradient_checkpointing \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--preprocessing_only=True \
	--youtube_dataset=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_whisperlarge/stage1/youtube \
	--svt_dataset1=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_whisperlarge/stage1/svt \
	--svt_dataset2=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_whisperlarge/stage1/svt2 \
	--smdb_dataset=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_whisperlarge/stage1/smdb \
	--riksdag_dataset_old=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_whisperlarge/stage1/riksdagen_old \
	--riksdag_dataset_web=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_whisperlarge/stage1/riksdagen_web \
	--nst_cv_dataset=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_whisperlarge/stage1/nst/ \
	--swedia_dataset=/leonardo_scratch/large/userexternal/jsikora0/parquet_stages_whisperlarge/stage1/isof/ \
	--multi_dataset \
	--probabilities 0.126,0.157,0.167,0.151,0.282,0.102,0.014,0.001 \
	--whisper_feature_extractor "whisper-large" \
	--preprocessing_num_workers 32 \
	--file_prefix "snappy"
	

echo "Finished"
