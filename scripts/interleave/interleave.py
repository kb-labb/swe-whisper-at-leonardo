#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
import pickle
import warnings
import pandas as pd
from tqdm import tqdm
from datasets.utils.py_utils import convert_file_size_to_int
from os import listdir
from os.path import isfile, join
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import datetime
import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset, load_from_disk, interleave_datasets, Sequence, Value


import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
#from transformers.utils import is_flash_attn_2_available
from pathlib import Path
from datasets import Array2D, Features, Sequence, Value, load_dataset

from concurrent.futures import ThreadPoolExecutor

#torch.set_num_threads(1)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.32.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio_tensor",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    
    sampling_rate: int = field(
        default=16_000,
        metadata={"help": "The sampling rate of the audio data."},
    )
    test_dataset_name: str = field(
        default=None, metadata={"help": "The name of the local test dataset to use."}
    )
    train_dataset_name: str = field(
        default=None, metadata={"help": "The name of the local train dataset to use."}
    )
    file_prefix: str = field(default=None, metadata={"help": "The prefix to add to the name of the final parquet files."}
    )
    

@dataclass
class CustomTrainingArguments():
    """
    Custom arguments for training with multiple datasets.
    """
    
    multi_dataset: bool = field(default=False, metadata={"help": "Whether to train with multiple datasets."})
    process_single_dataset: bool = field(default=False, metadata={"help": "Whether to process a single dataset."})

    
    smdb_dataset: str = field(default=None, metadata={"help": "Provide the path to the smdb parquet dataset"})
    svt_dataset1: str = field(default=None, metadata={"help": "Provide the path to the svt1 parquet dataset"})
    svt_dataset2: str = field(default=None, metadata={"help": "Provide the path to the svt2 parquet dataset"})
    youtube_dataset: str = field(default=None, metadata={"help": "Provide the path to the youtube parquet dataset"})
    riksdag_dataset_old: str = field(default=None, metadata={"help": "Provide the path to the riksdag parquet dataset"})
    riksdag_dataset_web: str = field(default=None, metadata={"help": "Provide the path to the riksdag parquet dataset"})

    swedia_dataset: str = field(default=None, metadata={"help": "Provide the path to the swedia parquet dataset"})
    common_voice_dataset: str = field(default=None, metadata={"help": "Provide the path to the common voice parquet dataset"})
    nst_cv_dataset: str = field(default=None, metadata={"help": "Provide the path to the nst parquet dataset"})
    fleurs_dataset: str = field(default=None, metadata={"help": "Provide the path to the fleurs parquet dataset"})
    
    single_dataset: str = field(default=None, metadata={"help": "Provide path to a single dataset to be processed"})
    
    probabilities: str = field(default=None, metadata={"help": "Probabilities for the datasets. Should contain the same number of elements as the number of datasets. Pass a string in format eg. '0.1,0.2,0.3,0.4' to set the probabilities for each dataset."})
    split_ratio: float = field(default=0.005, metadata={"help": "The ratio of the dataset to use for evaluation."})
    valid_split_ratio: float = field(default=0.1, metadata={"help": "The ratio of the dataset to use for validation."})
    whisper_feature_extractor: str = field(default="whisper-small", metadata={"help": "Which Whisper feature extractor the data was processed with"})
    save_dir: str = field(default="interleaved_outputs", metadata={"help": "Directory to save to"})

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, CustomTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_speech_recognition_seq2seq", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    num_workers = data_args.preprocessing_num_workers
    if custom_args.whisper_feature_extractor == "whisper-small":
        input_features_shape = (80, 3000)
    if custom_args.whisper_feature_extractor == "whisper-large":
        input_features_shape = (128, 3000)
    features = Features(
    {
        "input_features": Array2D(dtype="float32", shape=input_features_shape),
        "attention_mask": Sequence(Value(dtype="int32")),
        "labels": Sequence(Value(dtype="int32")),
        "labels_timestamps": Sequence(Value(dtype="int32")),
        "text": Value(dtype="string"),
        "text_timestamps": Value(dtype="string"),
        "previous_text": Value(dtype="string"),
        "duration": Value(dtype="float32"),
        "audio_path": Value(dtype="string"),
        "is_silence": Value(dtype="bool"),
        "stage2_whisper_timestamps": Value(dtype="bool"),
        "data_source": Value(dtype="string"),
    }
)
    

    # 4. Load dataset from parquet files
        # dataset = dataset.rename_column("old_column_name", "new_column_name")
    if custom_args.multi_dataset:
        # Load multiple datasets and interleave them, then split them into train and eval sets
        
        list_of_datasets = []
        probabilities = [float(prob) for prob in custom_args.probabilities.split(',')]
        print('Cache dir:', model_args.cache_dir)
        #columns = ["input_features", "attention_mask", "labels","labels_timestamps", "text","text_timestamps", "duration", "audio_path", "is_silence",  "stage2_whisper_timestamps", "data_source"]
        columns = ["input_features", "attention_mask", "labels","labels_timestamps", "text","text_timestamps", "previous_text", "duration", "audio_path", "is_silence",  "stage2_whisper_timestamps", "data_source"]
        if custom_args.youtube_dataset:
            print('Starting to load youtube')
            list_files = [custom_args.youtube_dataset +"/"+ f for f in listdir(custom_args.youtube_dataset) if isfile(join(custom_args.youtube_dataset, f))]
            youtube = load_dataset("parquet", data_files={'train': list_files}, cache_dir = model_args.cache_dir, columns=columns, features=features, streaming=False)
            youtube = youtube.with_format("torch",columns=["input_features", "attention_mask", "labels", "labels_timestamps"],output_all_columns=True)
            list_of_datasets.append(youtube["train"])
            print('Done loading youtube')
        if custom_args.svt_dataset1:
            print('Starting to load svt1')
            list_files = [custom_args.svt_dataset1 +"/"+ f for f in listdir(custom_args.svt_dataset1) if isfile(join(custom_args.svt_dataset1, f))]
            svt1 = load_dataset("parquet", data_files={'train': list_files}, cache_dir = model_args.cache_dir,  columns=columns, features=features, streaming=False)   
            svt1 = svt1.with_format("torch",columns=["input_features", "attention_mask", "labels", "labels_timestamps"],output_all_columns=True)
            list_of_datasets.append(svt1["train"])
            print('Done loading svt1')
        if custom_args.svt_dataset2:
            print('Starting to load svt2')
            list_files = [custom_args.svt_dataset2 +"/"+ f for f in listdir(custom_args.svt_dataset2) if isfile(join(custom_args.svt_dataset2, f))]
            svt2 = load_dataset("parquet", data_files={'train': list_files}, cache_dir = model_args.cache_dir,  columns=columns, features=features, streaming=False)   
            svt2 = svt2.with_format("torch",columns=["input_features", "attention_mask", "labels", "labels_timestamps"],output_all_columns=True)
            list_of_datasets.append(svt2["train"])
            print('Done loading svt2')
        if custom_args.smdb_dataset:
            print('Starting to load smdb')
            list_files = [custom_args.smdb_dataset +"/"+ f for f in listdir(custom_args.smdb_dataset) if isfile(join(custom_args.smdb_dataset, f))]
            smdb = load_dataset("parquet", data_files={'train': list_files}, cache_dir = model_args.cache_dir,  columns=columns, features=features, streaming=False)      
            smdb = smdb.with_format("torch",columns=["input_features", "attention_mask", "labels", "labels_timestamps"],output_all_columns=True)
            list_of_datasets.append(smdb["train"])   
            print('Done loading smdb')
        if custom_args.riksdag_dataset_old:
            print('Starting to load riksdag old')
            list_files = [custom_args.riksdag_dataset_old +"/"+ f for f in listdir(custom_args.riksdag_dataset_old) if isfile(join(custom_args.riksdag_dataset_old, f))]
            riksdag_old = load_dataset("parquet", data_files={'train': list_files}, cache_dir = model_args.cache_dir,  columns=columns, features=features, streaming=False)   
            riksdag_old = riksdag_old.with_format("torch",columns=["input_features", "attention_mask", "labels", "labels_timestamps"],output_all_columns=True)
            list_of_datasets.append(riksdag_old["train"])
            print('Done loading riksdag old')
        if custom_args.riksdag_dataset_web:
            print('Starting to load riksdag web')
            list_files = [custom_args.riksdag_dataset_web +"/"+ f for f in listdir(custom_args.riksdag_dataset_web) if isfile(join(custom_args.riksdag_dataset_web, f))]
            riksdag_web = load_dataset("parquet", data_files={'train': list_files}, cache_dir = model_args.cache_dir,  columns=columns, features=features, streaming=False)   
            riksdag_web = riksdag_web.with_format("torch",columns=["input_features", "attention_mask", "labels", "labels_timestamps"],output_all_columns=True)
            list_of_datasets.append(riksdag_web["train"])
            print('Done loading riksdag web')
        if custom_args.nst_cv_dataset:
            print('Starting to load nst cv')
            list_files = [custom_args.nst_cv_dataset +"/"+ f for f in listdir(custom_args.nst_cv_dataset) if isfile(join(custom_args.nst_cv_dataset, f))]
            nst_cv = load_dataset("parquet", data_files={'train': list_files}, cache_dir = model_args.cache_dir,  columns=columns, features=features, streaming=False)
            nst_cv = nst_cv.with_format("torch",columns=["input_features", "attention_mask", "labels", "labels_timestamps"],output_all_columns=True)
            list_of_datasets.append(nst_cv["train"])
            print('Done loading nst cv')
        if custom_args.swedia_dataset:
            print('Starting to load swedia')
            list_files = [custom_args.swedia_dataset +"/"+ f for f in listdir(custom_args.swedia_dataset) if isfile(join(custom_args.swedia_dataset, f))]
            swedia = load_dataset("parquet", data_files={'train': list_files}, cache_dir = model_args.cache_dir,  columns=columns, features=features, streaming=False)
            swedia = swedia.with_format("torch",columns=["input_features", "attention_mask", "labels", "labels_timestamps"],output_all_columns=True)
            list_of_datasets.append(swedia['train'])  
            print('Done loading swedia')
        
        print('About to start interleaving')
        all_datasets = interleave_datasets(list_of_datasets, probabilities=probabilities, seed=training_args.seed, stopping_strategy="all_exhausted")
        print('Done interleaving')
        print(all_datasets)
        
        all_datasets = all_datasets.train_test_split(test_size=custom_args.split_ratio, seed=training_args.seed)        
        print('Done making train test splits')
        all_datasets = all_datasets.flatten()        
        print('Done flattening')
        print(all_datasets)
    
    vectorized_datasets_test_eval = all_datasets["test"].train_test_split(test_size=custom_args.valid_split_ratio, seed=training_args.seed)  
    
    train_test_valid_vectorized_datasets = DatasetDict({
    'train': all_datasets['train'],
    'test': vectorized_datasets_test_eval['train'],
    'valid': vectorized_datasets_test_eval['test']})
    
    print(train_test_valid_vectorized_datasets)
    print('Done making train test valid splits')
    print('Start saving shards')
 
#    if data_args.preprocessing_only:
#        for lang, dataset in tqdm(train_test_valid_vectorized_datasets.items()): #remember to change the name of the dataset to save
#            # Shard the dataset to 500MB
#            dataset_nbytes = dataset._estimate_nbytes()
#            print(dataset_nbytes)
#            max_shard_size = convert_file_size_to_int("500MB")
#            num_shards = dataset_nbytes // max_shard_size + 1
#            print('num shards', num_shards)
#            num_shards = int(max(num_shards, 1))
#            print('num shards ', num_shards)
#            shards = (
#                dataset.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards)
#            )
#            for index, shard in enumerate(shards):
#                os.makedirs(f"/leonardo_scratch/large/userexternal/jsikora0/interleaved_datasets_nov25_new/{lang}", exist_ok=True)
#                shard.to_parquet(f"/leonardo_scratch/large/userexternal/jsikora0/interleaved_datasets_nov25_new/{lang}/{lang}--{data_args.file_prefix}-{index:04d}.parquet")
    
    
    #first_time = datetime.datetime.now()
    #print("starting saving with ", first_time)
    def process_shard(lang, shard, index, data_args):
        """Process and save a single shard."""
        print('save dir is ', custom_args.save_dir)
        output_dir = custom_args.save_dir
        #output_dir = "/leonardo_scratch/large/userexternal/jsikora0/interleave_small_stage1"
        os.makedirs(output_dir, exist_ok=True)
        shard.to_parquet(f"{output_dir}/{lang}--{data_args.file_prefix}-{index:04d}.parquet", batch_size=128)
        print('done saving parquet ')


    def process_dataset(lang, dataset, data_args):
        """Process a dataset: calculate shards, split, and save them."""
        print('dataset', dataset)
        dataset_nbytes = dataset._estimate_nbytes()
        max_shard_size = convert_file_size_to_int("2GB")
        num_shards = dataset_nbytes // max_shard_size + 1
        num_shards = int(max(num_shards, 1))
        print('num shards', num_shards)

        shards = [
            (lang, dataset.shard(num_shards=num_shards, index=i, contiguous=True), i)
            for i in range(num_shards)
        ]
        return shards

    def parallelize_datasets(data_args, train_test_valid_vectorized_datasets):
        """Parallelize the dataset processing."""
        all_shards = []
        for lang, dataset in train_test_valid_vectorized_datasets.items():
            shards = process_dataset(lang, dataset, data_args)
            all_shards.extend(shards)
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(process_shard, lang, shard, index, data_args)
                for lang, shard, index in tqdm(all_shards)
            ]
            for future in tqdm(futures):
                future.result()  

    if data_args.preprocessing_only:
        parallelize_datasets(data_args, train_test_valid_vectorized_datasets)

if __name__ == "__main__":
    main()
