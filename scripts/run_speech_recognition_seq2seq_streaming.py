#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Additions and modifications Copyright 2024 National Library of Sweden. All rights reserved.
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
Fine-tuning the library models for sequence to sequence speech recognition
with 🤗 Datasets' streaming mode.
"""
# You can also adapt this script for your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
#from torchdata.stateful_dataloader import StatefulDataLoader
from os import listdir
from os.path import isfile, join

import datasets
import torch
from datasets import DatasetDict, IterableDatasetDict, interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset


import evaluate
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
    TrainerCallback,
    set_seed,
)

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import random
import tokenizers
from tokenizers.models import BPE


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

require_version("datasets>=1.18.2", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

logger = logging.getLogger(__name__)


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
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
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
    model_index_name: str = field(default=None, metadata={"help": "Pretty name for the model card."}
    )
    activation_dropout: float = field(default=0.0, metadata={"help": "Dropout rate for the activations."})
    max_length: int = field(default=448, metadata={"help": "The maximum length of the tokenized input text + special tokens."})


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
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
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
        default="audio",
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
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    do_remove_punctuation: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be striped of punctuation."},
    )
    do_normalize_eval: bool = field(
        default=True,
        metadata={"help": "Whether to normalise the references and predictions in the eval WER calculation."},
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
    shuffle_buffer_size: Optional[int] = field(
        default=20,
        metadata={
            "help": (
                "The number of streamed examples to download before shuffling them. The large the buffer, "
                "the closer it is to real offline shuffling."
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use streaming mode to load and pre-process the data."},
    )
    
    

@dataclass
class CustomTrainingArguments():
    """
    Custom arguments for training with multiple datasets.
    """
    
    node_id: int = field(default=None, metadata={"help": "Rank of the node."})
    proc_id: int = field(default=None, metadata={"help": "Id of the process."})
    multi_dataset: bool = field(default=False, metadata={"help": "Whether to train with multiple datasets."})
    
    train_with_timestamps: bool = field(default=False, metadata={"help": "Whether to train with timestamps or not"})  
    stamps_probs: float = field(default=0.0, metadata={"help" :"Provide probability for choosing labels with timestamps"})
    truncation: bool = field(default=True, metadata={"help": "Whether to truncate the input text to max_length"})
    
    
    bpe_dropout: float = field(default=0.0, metadata={"help":"BPE dropout rate. Makes the tokenizer use differnt subwords to encode the same word. Good for regularization to prevent overfitting."})

    cache_dir_tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    smdb_dataset: str = field(default=None, metadata={"help": "Provide the path to the smdb parquet dataset"})
    svt_dataset: str = field(default=None, metadata={"help": "Provide the path to the svt parquet dataset"})
    youtube_dataset: str = field(default=None, metadata={"help": "Provide the path to the youtube parquet dataset"})
    riksdag_dataset: str = field(default=None, metadata={"help": "Provide the path to the riksdag parquet dataset"})
    swedia_dataset: str = field(default=None, metadata={"help": "Provide the path to the swedia parquet dataset"})
    common_voice_dataset: str = field(default=None, metadata={"help": "Provide the path to the common voice parquet dataset"})
    nst_dataset: str = field(default=None, metadata={"help": "Provide the path to the nst parquet dataset"})
    fleurs_dataset: str = field(default=None, metadata={"help": "Provide the path to the fleurs parquet dataset"})
    
    probabilities: str = field(default=None, metadata={"help": "Probabilities for the datasets. Should contain the same number of elements as the number of datasets. Pass a string in format eg. '0.1,0.2,0.3,0.4' to set the probabilities for each dataset."})
    split_ratio: float = field(default=0.1, metadata={"help": "The ratio of the dataset to use for evaluation."})


@dataclass
class DataCollatorSpeechSeq2SeqWithBPEDropoutPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (`any`)
            The feature extractor that will extract features from the data.
        tokenizer (`any`)
            The BPE dropout tokenizer used to tokenize text.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        train_with_timestamps (`bool`)
            Whether to use timestamp labels during training.
        timestamp_probability (`float`)
            Probability of using timestamp labels.
    """
    feature_extractor: Any
    tokenizer: Any
    decoder_start_token_id: int
    train_with_timestamps: bool = False
    timestamp_probability: float = 0.8
    seed: int = None  # seed for reproducibility
    truncation: bool = True
    max_length: int = 448
    generation_max_length: int = 448

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if self.seed is not None:
            random.seed(self.seed)  
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = "input_features"
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]

        labels = []
        attention_masks = []
        if self.train_with_timestamps:
            texts = [feature["text"] for feature in features]
            texts_with_timestamps = [feature["text_timestamps"] for feature in features]
            # BPE dropout on the fly introduces randomness, and will lead to longer tokenized sequences 
            # than in our pre-processing step. We need to handle when `len(input_ids) > max_length` 
            # with truncation and max_length.
            self.tokenizer.set_prefix_tokens(predict_timestamps=False)
            tokenized_texts = self.tokenizer(texts, truncation=self.truncation, max_length=self.max_length)
            self.tokenizer.set_prefix_tokens(predict_timestamps=True)
            tokenized_text_timestamps = self.tokenizer(texts_with_timestamps, truncation=self.truncation, max_length=self.max_length)
            
            for i, feature in enumerate(features):
                # Use timestamp labels with probability `timestamp_probability` whenever observation is suitable
                if (random.random() < self.timestamp_probability) and feature["stage2_whisper_timestamps"]:
                    labels.append(tokenized_text_timestamps["input_ids"][i])
                    attention_masks.append(tokenized_text_timestamps["attention_mask"][i])
                else:
                    labels.append(tokenized_texts["input_ids"][i])
                    attention_masks.append(tokenized_texts["attention_mask"][i])

            # Only labels necessary with this approach
            labels = self._pad_labels(labels)
        else:
            texts = [feature["text"] for feature in features]
            self.tokenizer.set_prefix_tokens(predict_timestamps=False)
            tokenized_texts = self.tokenizer(texts, truncation=self.truncation, padding=True, max_length=self.max_length, return_tensors="pt")
            
            labels = tokenized_texts["input_ids"]
            attention_masks = tokenized_texts["attention_mask"]

            # replace padding with -100 to ignore loss correctly
            labels = labels.masked_fill(attention_masks.ne(1), -100)


        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

    def _pad_labels(self, labels: List[List[int]]) -> torch.Tensor:
        """
        Regular texts and texts with timestamps are processed independently and have different lengths 
        when combined. This function pads the combined labels to the same length, inserting -100 in 
        the labels to ignore the loss correctly. 

        NOTE: Label attention masks are not included in batch output returned by collator, 
        so we don't need to pad and return them using this approach.
        """
        labels = [torch.tensor(label) for label in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return labels
        
    
def load_maybe_streaming_dataset(dataset_name, split, streaming=False, buffer_size=20, cache_dir="/leonardo_scratch/large/userexternal/jsikora0"):
    """
    Utility function to load a dataset in streaming mode. For datasets with multiple splits,
    each split is loaded individually and then splits combined by taking alternating examples from
    each (interleaving).
    """
    split_path = join(dataset_name, split)
    split_files = [join(split_path, f) for f in listdir(split_path) if isfile(join(split_path, f))]

    # Load dataset for the specified split only
    if streaming == True:
        dataset = load_dataset(
            "parquet",
            data_files={split: split_files},
            streaming=streaming
        )[split] 
        dataset = dataset.shuffle(buffer_size=buffer_size)  # shuffles the shards order and use a shuffle buffer when you start iterating
    else:
        dataset = load_dataset(
            "parquet",
            data_files={split: split_files},
            cache_dir=cache_dir
        )[split]  

    return dataset


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


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_speech_recognition_seq2seq_streaming", model_args, data_args)

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

    logger.info(f"Node ID: {custom_args.node_id}. Process ID: {custom_args.proc_id}")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if custom_args.node_id == 0 and custom_args.proc_id == 0:
        logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

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

    # If main process  log training args
    if custom_args.node_id == 0 and custom_args.proc_id == 0:
        logger.info("Training/evaluation parameters %s", training_args)
        # Print version of transformers
        logger.info("Transformers version: %s", transformers.__version__)
        logger.info("Datasets version: %s", datasets.__version__)
        logger.info("Torch version: %s", torch.__version__)
        logger.info("Tokenizers version: %s", tokenizers.__version__)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    vectorized_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    if training_args.do_train:
        vectorized_datasets["train"] = load_maybe_streaming_dataset(
            data_args.dataset_name,
            split="train",
            streaming=data_args.streaming,
            buffer_size=data_args.shuffle_buffer_size,
            cache_dir=model_args.cache_dir,
        )
    


    if training_args.do_eval:
        vectorized_datasets["test"] = load_maybe_streaming_dataset(
            data_args.dataset_name,
            split="test",
            streaming=data_args.streaming,
            buffer_size=data_args.shuffle_buffer_size,
            cache_dir=model_args.cache_dir,
        )
    
    if custom_args.node_id == 0 and custom_args.proc_id == 0:
        print(vectorized_datasets)

   # logging.info(f"Splitting dataset by node: {training_args.local_rank}")
   # vectorized_datasets["train"] = datasets.distributed.split_dataset_by_node(vectorized_datasets["train"], rank=training_args.local_rank, world_size=training_args.world_size)


    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.update({"forced_decoder_ids": model_args.forced_decoder_ids, "suppress_tokens": model_args.suppress_tokens})
    config.update({"activation_dropout": model_args.activation_dropout})

    if training_args.gradient_checkpointing:
        config.update({"use_cache": False})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
        dropout=custom_args.bpe_dropout,
    )
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # Flash attention?
    )
    
    inference_tokenizer = tokenizer
    
    # 3. Regularization settings.
    #   a) BPE dropout in the tokenizer (randomly uses different subwords to encode the same word)
    if custom_args.bpe_dropout > 0:
            # Need a workaround to successfully load the tokenizer with BPE dropout.
            # See https://github.com/huggingface/tokenizers/issues/201#issue-584182286
            # Should only be used for training, not for inference/eval.
            logger.info(f"cache_dir_tokenizer: {custom_args.cache_dir_tokenizer}")

            with training_args.main_process_first():
                if is_main_process(training_args.local_rank):
                    workaround_files = tokenizer._tokenizer.model.save(custom_args.cache_dir_tokenizer, "training_tokenizer")

            workaround_tokenizer = os.path.join(custom_args.cache_dir_tokenizer, "training_tokenizer-vocab.json")
            workaround_merges = os.path.join(custom_args.cache_dir_tokenizer, "training_tokenizer-merges.txt")
            workaround_files = [workaround_tokenizer, workaround_merges]
            tokenizer._tokenizer.model = BPE.from_file(*workaround_files, dropout=custom_args.bpe_dropout)
    
    #tokenizer.set_prefix_tokens(
    #        language=data_args.language, task=data_args.task
    #    )  

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()

    if data_args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)


    normalizer = BasicTextNormalizer()  # 'official' text normalizer from OpenAI
#
#    if data_args.max_train_samples is not None:
#        vectorized_datasets["train"] = (
#            vectorized_datasets["train"].take(data_args.max_train_samples)
#            if data_args.streaming
#            else vectorized_datasets["train"].select(range(data_args.max_train_samples))
#        )

    
    #dataloader_train = StatefulDataLoader(vectorized_datasets["train"], batch_size=32, num_workers=4)
    #dataloader_eval = StatefulDataLoader(vectorized_datasets["eval"], batch_size=32, num_workers=4)


    # 8. Load Metric
    metric = evaluate.load("wer")
    do_normalize_eval = data_args.do_normalize_eval

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = inference_tokenizer.pad_token_id

        pred_str = inference_tokenizer.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)
        # we do not want to group tokens when computing the metrics
        label_str = inference_tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True, decode_with_timestamps=False)

        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]
            # filtering step to only evaluate the samples that correspond to non-zero references:
            #pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
            #label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]
            filtered_data = [(label, pred) for label, pred in zip(label_str, pred_str) if len(pred.strip()) > 0]
            label_str, pred_str = zip(*filtered_data)
            filtered_data = [(label, pred) for label, pred in zip(label_str, pred_str) if len(label.strip()) > 0]
            label_str, pred_str = zip(*filtered_data)
            
            assert len(pred_str) == len(label_str), "The number of predictions and reference texts should be the same."

            # Filter out entires from both pred_str and label_str, when label_str contains "<|nospeech|>"
            pred_str = [pred_str[i] for i in range(len(pred_str)) if " <|nospeech|>" not in label_str[i]]
            label_str = [label_str[i] for i in range(len(label_str)) if " <|nospeech|>" not in label_str[i]]

            if custom_args.node_id == 0 and custom_args.proc_id == 0:
                logger.info(f"len label_str after filtering {len(label_str)}")
                logger.info(f"len pred_str after filtering {len(pred_str)}")

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # 9. Create a single speech processor
    logger.info("Creating processor on Main Process First")
    with training_args.main_process_first():
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            inference_tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithBPEDropoutPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        train_with_timestamps=custom_args.train_with_timestamps,
        timestamp_probability=custom_args.stamps_probs,
        seed=training_args.seed,
        truncation=custom_args.truncation,
        max_length=model_args.max_length,
        generation_max_length=training_args.generation_max_length,
    )

    # 11. Configure Trainer
    # Trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
    # Only required for streaming: Trainer automatically shuffles non-streaming datasets
    class ShuffleCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
            if isinstance(train_dataloader.dataset, IterableDatasetShard):
                pass  # set_epoch() is handled by the Trainer
            elif isinstance(train_dataloader.dataset, IterableDataset):
                train_dataloader.dataset.set_epoch(train_dataloader.dataset.epoch + 1)

    train_dataset = vectorized_datasets["train"]
    valid_dataset = vectorized_datasets["test"]

    if data_args.max_eval_samples is not None:
        valid_dataset = (
            valid_dataset.take(data_args.max_eval_samples)
            if data_args.streaming
            else valid_dataset.select(range(data_args.max_eval_samples))
        )

    # Remove columns labels_timestamps, data_source, stage2_whisper_timestamps, audio_path, duration, text_timestamps, is_silence, text.
    valid_dataset = valid_dataset.remove_columns(["data_source", "audio_path", "duration", "is_silence",])

    # # checkpoint
    # state_dict = dataloader.state_dict()  # uses iterable_dataset.state_dict() under the hood
    # # resume from checkpoint
    # dataloader.load_state_dict(state_dict)  # uses iterable_dataset.load_state_dict() under the hood

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.shuffle(seed=training_args.seed, buffer_size=data_args.shuffle_buffer_size) if data_args.streaming else train_dataset,
        eval_dataset=valid_dataset.shuffle(seed=training_args.seed, buffer_size=data_args.shuffle_buffer_size) if data_args.streaming else valid_dataset,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[ShuffleCallback()] if data_args.streaming else None,
    )

    # 12. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        if data_args.max_eval_samples:
            metrics["eval_samples"] = data_args.max_eval_samples

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 14. Write Training Stats
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": "whisper-event",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
        if "common_voice" in data_args.dataset_name:
            kwargs["language"] = data_args.dataset_config_name.split('-')[0]
        if model_args.model_index_name is not None:
            kwargs["model_name"] = model_args.model_index_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
