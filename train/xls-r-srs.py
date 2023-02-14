#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A short script to fine-tune an existing, pretrained HuggingFace XLS-R
# wav2vec2 model for use in Tsuut'ina ASR.
#
# See:
#
#   https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
#   https://github.com/macairececile/internship_lacito_2021/blob/main/notebooks/Fine-Tune%20XLSR-Wav2Vec2%20-%20Na.ipynb
#
# !pip install datasets==1.18.3
# !pip install transformers==4.11.3
# !pip install huggingface_hub==0.1
# !pip install torchaudio
# !pip install librosa
# !pip install jiwer

import json
import os
import os.path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

import preprocessing_srs as prep_srs

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
import numpy as np
import pandas as pd
import torchaudio
import transformers

srs_model = "wav2vec2-xls-r-300-srs-test1"


# Dataset format:
#
#   dataset_dir/
#       train.tsv
#       test.tsv
#       clips/
#           clip_001.mp3
#           clip_002.mp3
#           (...)
#
# TSV columns:
#   "path" (filename inside "wav" directory, e.g., "cv_tr_23660893.mp3")
#   "sentence" (orthographic text of sentence, with regular punctuation,
#      spelling, etc., e.g., "Hatırlamıyor musun?")

# Load the Tsuut'ina training and validation data sets from the given TSV
# files, remove the object packaging to get at just the primary data, then
# process each set to produce orthographic sentences in 'sentence' and a
# list of phonemes in 'phonemes' (with "|" as the word separator in both).
srs_train_data = datasets.load_dataset("csv", \
    data_files = ['tsuutina_data/train.tsv'], delimiter = '\t')
srs_train_data = srs_train_data['train']
srs_train_data = srs_train_data.map(prep_srs.srs_process_row)

srs_valid_data = datasets.load_dataset("csv", \
    data_files = ['tsuutina_data/validate.tsv'], delimiter = '\t')
srs_valid_data = srs_valid_data['train']
srs_valid_data = srs_valid_data.map(prep_srs.srs_process_row)

def extract_all_chars(batch):
    all_text = "|".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

srs_vocab_train = srs_train_data.map(extract_all_chars, batched = True, \
    batch_size = -1, keep_in_memory = True, \
    remove_columns = srs_train_data.column_names)

srs_vocab_valid = srs_valid_data.map(extract_all_chars, batched = True, \
    batch_size = -1, keep_in_memory = True, \
    remove_columns = srs_valid_data.column_names)

# Combine the character sets from both the training and test data into a dict
# that associates each character with a unique number (e.g., {'a': 1, 'b': 2,
# etc.), then add an unknown ("[UNK]") token to handle characters that might
# not have shown up in the training data and a blank token ("[PAD]") that's
# used by the CTC algorithm. (NB: Characters, not phonemes -- this is the
# output dimension of the linear layer that we add on top of the pre-trained
# XLS-R model)
vocab_list = list(set(srs_vocab_train["vocab"][0]) | \
                  set(srs_vocab_valid["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

# Save the vocabulary to JSON, then use that to instantiate a tokenizer that
# knows about the above special characters and our word separator.
vocab_json = srs_model + '_vocab.json'
with open(vocab_json, 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = transformers.Wav2Vec2CTCTokenizer(vocab_json, \
    unk_token = "[UNK]", pad_token = "[PAD]", word_delimiter_token = "|")


# Now set up a feature extractor that uses the same 16KHz sample rate that
# XLS-R was pretrained on, then a processor that wraps the feature extractor
# and tokenizer for easier calling.
feature_extractor = transformers.Wav2Vec2FeatureExtractor(feature_size = 1, \
    sampling_rate = 16000, padding_value = 0.0, do_normalize = True, \
    return_attention_mask = True)

processor = transformers.Wav2Vec2Processor(\
    feature_extractor = feature_extractor, tokenizer = tokenizer)

processor.save_pretrained(srs_model)

# Now preprocess the data, loading in all of the audio clips as a one-
# dimensional array (and saving their sample rates and the original target
# text (= sentence) at the same time).  (This ditches 'phonemes' here; if
# that information is needed later, this is where it's dropped)
def speech_file_to_array_fn(batch):
    # FIXME: de-hardcode path
    speech_array, sampling_rate = \
        torchaudio.load("tsuutina_data/wav/" + batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

srs_train_data = srs_train_data.map(speech_file_to_array_fn, \
    remove_columns = srs_train_data.column_names)

srs_valid_data = srs_valid_data.map(speech_file_to_array_fn, \
    remove_columns = srs_valid_data.column_names)

# We could resample the audio here, but our training data is already available
# in 16KHz.

# Now call the processor to extract the 'input_values' that the models expects
# to use in training from the loaded audio, then encode our target text as
# label IDs (i.e., as character sequences in our vocabulary).  (Our basic
# processor only does normalization of the audio for now, but in theory, other
# features could be extracted here)
"""
def prepare_dataset_not_batched(batch):
    # Check that all files have the correct sampling rate.
#    assert (
#        len(set(batch["sampling_rate"])) == 1
#    ), f"Make sure all inputs have the same sampling rate "\
#        "of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], \
        sampling_rate = batch["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids

    return batch

srs_train_data = srs_train_data.map(prepare_dataset_not_batched, \
    remove_columns = srs_train_data.column_names)

srs_valid_data = srs_valid_data.map(prepare_dataset_not_batched, \
    remove_columns = srs_valid_data.column_names)
"""

def prepare_dataset(batch):
    # Check that all files have the correct sampling rate.
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of "\
        "{processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], \
        sampling_rate = batch["sampling_rate"][0]).input_values
    batch["input_length"] = list(map(len, batch["input_values"]))
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

srs_train_data = srs_train_data.map(prepare_dataset, \
    remove_columns = srs_train_data.column_names, batch_size = 8, \
    num_proc = 4, batched = True)

srs_valid_data = srs_valid_data.map(prepare_dataset, \
    remove_columns = srs_valid_data.column_names, batch_size = 8, \
    num_proc = 4, batched = True)

# OK, with input values and labels now in hand, we're ready to start training!
# (The following starts by defining a data collator that will pad the inputs
# it's given for training batches)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: transformers.Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: \
                 List[Dict[str, Union[List[int], torch.Tensor]]]) -> \
                 Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for \
            feature in features]
        label_features = [{"input_ids": feature["labels"]} for \
            feature in features]

        batch = self.processor.pad(
            input_features,
            padding = self.padding,
            max_length = self.max_length,
            pad_to_multiple_of = self.pad_to_multiple_of,
            return_tensors = "pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding = self.padding,
                max_length = self.max_length_labels,
                pad_to_multiple_of = self.pad_to_multiple_of_labels,
                return_tensors = "pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(\
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor = processor, \
    padding = True)

wer_metric = datasets.load_metric("wer")
cer_metric = datasets.load_metric("cer")

def compute_metrics(pred):
    # The model returns a sequence of logit vectors, with each logit vector
    # containig the log-odds for each element of the vocabulary we defined
    # earlier.  We're most interested in the top-ranked prediction for each
    # such vector, so we take the argmax of each logit to get the top-ranked
    # label.
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis = -1)

    # We then turn our labels back into our original strings, replacing -100
    # with the pad_token_id, then decoding the rest.
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)

    # We do not want to group tokens when computing the metrics.
    label_str = processor.batch_decode(pred.label_ids, group_tokens = False)
    wer = wer_metric.compute(predictions = pred_str, references = label_str)
    cer = cer_metric.compute(predictions = pred_str, references = label_str)

#    print("WER : ", wer)
    return {"cer": cer}

# We can now load the pre-trained Wav2Vec2-XLS-R-300M checkpoint, making sure
# that our tokenizer's pad_token_id matches what that model uses.
model = transformers.Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout = 0.0,    # CM: 0.1
    hidden_dropout = 0.0,       # CM: 0.1
    feat_proj_dropout = 0.0,
    mask_time_prob = 0.05,      # CM: 0.075
    layerdrop = 0.0,            # CM: 0.1
    gradient_checkpointing = True,
    ctc_loss_reduction = "mean", 
    pad_token_id = processor.tokenizer.pad_token_id,
    vocab_size = len(processor.tokenizer),
)

model.freeze_feature_extractor()

# Finally, define all of the training parameters, then pass everything along
# to the trainer to do the actual model training. (HF: parameter values in the
# HuggingFace tutorial; CM-G: parameter values from Cecile Macaire's work on
# GitHub; CM-P: parameter values from Cecile Macaire's report)

training_args = transformers.TrainingArguments(
  output_dir="./" + srs_model,
  group_by_length = True,
  per_device_train_batch_size = 6,  # HF: 16, CM-G/P: 8
  gradient_accumulation_steps = 2,
  evaluation_strategy = "steps",
  num_train_epochs = 60,            # HF: 30
  fp16 = True,                      # HF/CM: True
  save_steps = 100,                 # HF: 400, CM-G: 1000
  eval_steps = 50,                  # HF: 400
  logging_steps = 50,               # HF: 400
  learning_rate = 3e-4,
  warmup_steps = 500,
  save_total_limit = 2,
)

trainer = transformers.Trainer(
    model = model,
    data_collator = data_collator,
    args = training_args,
    compute_metrics = compute_metrics,
    train_dataset = srs_train_data,
    eval_dataset= srs_valid_data,
    tokenizer = processor.feature_extractor,
)

trainer.train()
