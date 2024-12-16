import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import warnings

warnings.filterwarnings("ignore")

import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union

import numpy as np

import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BertModel
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from tqdm import tqdm
import sys
from peft import PeftModel
sys.path.append('../')
print(sys.path)
import llama
import transformers

import datetime
current_time = datetime.datetime.now()
# 将时间格式化为字符串
time_string = current_time.strftime("%Y-%m-%d-%H:%M:%S")
import argparse


def train():

    device = torch.device("cuda:1")


    main_model = llama.LlamaForCausalLM.from_pretrained(
        "",
        trust_remote_code=True,
        load_in_8bit=False,
        device_map=device,
        bank_intervention_dir = ""
    )

    for k, v in main_model.named_parameters():
        print(k)
    

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    train()
