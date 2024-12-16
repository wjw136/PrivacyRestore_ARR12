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
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    BertModel,
)
from collections import defaultdict

from tqdm import tqdm
import sys
sys.path.append('../')
print(sys.path)
from common_utils import gen_maskEvi_dict, gen_question, format_questions_mc, compute_rouge, format_ans
import llama
import transformers
import torch.nn as nn

import datetime
import argparse
import pickle

current_time = datetime.datetime.now()
# 将时间格式化为字符串
time_string = current_time.strftime("%Y-%m-%d-%H:%M:%S")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/data/transformers/Llama-2-7b-chat-hf"
    )
    bank_interventions_dir: Optional[str] = field(
        default="/data/transformers/Llama-2-7b-chat-hf"
    )

    auxiliary_model_type:Optional[str] = field(
        default="bert"
    )
    auxiliary_model_name_or_path: Optional[str] = field(
        default=""
    )

    init_inter:bool = field(default=True)
    use_agg:bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    training_batch_size: int = field(default=128)
    evaluation_batch_size: int = field(default=128)
    
    gradients_accumulation_steps: Optional[int] = field(default=100)
    output_dir: Optional[str] = field(default="")
    warm_up_steps: Optional[int] = field(default=0)
    model_max_length: Optional[int] = field(default=0)
    eval_size: Optional[int] = field(default=0)

    use_weight: bool = field(default=True)

    epsilon: Optional[int] = field(default=75)

    use_kl: bool = field(default=False)
    use_neg: bool = field(default=False)

    symptom_antecedent_data_path: Optional[str] = field(default="")
    sym2level: Optional[str] = field(default="")

    


class SupervisedDataset(Dataset):

    def __init__(self,
                 data_path,
                 tokenizer,
                 model_max_length = 2047,
                 user_prefix="",
                 assistant_prefix="",
                 train_on_inputs=False,
                 ignore_index=-100,
                 start=None,
                 end=None,
                 ):
        super(SupervisedDataset, self).__init__()
        self.data = []
        with open(data_path, 'r') as data_file:
            for line in data_file:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.train_on_inputs = train_on_inputs
        self.ignore_index = ignore_index

        self.processed_data = self.preprocessing(self.data)
        random.shuffle(self.processed_data) # 打乱训练顺序

    def __len__(self):
        return len(self.processed_data)

    def preprocessing(self, examples:list):
        return_data = []
        for example in examples:
            return_data.append(
                {
                    'restore_evidences': example['restore_evidences'],
                    'keep_evidences': [item for item in example['symptoms'] if item not in example['restore_evidences']],
                    'question_init': example['question_init'],
                    "question_mask": example['question_mask'],
                    'correct_answer': example['correct_answer'],
                    "output_init": example['output_init'],
                    "output_mask": example['output_mask']
                }
            )
        return return_data

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, List]]:
        return self.processed_data[idx]
    

class Metric(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_pred):    
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        pred_text = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        pred_text = [t.split("\nHelpful:")[-1].strip().lower() for t in pred_text]
        label_text = self.tokenizer.batch_decode([[l for l in label if l >= 0] for label in labels], skip_special_tokens=True)
        label_text = [t.strip().lower() for t in label_text]

        # print(list(zip(pred_text, label_text)))
        
        accuracy = sum(
            [1 if p == l else 0 
             for p, l in zip(pred_text, label_text)]
        ) / len(label_text)
        
        result = {
            "accuracy": accuracy * 100,
        }
        
        return result


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_sub_model(main_model, path):
    for k, v in main_model.named_parameters():
        if k == "model.head_out_intervention.weight":
            # 假设我们只想保存模型的第一个卷积层的参数
            weight_data = v.data
            # 创建一个字典保存所需的参数
            params_to_save = {k: weight_data}
            # 保存到文件
            torch.save(params_to_save, path)

def train():
    def str2bool(s):
        return s.lower() in ("true", "t", "yes", "y", "1", "True")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--bank_interventions_dir", type=str, default="", help='local directory with model data')
    parser.add_argument("--auxiliary_model_name_or_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--inter_heads_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--evidences_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--symptom_antecedent_data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--submodule_path", type=str, default="", help='local directory with model data')

    parser.add_argument("--use_weight", type=str2bool, default="true", help='local directory with model data')
    parser.add_argument("--use_agg", type=str2bool, default="true", help='local directory with model data')


    parser.add_argument("--epoch", type=int, default=5, help='local directory with model data')

    parser.add_argument("--seed", type=int, default=42, help='local directory with model data')
    parser.add_argument("--eval_size", type=int, default=200, help='local directory with model data')
    parser.add_argument("--learning_rate", type=float, default=1e-04, help='local directory with model data')
    parser.add_argument("--weight_decay", type=float, default=0.001, help='local directory with model data')
    parser.add_argument("--warm_up_steps", type=int, default=0, help='local directory with model data')
    parser.add_argument("--gradients_accumulation_steps", type=int, default=8, help='local directory with model data')
    parser.add_argument("--eval_steps", type=int, default=256, help='local directory with model data')


    parser.add_argument("--output_dir", type=str, default="", help='local directory with model data')
    args = parser.parse_args()
    print(args)

    # 解析超参数配置文件
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    # 固定随机种子
    set_seed(args.seed)


    # 设置训练时的 cuda 设备
    # 单卡或多卡单进程（DP）时，device_map 为 auto
    # 多卡多进程（DDP）时，device_map 需要对应到进程编号
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 载入模型分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    # 对于自回归生成式模型，设置 padding 策略为左侧
    tokenizer.padding_side = "left"

    
    main_model = llama.LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            load_in_8bit=False,
            device_map=device,
            bank_intervention_dir = args.bank_interventions_dir,
            use_agg = args.use_agg,
            inter_heads_path = args.inter_heads_path,
            evidences_path = args.evidences_path
    )
    if args.submodule_path != "":
        print(f"use sub model from {args.submodule_path}")
        param_to_load = torch.load(args.submodule_path, map_location="cuda:0")
        # print(param_to_load.keys())
        model_dict = main_model.state_dict()
        # print(model_dict.keys())
        model_dict.update(param_to_load)
        main_model.load_state_dict(model_dict)
       
    
    auxiliary_model = BertModel.from_pretrained(
        args.auxiliary_model_name_or_path,
        device_map=device,
    )
    auxiliary_tokenizer = AutoTokenizer.from_pretrained(
        args.auxiliary_model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    auxiliary_model.eval()
    for name, param in main_model.named_parameters():
        if 'head_out_intervention' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = 0
    all_param = 0
    for _, param in main_model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


    # 准备数据集-->level==3
    train_dataset = SupervisedDataset(
        args.data_path, tokenizer,
    )
    eval_dataset = None
    if args.eval_size > 0:
        train_dataset, eval_dataset = random_split(
            train_dataset,
            lengths=[len(train_dataset) - args.eval_size, args.eval_size],
        )
    

    
    # create optimizer
    optimizer = AdamW(main_model.parameters(), lr=args.learning_rate, correct_bias=False, weight_decay=args.weight_decay) 
    scheduler = transformers.get_cosine_schedule_with_warmup(                                    
        optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps = args.epoch * (len(train_dataset))
    )
    


    def eval_model():
        eval_rl_list = []
        eval_loss_list = []
        with torch.no_grad():
            for idx, example in enumerate(tqdm(eval_dataset)):

                # compute rouge
                gen_str_init = example['output_init']
                gen_str_inter = main_model.intervention_generate(
                    input = example,
                    tokenizer = tokenizer,
                    device = device,
                    auxiliary_model = auxiliary_model,
                    auxiliary_tokenizer = auxiliary_tokenizer,
                )
                r1, r2, rL = compute_rouge(gen_str_init, gen_str_inter)
                print(f"GENERATE ANSWER-{idx}-r1={r1}-r2={r2}-rl={rL}:\n[INTER]: {gen_str_inter}\n[INIT]: {gen_str_init}", flush = True)
                eval_rl_list.append(rL)

                # compute loss:
                orpo_loss, LM_loss = main_model.forward_intervention_orpo_client(
                    input = example,
                    tokenizer = tokenizer,
                    device = device,
                    auxiliary_model = auxiliary_model,
                    auxiliary_tokenizer = auxiliary_tokenizer
                )
                eval_loss_list.append(LM_loss.detach().cpu().numpy())


        return np.mean(eval_rl_list), np.mean(eval_loss_list)
    
    # handby training process
    best_eval_rl = -np.inf
    best_eval_loss = np.inf
    past_eval_rl = np.inf
    past_eval_loss = np.inf
    for epoch in range(args.epoch):
        orpo_loss_list = []
        LM_loss_list = []
        LM_loss_extra_list = []
        train_loss = 0
        for idx, example in enumerate(tqdm(train_dataset)):
            # print(example.keys())
            print(f"train example: restore_evidences: {example['restore_evidences']}", flush=True)        
            orpo_loss, LM_loss = main_model.forward_intervention_orpo_client(
                input = example,
                tokenizer = tokenizer,
                device = device,
                auxiliary_model = auxiliary_model,
                auxiliary_tokenizer = auxiliary_tokenizer
            )
            train_loss+=orpo_loss
            train_loss+=LM_loss
            # train_loss+=LM_loss_extra
            orpo_loss_list.append(orpo_loss.detach().cpu().numpy())
            LM_loss_list.append(LM_loss.detach().cpu().numpy())
            # LM_loss_extra_list.append(LM_loss_extra.detach().cpu().numpy())

            if idx % args.gradients_accumulation_steps == 0 and idx!=0:
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()    
                torch.cuda.empty_cache()
                train_loss = 0
                print(f"[TRAIN]: orpo loss: {np.mean(orpo_loss_list)}, LM loss: {np.mean(LM_loss_list)}")
                orpo_loss_list.clear()
                LM_loss_list.clear()
                LM_loss_extra_list.clear()

            # log and choose to save model
            if idx % args.eval_steps == 0 and idx != 0:

                main_model.eval()
                eval_rl, eval_loss = eval_model()
                main_model.train()
                if eval_rl >= best_eval_rl:
                    model_path = os.path.join(args.output_dir, f"{time_string}_head_out_intervention.pth")
                    print('\r\r', f"[New] best eval_rl model saved to {model_path}", flush=True)
                    best_eval_rl = eval_rl
                    save_sub_model(main_model, model_path)
                elif eval_rl >= best_eval_rl * 0.95 and eval_loss <= best_eval_loss:
                    model_path = os.path.join(args.output_dir, f"{time_string}_head_out_intervention.pth")
                    print('\r\r', f"[New] best eval_rl model saved to {model_path}", flush=True)
                    best_eval_rl = eval_rl
                    save_sub_model(main_model, model_path)

                # logging
                print('\r', "EPOCH {}, STEP {}/{} [EVAL] RougeL: {:.4f}".format(
                    epoch, idx+1, len(train_dataset), 
                    eval_rl
                    ), flush=True)
                if eval_rl < past_eval_rl and eval_loss > past_eval_loss:
                    print("!!! Early Breaking")
                    break
                else:
                    past_eval_rl = eval_rl
                    past_eval_loss = eval_loss




if __name__ == "__main__":
    train()