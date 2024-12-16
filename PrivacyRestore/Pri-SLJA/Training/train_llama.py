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
from common_utils import format_questions_legal, compute_rouge
import llama
import transformers
import torch.nn as nn

import datetime
import argparse
import pickle

current_time = datetime.datetime.now()
# 将时间格式化为字符串
time_string = current_time.strftime("%Y-%m-%d-%H:%M:%S")
    

class SupervisedDataset(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 model_max_length = 2047,
                 train_on_inputs=False,
                 ignore_index=-100,
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
        # random.shuffle(self.processed_data) # 打乱训练顺序
    def __len__(self):
        return len(self.processed_data)

    def preprocessing(self, examples:list):
        return_data = []
        for example in examples:
            input_ids = []
            labels = []
            
            # example['restore_evidences'] = example['minor_premise_atoms']['objective_atoms']['private_objective_atoms'] + \
            #                     example['minor_premise_atoms']['subject_atoms']['private_subject_atoms'] + \
            #                     example['minor_premise_atoms']['subjective_atoms']['private_subjective_atoms'] + \
            #                     example['minor_premise_atoms']['object_atoms']['private_object_atoms']
            
            # example['keep_evidences'] = example['minor_premise_atoms']['objective_atoms']['non_private_objective_atoms'] + \
            #                     example['minor_premise_atoms']['subject_atoms']['non_private_subject_atoms'] + \
            #                     example['minor_premise_atoms']['subjective_atoms']['non_private_subjective_atoms'] + \
            #                     example['minor_premise_atoms']['object_atoms']['non_private_object_atoms']
            
            if len(example['restore_evidences']) == 0:
                continue
            
            user_text = format_questions_legal(example['question_mask'])
            assistant_text = example["answer"] + '.'
            user_ids = self.tokenizer.encode(user_text)
            assistant_ids = self.tokenizer.encode(assistant_text)
            input_ids += user_ids + assistant_ids
            labels += [self.ignore_index] * len(user_ids) + assistant_ids
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

            if len(input_ids) > self.model_max_length:
                print(len(input_ids))
            if len(input_ids) != len(labels):
                print(len(input_ids), len(labels))
            if any([x < 0 for x in input_ids]):
                print(input_ids)
            input_ids = input_ids[: self.model_max_length]
            labels = labels[: self.model_max_length]

            for i in example['keep_evidences'] + example['restore_evidences']:
                if i not in example['question_init']:
                    print(i)
                    print(example)
                    ab+=1
            return_data.append(
                {
                    'restore_evidences': example['restore_evidences'],
                    'keep_evidences': example['keep_evidences'],
                    'question_init': example['question_init'],
                    "question_mask": example['question_mask'],
                    'correct_answer': example['answer'],
                    "output_init": example['output_init'] if 'output_init' in example.keys() else "",
                    "output_mask": example['output_mask'] if 'output_mask' in example.keys() else ""
                }
            )
        return return_data

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, List]]:
        return self.processed_data[idx]
    


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


    # 准备数据集
    train_dataset = SupervisedDataset(args.data_path, tokenizer)
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
    cid = 0
    past_eval_rl = np.inf
    past_eval_loss = np.inf
    for epoch in range(args.epoch):
        orpo_loss_list = []
        LM_loss_list = []
        LM_loss_extra_list = []
        train_loss = 0
        for idx, example in enumerate(tqdm(train_dataset)):
            print(f"train example: {idx}--restore_evidences: {example['restore_evidences']}", flush=True)        
            orpo_loss, LM_loss = main_model.forward_intervention_orpo_client(
                input = example,
                tokenizer = tokenizer,
                device = device,
                auxiliary_model = auxiliary_model,
                auxiliary_tokenizer = auxiliary_tokenizer
            )
            train_loss+=orpo_loss
            # train_loss+=LM_loss
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
                candidate_model_path = os.path.join(args.output_dir, f"candidate{cid}_{time_string}_head_out_intervention.pth")
                cid+=1
                save_sub_model(main_model, candidate_model_path)
                print('\r\r', f"[New] candidate-{cid-1} model saved to {candidate_model_path}", flush=True)
                if eval_rl >= best_eval_rl:
                    model_path = os.path.join(args.output_dir, f"{time_string}_head_out_intervention.pth")
                    print('\r\r', f"[New] best eval_rl model saved to {model_path}", flush=True)
                    best_eval_rl = eval_rl
                    save_sub_model(main_model, model_path)
                elif eval_rl >= best_eval_rl * 0.85 and eval_loss < best_eval_loss:
                    model_path = os.path.join(args.output_dir, f"{time_string}_head_out_intervention.pth")
                    print('\r\r', f"[New] best eval_rl model saved to {model_path}", flush=True)
                    best_eval_rl = eval_rl
                    save_sub_model(main_model, model_path)
                # logging
                print('\r', "EPOCH {}, STEP {}/{} [EVAL] RougeL: {:.4f}, Eval Loss: {:.4f}".format(
                    epoch, idx+1, len(train_dataset), 
                    eval_rl,
                    eval_loss
                    ), flush=True)

                if eval_rl < past_eval_rl and eval_loss > past_eval_loss:
                    print("!!! Early Breaking")
                    break
                else:
                    past_eval_rl = eval_rl
                    past_eval_loss = eval_loss




if __name__ == "__main__":
    train()
