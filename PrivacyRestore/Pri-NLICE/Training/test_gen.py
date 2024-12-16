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
    TaskType
)

from tqdm import tqdm
import sys
from peft import PeftModel
sys.path.append('../')
print(sys.path)
import llama
import transformers

import datetime
import pandas as pd
current_time = datetime.datetime.now()
# 将时间格式化为字符串
time_string = current_time.strftime("%Y-%m-%d-%H:%M:%S")
import argparse

from common_utils import compute_rouge

sys.path.append('../../')
print(sys.path)
from gpt_utils.gpt_eval import GPTJudge
from dx_privacy.dx_utils import DX_privacy


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
    use_agg:bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )


@dataclass
class EvaluateArguments:
    lora_path: Optional[str] = field(default="")
    weight_path: Optional[str] = field(default="")
    task_type: Optional[Union[str, TaskType]] = field(default="CAUSAL_LM")
    seed: Optional[int] = field(default=42)
    epsilon: Optional[int] = field(default=75)
    use_lora: bool = field(default=False)

    use_weight: bool = field(default=True)
    output_dir: Optional[str] = field(default="")


class EvaluateDataset(Dataset):

    def __init__(self,
                 data_path
                 ):
        super(EvaluateDataset, self).__init__()
        self.data = []
        with open(data_path, 'r') as data_file:
            for line in data_file:
                self.data.append(json.loads(line))
        # random.shuffle(self.data)
    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        return {
            'restore_evidences': example['restore_evidences'],
            'keep_evidences': [item for item in example['symptoms'] if item not in example['restore_evidences']],
            'question_init': example['question_init'],
            "question_mask": example['question_mask'],
            'correct_answer': example['correct_answer'],
            'options': example['options']
        }

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, List]]:
        return self.preprocessing(self.data[idx])
    

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


def train():
    def str2bool(s):
        return s.lower() in ("true", "t", "yes", "y", "1", "True")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--submodule_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--use_weight", type=str2bool, default="true", help='local directory with model data')
    parser.add_argument("--use_agg", type=str2bool, default="true", help='local directory with model data')
    parser.add_argument("--score_mask", type=str2bool, default="true", help='local directory with model data')
    parser.add_argument("--bank_interventions_dir", type=str, default="", help='local directory with model data')
    parser.add_argument("--auxiliary_model_name_or_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--inter_heads_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--evidences_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--seed", type=int, default=42, help='local directory with model data')

    parser.add_argument("--output_ans_dir", type=str, default="", help='local directory with model data')
    args = parser.parse_args()
    print(args)

    # 解析超参数配置文件
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, EvaluateArguments)
    )
    model_args, data_args, eval_args = parser.parse_json_file("./config/gen_args_7b.json")

    data_args.data_path = args.data_path
    eval_args.use_weight = args.use_weight
    model_args.use_agg = args.use_agg
    eval_args.submodule_path = args.submodule_path
    model_args.bank_interventions_dir = args.bank_interventions_dir
    model_args.model_name_or_path = args.model_name_or_path
    if args.auxiliary_model_name_or_path != "":
        model_args.auxiliary_model_name_or_path = args.auxiliary_model_name_or_path
    # 固定随机种子
    set_seed(args.seed)


    # 设置训练时的 cuda 设备
    # 单卡或多卡单进程（DP）时，device_map 为 auto
    # 多卡多进程（DDP）时，device_map 需要对应到进程编号
    device = torch.device("cuda:0")
    # 载入模型分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    # 对于自回归生成式模型，设置 padding 策略为左侧
    tokenizer.padding_side = "left"

    

    main_model = llama.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        load_in_8bit=False,
        device_map=device,
        bank_intervention_dir = model_args.bank_interventions_dir,
        use_agg = model_args.use_agg,
        inter_heads_path = args.inter_heads_path,
        evidences_path = args.evidences_path
    )
    if args.submodule_path != "":
        if model_args.use_agg:
            print(f"use sub model from {args.submodule_path}")
            param_to_load = torch.load(args.submodule_path, map_location="cuda:0")
            # print(param_to_load.keys())
            model_dict = main_model.state_dict()
            # print(model_dict.keys())
            model_dict.update(param_to_load)
            main_model.load_state_dict(model_dict)
        else:
            print(f"use sub model from {args.submodule_path}")
            param_to_load = torch.load(args.submodule_path, map_location="cuda:0")

            param = param_to_load['model.head_out_intervention.weight'].cpu().detach().numpy().transpose(1, 0)
            print(param.shape)
            

            interventions_list = [None for i in range(main_model.model.evidences_len)]
            for idx, evidence in enumerate(main_model.model.bank_evidences):
                symptom_dir = os.path.join(main_model.config.bank_intervention_dir, f"{evidence.lower()}/")
                directions_path = os.path.join(symptom_dir, 'directions.npy')
                symptom_com_directions = np.load(directions_path)
                param_per_evidence = param[main_model.model.evidences2id[evidence] * len(main_model.model.valid_intervention_heads): \
                                           (main_model.model.evidences2id[evidence] + 1) * len(main_model.model.valid_intervention_heads) ]
                evidence_intervention = main_model.get_intervention_hybri(main_model.model.intervention_heads, main_model.model.valid_intervention_heads,
                                                               symptom_com_directions, 
                                                               param_per_evidence)
                interventions_list[main_model.model.evidences2id[evidence]] = evidence_intervention
            
            interventions_tensor = torch.cat(interventions_list, dim=0).transpose(1, 0)
            main_model.model.head_out_intervention.load_state_dict({
                "weight": interventions_tensor
            })



    if model_args.auxiliary_model_type == "bert":
        auxiliary_model = BertModel.from_pretrained(
            model_args.auxiliary_model_name_or_path,
            device_map=device,
        )
        auxiliary_tokenizer = AutoTokenizer.from_pretrained(
            model_args.auxiliary_model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

    else:
        auxiliary_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            load_in_8bit=False,
            device_map=device
        )
        auxiliary_tokenizer = AutoTokenizer.from_pretrained(
            model_args.auxiliary_model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

    print(f'load auxiliary model from {auxiliary_model.name_or_path}')
    auxiliary_model.eval()



    # 准备数据集
    evaluate_dataset = EvaluateDataset(
        data_args.data_path
    )

    judger = GPTJudge()
    
    r1_inter_list = []
    r2_inter_list = []
    rl_inter_list = []
    r1_mask_list = []
    r2_mask_list = []
    rl_mask_list = []
    gpt_scores_inter = []
    gpt_scores_mask = []
    main_model.eval()
    output_ans_path = os.path.join(args.output_ans_dir, f"{time_string}.json")
    print(f"[SETTING]-- Output ans into {output_ans_path}")
    with open(output_ans_path, 'w') as output_ans_file:
        with torch.no_grad():
            for idx, data in enumerate(tqdm(evaluate_dataset)):
                # if idx < 100:
                #     continue
                # if idx % 100 == 0 and idx != 0 :
                #     break
                if idx % 100 == 0:
                    print(f">>>> {idx}/{len(evaluate_dataset)}")

                dx_privacy = DX_privacy(75)


                intervention_gen_str= main_model.intervention_generate(data, tokenizer, device = device,
                                                                auxiliary_model = auxiliary_model, auxiliary_tokenizer=auxiliary_tokenizer,
                                                                dx_privacy = dx_privacy,
                                                                use_weight = args.use_weight)

                init_gen_str= main_model.base_generate(data['question_init'], tokenizer,device = device,)

                if args.score_mask:
                    mask_gen_str= main_model.base_generate(data['question_mask'], tokenizer, device = device,)

                r1_inter, r2_inter, rl_inter= compute_rouge(init_gen_str, intervention_gen_str)
                if args.score_mask:
                    r1_mask, r2_mask, rl_mask = compute_rouge(init_gen_str, mask_gen_str)

                r1_inter_list.append(r1_inter)
                r2_inter_list.append(r2_inter)
                rl_inter_list.append(rl_inter)

                if args.score_mask:
                    r1_mask_list.append(r1_mask)
                    r2_mask_list.append(r2_mask)
                    rl_mask_list.append(rl_mask)

                # GPT score:
                if args.score_mask:
                    score_mask, reason_mask = judger.single_score(data['question_init'], mask_gen_str)
                score_inter, reason_inter = judger.single_score(data['question_init'], intervention_gen_str)
                if args.score_mask:
                    gpt_scores_mask.append(score_mask)
                gpt_scores_inter.append(score_inter)

                print("======================================================")
                print(data['restore_evidences'])
                print(data['keep_evidences'])
                print(f"[QUESTION]: {data['question_init']}")
                print(f"[CORRECT ANSWER]: {data['correct_answer']}")
                print(f"[METRIC_inter]: r1-{r1_inter}, r2-{r2_inter}, rl-{rl_inter}")
                if args.score_mask:
                    print(f"[METRIC_mask]: {r1_mask}, r2-{r2_mask}, rl-{rl_mask}")
                print(f"[Inter GPT]: {score_inter}---{reason_inter}")
                if args.score_mask:
                    print(f"[Mask GPT]: {score_mask}---{reason_mask}")
                    outcome = "GOOD" if rl_inter >= rl_mask else "BAD"
                    print(f"[OUTCOME]: {outcome}")
                    print(f"[INTERVENTION ANSWER]: {intervention_gen_str}\n[INIT RESULT]: {init_gen_str}\n[MASK RESULT]: {mask_gen_str}", flush=True)
                else:
                    print(f"[INTERVENTION ANSWER]: {intervention_gen_str}\n[INIT RESULT]: {init_gen_str}", flush=True)


                # output_ans_file.write(
                #     json.dumps(
                #         {
                #             "init_gen_str": init_gen_str,
                #             "mask_gen_str": mask_gen_str,
                #             "intervention_gen_str": intervention_gen_str,
                #             "correct_answer": data['correct_answer'],
                #             "restore_evidences": data['restore_evidences'],
                #             "keep_evidences": data['keep_evidences'],
                #             "question_init": data['question_init'],
                #             "question_mask": data['question_mask'],
                #             "inter_score_gpt": score_inter,
                #             "inter_reason_gpt": reason_inter, 
                #             "mask_score_gpt": score_mask,
                #             "mask_reason_gpt": reason_mask
                #         }
                #     ) + '\n'
                # )
                # output_ans_file.flush()

    print(f"[SETTING]-- Output ans into {output_ans_file}")
    print("****************************************")
    if args.score_mask:
        print(f"[MASK]====>")
        print(f"r1: {np.mean(r1_mask_list)}")
        print(f"r2: {np.mean(r2_mask_list)}")
        print(f"rl: {np.mean(rl_mask_list)}")
        print(f"GPT: {np.mean(gpt_scores_mask)}")
    print(f"[INTER]====>")
    print(f"r1: {np.mean(r1_inter_list)}")
    print(f"r2: {np.mean(r2_inter_list)}")
    print(f"rl: {np.mean(rl_inter_list)}")
    print(f"GPT: {np.mean(gpt_scores_inter)}")
        


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    train()