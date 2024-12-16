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
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BertModel
)
from peft import (
    TaskType
)

from tqdm import tqdm
import sys
sys.path.append('../')
print(sys.path)
import llama

import datetime
current_time = datetime.datetime.now()
# 将时间格式化为字符串
time_string = current_time.strftime("%Y-%m-%d-%H:%M:%S")
import argparse
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
    submodule_path: Optional[str] = field(default="")


class EvaluateDataset(Dataset):

    def __init__(self,
                 data_path
                 ):
        super(EvaluateDataset, self).__init__()
        self.data = []
        with open(data_path, 'r') as data_file:
            for line in data_file:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        return {
            'restore_evidences': example['restore_evidences'],
            'keep_evidences': example['keep_evidences'],
            'question_init': example['question_init'],
            'question_mask': example['question_mask'],
            'correct_answer': example['answer'],
            'options': example['options'].values()
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
    parser.add_argument("--bank_interventions_dir", type=str, default="", help='local directory with model data')
    parser.add_argument("--auxiliary_model_name_or_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--inter_heads_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--evidences_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--seed", type=int, default=42, help='local directory with model data')
    parser.add_argument("--epislon", type=float, default=75, help='local directory with model data')
    
    args = parser.parse_args()
    print(args)

    # 解析超参数配置文件
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, EvaluateArguments)
    )
    model_args, data_args, eval_args = parser.parse_json_file("./config/mc_args_7b.json")

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
        bank_intervention_dir = args.bank_interventions_dir,
        use_agg = args.use_agg,
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
        args.data_path
    )
    


    total_mc1 = []
    total_mc2 = []
    main_model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(evaluate_dataset)):
            if idx % 100 == 0:
                print(f">>>> {idx}/{len(evaluate_dataset)}")
            dx_privacy = DX_privacy(args.epislon)

            mc1, mc2 = main_model.intervention_infer_mc_score(data, tokenizer, device = device,
                                                            auxiliary_model = auxiliary_model, auxiliary_tokenizer= auxiliary_tokenizer, 
                                                            dx_privacy= dx_privacy,
                                                            use_weight = args.use_weight)
            total_mc1.append(mc1)
            total_mc2.append(mc2)
            torch.cuda.empty_cache()
    

    print("++++++++++++++++++++++++++++++++++")
    print("\tAVERAGE SCORE")
    print(f"AVERAGE MC1: {np.average(total_mc1)}")
    print(f"AVERAGE MC2: {np.average(total_mc2)}")
    print("++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    train()
