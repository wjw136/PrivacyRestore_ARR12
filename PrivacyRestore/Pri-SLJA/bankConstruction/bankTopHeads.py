import torch
import numpy as np
import pickle
import os
import numpy as np
import argparse

import sys
sys.path.append('../')
print(sys.path)
from common_utils import numpy_load, layer_head_to_flattened_idx
from transformers import LlamaForCausalLM
from collections import defaultdict


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank_dir', type=str, default='', help='feature bank for calculating std along direction')
    parser.add_argument("--mask_symptoms_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--output_dir", type=str, default="", help='local directory with model data')
    parser.add_argument('--num_heads', type=int, default=0, help='device')

    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--device', type=int, default=0, help='device')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        print(args.output_dir)

    # create model to obtain number of layers and heads
    device= torch.device('cuda:{}'.format(args.device))
    model = LlamaForCausalLM.from_pretrained(args.model_dir, low_cpu_mem_usage = True, torch_dtype=torch.float32, device_map=device)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    with open(args.mask_symptoms_path, 'rb') as mask_symptom_file:
        mask_evidences = pickle.load(mask_symptom_file)
    evidences2id = {evidence : index for index, evidence in enumerate(mask_evidences) }
    
    with open(os.path.join(args.output_dir, "evidences_list.pkl"), 'wb') as output_file:
        pickle.dump(mask_evidences, output_file)
    print(f"mask evidence len: {len(mask_evidences)}")



    list_accuracy = defaultdict(list) # 用于记录每个sym下的acc
    heads_list = [] # 用于统计head的频率
    for symptom in mask_evidences:
        symptom_dir = os.path.join(args.bank_dir, f"{evidences2id[symptom]}/")
        top_heads_path = os.path.join(symptom_dir, "top_heads.npy")
        top_heads = numpy_load(top_heads_path)
        heads_list.extend([layer_head_to_flattened_idx(top_head[0], top_head[1], num_heads) for top_head in top_heads])
        for idx, top_head in enumerate(top_heads):
            list_accuracy[layer_head_to_flattened_idx(top_head[0], top_head[1], num_heads)].append(args.num_heads - idx) # 48 = num_to_intervene ：代表排位，越大，排位越大，排位是probe的train出来的eval的acc

    # print(len(set(heads_list)))
    head_accuracy = { key : np.mean(accs) for key, accs in list_accuracy.items()}
    common_top_heads = []
    for head, acc in sorted(head_accuracy.items(), key=lambda v: v[1], reverse=True):
        if len(common_top_heads) < args.num_heads:
            common_top_heads.append(head)
    
    print(common_top_heads)
    with open(os.path.join(args.output_dir, "common_top_head.pkl"), 'wb') as output_top_heads_file:
        pickle.dump(common_top_heads, output_top_heads_file)




if __name__ == "__main__":
    main()