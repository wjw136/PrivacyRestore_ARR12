import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

import sys
sys.path.append('../')
print(sys.path)
from utils import  get_top_heads, get_com_directions
from common_utils import get_interventions_dict, \
    pickle_save, numpy_save
import llama
import json
from collections import defaultdict
import shutil
import re
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--activations_dir', type=str, \
                        default='', \
                            help='feature bank for calculating std along direction')
    parser.add_argument('--bank_dir', type=str, \
                        default='', \
                            help='feature bank for calculating std along direction')
    parser.add_argument("--mode", type=str, default="med_dataset2", help='local directory with model data')
    
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)

    args = parser.parse_args()
    print(args)

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    # create model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    device= torch.device('cuda:{}'.format(args.device))
    model = llama.LlamaForCausalLM.from_pretrained(args.model_dir, low_cpu_mem_usage = True, torch_dtype=torch.float32, device_map=device,
                                                   bank_intervention_dir = "")
    
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads



    # load activations dataset
    probs_dict = defaultdict(list)
    directions_dict = defaultdict(list)
    heads_dict = defaultdict(list)

    # mode = "dataset2"
    index_path = os.path.join(args.activations_dir, f"index_{args.mode}.json")
    label_path = os.path.join(args.activations_dir, f"label_{args.mode}.npy")
    head_activations_path = os.path.join(args.activations_dir, f"head_wise_{args.mode}.npy")
    with open(index_path) as index_file:
        symptom_indexs = json.load(index_file)
    labels = np.load(label_path)
    head_activations = np.load(head_activations_path)
    head_activations = rearrange(head_activations, 'b l (h d) -> b l h d', h = num_heads) # b l h d
    print(head_activations.shape)
    
    # get directions and top head
    # use_center_of_mass:
    for symptom, symptom_index in symptom_indexs.items():
        print(f"process {symptom}")
        symptom_head_activations = [head_activations[idx] for idx in symptom_index]
        symptom_labels = [labels[idx] for idx in symptom_index]

        symptom_com_directions = get_com_directions(num_layers, num_heads, symptom, symptom_head_activations, symptom_labels)

        symptom_top_heads, symptom_probes = get_top_heads(symptom, symptom_head_activations, symptom_labels, \
                                        num_layers, num_heads, args.seed, num_layers * num_heads, args.val_ratio)

        print(f"symptom {symptom}, heads intervened example: {sorted(symptom_top_heads)}")

        # save directions and heads
        symptom_dir = os.path.join(args.bank_dir, f"{symptom.lower()}/")

        if os.path.exists(symptom_dir):
            shutil.rmtree(symptom_dir, )
        os.makedirs(symptom_dir)

        directions_path = os.path.join(symptom_dir, 'directions.npy')
        top_heads_path = os.path.join(symptom_dir, "top_heads.npy")
        probes_path = os.path.join(symptom_dir, 'probes.pkl')
        numpy_save(symptom_com_directions, directions_path)
        numpy_save(symptom_top_heads, top_heads_path)
        pickle_save(symptom_probes, probes_path)

        directions_dict[symptom] = symptom_com_directions
        probs_dict[symptom] = symptom_probes
        heads_dict[symptom] = symptom_top_heads
    
        
    # group into intervention_dict
    interventions_dict = defaultdict(None)

    for symptom, symptom_index in symptom_indexs.items():
        print(f"process {symptom}")

        symptom_top_heads = heads_dict[symptom]
        symptom_probes = probs_dict[symptom]
        symptom_com_directions = directions_dict[symptom]

        symptoms_interventions = get_interventions_dict(
            symptom_top_heads, symptom_com_directions, \
            num_heads)
        
        # save directions and heads
        symptom_dir = os.path.join(args.bank_dir, f"{symptom.lower()}/")

        interventions_path = os.path.join(symptom_dir, 'interventions.pkl')
        pickle_save(symptoms_interventions, interventions_path)

        interventions_dict[symptom] = symptoms_interventions


        

if __name__ == "__main__":
    main()
