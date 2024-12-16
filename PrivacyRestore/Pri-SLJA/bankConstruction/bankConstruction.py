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
from common_utils import get_interventions_dict, \
    pickle_save, numpy_save, pickle_load
import json
from collections import defaultdict, Counter
import shutil
import re
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from common_utils import flattened_idx_to_layer_head



def train_probes(seed, train_set_idxs, val_set_idxs, symptom_head_activations, symptom_labels, num_layers, num_heads):
    all_head_accs = []
    probes = []
    all_X_train = np.concatenate([[symptom_head_activations[i]] for i in train_set_idxs], axis = 0)
    all_X_val = np.concatenate([[symptom_head_activations[i]] for i in val_set_idxs], axis = 0)
    y_train = [symptom_labels[i] for i in train_set_idxs]
    print(f"y_train counter: {Counter(y_train)}")
    y_val = [symptom_labels[i] for i in val_set_idxs]
    print(f"y_val counter: {Counter(y_val)}")
    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]
            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train) #! probe是逻辑回归，预测出属于正类别的概率，权重可以作为负类指向正类的向量
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)
    all_head_accs_np = np.array(all_head_accs)
    return probes, all_head_accs_np

def get_com_directions(num_layers, num_heads, symptom, symptom_head_activations, symptom_labels): 
    com_directions = []
    for layer in range(num_layers): 
        for head in range(num_heads): 
            true_mass_activations = np.concatenate([[activations[layer,head,:]] \
                                                     for activations, label in zip(symptom_head_activations, symptom_labels) if label == 1], axis=0)
            false_mass_activations = np.concatenate([[activations[layer,head,:]] \
                                                     for activations, label in zip(symptom_head_activations, symptom_labels) if label == 0], axis=0)
            true_mass_mean = np.mean(true_mass_activations, axis=0)
            false_mass_mean = np.mean(false_mass_activations, axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    return com_directions

def get_top_heads(symptom, symptom_head_activations, symptom_labels, num_layers, num_heads, seed, num_to_intervene, val_ratio):
    train_set_idxs = np.random.choice(list(range(len(symptom_labels))), \
                                      size=int(len(symptom_labels)*(1-val_ratio)), replace=False)
    val_set_idxs = np.array([x for x in list(range(len(symptom_labels))) if x not in train_set_idxs])
    probes, all_head_accs_np = train_probes(seed, train_set_idxs, val_set_idxs, \
                                            symptom_head_activations, symptom_labels, num_layers=num_layers, num_heads=num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)
    print(f"{symptom} top heads, top heads accuracy: {np.sort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene]}")
    top_heads = []
    top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    return top_heads, probes

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--activations_dir', type=str, \
                        default='', \
                            help='feature bank for calculating std along direction')
    parser.add_argument('--bank_dir', type=str, \
                        default='', \
                            help='feature bank for calculating std along direction')
    
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--mask_evidences_list', type=str, default="")

    args = parser.parse_args()
    print(args)
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load mask_evidences
    mask_evidences = pickle_load(args.mask_evidences_list)
    evidences2id = {evidence : index for index, evidence in enumerate(mask_evidences) }

    # create model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    device= torch.device('cuda:{}'.format(args.device))
    model = LlamaForCausalLM.from_pretrained(args.model_dir, low_cpu_mem_usage = True, torch_dtype=torch.float32, device_map=device)
    
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations dataset
    probs_dict = defaultdict(list)
    directions_dict = defaultdict(list)
    heads_dict = defaultdict(list)

    index_path = os.path.join(args.activations_dir, f"index.json")
    label_path = os.path.join(args.activations_dir, f"label.npy")
    head_activations_path = os.path.join(args.activations_dir, f"head_wise.npy")
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
                                        num_layers, num_heads, args.seed, num_heads * num_layers, args.val_ratio)
        print(f"symptom {symptom}, heads intervened example: {sorted(symptom_top_heads)}")
        # save directions and heads
        symptom_dir = os.path.join(args.bank_dir, f"{evidences2id[symptom]}/")
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
        symptom_dir = os.path.join(args.bank_dir, f"{evidences2id[symptom]}/")
        interventions_path = os.path.join(symptom_dir, 'interventions.pkl')
        pickle_save(symptoms_interventions, interventions_path)
        interventions_dict[symptom] = symptoms_interventions


if __name__ == "__main__":
    main()
