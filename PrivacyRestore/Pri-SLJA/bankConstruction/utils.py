

import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import llama
from tqdm import tqdm
import numpy as np
import llama
import pandas as pd
import warnings
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import Trace, TraceDict
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
from functools import partial

import openai
import json
from collections import defaultdict
import random
from scipy.stats import entropy
from scipy.special import softmax
from collections import Counter
from common_utils import flattened_idx_to_layer_head

#-----------------------------------
#: get activations process
#-----------------------------------

def tokenized_seed_question(dataList, tokenizer):
    all_questions = []
    all_questions_id = []
    all_labels = []
    all_symptoms = []
    
    for index, item in enumerate(dataList):
        question = item['question_init']
        if index == 0:
            print(f"question: {question}")
        question_id = tokenizer(question, return_tensors = 'pt').input_ids
        all_questions_id.append(question_id)
        all_labels.append(1)
        all_symptoms.append(item['seed_evidence'])
        all_questions.append(question)

        if index == 0:
            print(f"mask_question: {item['question_mask']}")
        mask_question_id = tokenizer(item['question_mask'], return_tensors = 'pt').input_ids
        all_questions_id.append(mask_question_id)
        all_labels.append(0)
        all_symptoms.append(item['seed_evidence'])
        all_questions.append(item['question_mask'])

    return all_questions, all_questions_id, all_labels, all_symptoms



#-----------------------------------
#: bankConstrcution process
#-----------------------------------

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

