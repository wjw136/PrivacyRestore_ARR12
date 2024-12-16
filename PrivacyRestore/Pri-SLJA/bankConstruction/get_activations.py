import os
import sys
sys.path.append('../')

import torch
from tqdm import tqdm
import numpy as np
import pickle
from common_utils import get_llama_activations_bau, load_json, \
    pickle_save, pickle_load, gen_question_legal, chunk_load, chunk_save
import llama
import pickle
import argparse
from collections import defaultdict
import json
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


TIME = datetime.now()
TIME = TIME.strftime("%Y-%m-%d-%H:%M:%S")

def tokenized_seed_question(dataList, tokenizer):
    all_questions = []
    all_questions_id = []
    all_labels = []
    all_symptoms = []
    
    for index, item in enumerate(dataList):
        new_data = item.copy()
        new_data['minor_premise_atoms']['object_atoms']['private_object_atoms'] = \
                  [ i for i in item['minor_premise_atoms']['object_atoms']['private_object_atoms'] + item['minor_premise_atoms']['object_atoms']['non_private_object_atoms'] if i == item['seed_privateAtom']]
        new_data['minor_premise_atoms']['object_atoms']['non_private_object_atoms'] = \
                [ i for i in item['minor_premise_atoms']['object_atoms']['private_object_atoms'] + item['minor_premise_atoms']['object_atoms']['non_private_object_atoms'] if i != item['seed_privateAtom']]
        
        new_data['minor_premise_atoms']['objective_atoms']['private_objective_atoms'] = \
                [ i for i in item['minor_premise_atoms']['objective_atoms']['private_objective_atoms'] + item['minor_premise_atoms']['objective_atoms']['non_private_objective_atoms'] if i == item['seed_privateAtom']]
        new_data['minor_premise_atoms']['objective_atoms']['non_private_objective_atoms'] = \
                [ i for i in item['minor_premise_atoms']['objective_atoms']['private_objective_atoms'] + item['minor_premise_atoms']['objective_atoms']['non_private_objective_atoms'] if i != item['seed_privateAtom']]
        
        new_data['minor_premise_atoms']['subject_atoms']['private_subject_atoms'] = \
                [ i for i in item['minor_premise_atoms']['subject_atoms']['private_subject_atoms'] + item['minor_premise_atoms']['subject_atoms']['non_private_subject_atoms'] if i == item['seed_privateAtom']]
        new_data['minor_premise_atoms']['subject_atoms']['non_private_subject_atoms'] = \
                [ i for i in item['minor_premise_atoms']['subject_atoms']['private_subject_atoms'] + item['minor_premise_atoms']['subject_atoms']['non_private_subject_atoms'] if i != item['seed_privateAtom']]
        
        new_data['minor_premise_atoms']['subjective_atoms']['private_subjective_atomss'] = \
                [ i for i in item['minor_premise_atoms']['subjective_atoms']['private_subjective_atoms'] + item['minor_premise_atoms']['subjective_atoms']['non_private_subjective_atoms'] if i == item['seed_privateAtom']]
        new_data['minor_premise_atoms']['subjective_atoms']['non_private_subjective_atoms'] = \
                [ i for i in item['minor_premise_atoms']['subjective_atoms']['private_subjective_atoms'] + item['minor_premise_atoms']['subjective_atoms']['non_private_subjective_atoms'] if i != item['seed_privateAtom']]
        
        question_init, question_mask = gen_question_legal(new_data['minor_premise_atoms'])
        if index == 0:
            print(f"question: {question_init}")
        question_id = tokenizer(question_init, return_tensors = 'pt').input_ids
        all_questions_id.append(question_id)
        all_labels.append(1)
        all_symptoms.append(item['seed_privateAtom'])
        all_questions.append(question_init)

        if index == 0:
            print(f"mask_question: {question_mask}")
        mask_question_id = tokenizer(question_mask, return_tensors = 'pt').input_ids
        all_questions_id.append(mask_question_id)
        all_labels.append(0)
        all_symptoms.append(item['seed_privateAtom'])
        all_questions.append(question_mask)

    return all_questions, all_questions_id, all_labels, all_symptoms


def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--output_activation_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--restore_chunk_index", type=int, default=0, help='local directory with model data')
    parser.add_argument("--chunk_size", type=int, default=5000, help='local directory with model data')

    parser.add_argument("--train_data_path", type=str, default="", \
                        help='local directory with model data')
    args = parser.parse_args()
    print(args)
    device= torch.device('cuda:{}'.format(args.device))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = llama.LlamaForCausalLM.from_pretrained(args.model_dir, device_map=device, use_inter=False) # 32全精度 16半精度 8int推理
    dataset = load_json(args.train_data_path)
    formatter = tokenized_seed_question

    print("Tokenizing prompts")
    raw_quesions, questions_id, labels, symptoms = formatter(dataset, tokenizer)

    chunk_layer_wise_activations = []
    chunk_head_wise_activations = []
    layer_index = args.restore_chunk_index
    head_index = args.restore_chunk_index

    
    # print(model)
    # (30): LlamaDecoderLayer(
    #     (self_attn): LlamaAttention(
    #       (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #       (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #       (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #       (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
    #       (rotary_emb): LlamaRotaryEmbedding()
    #       (att_out): Identity() =>新加的恒等映射，用来输出中间变量
    #       (value_out): Identity()
    #       (head_out): Identity() ====>1 
    #     )
    #     (mlp): LlamaMLP(
    #       (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
    #       (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
    #       (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
    #       (act_fn): SiLUActivation() ====>2
    #     )
    #     (input_layernorm): LlamaRMSNorm()
    #     (post_attention_layernorm): LlamaRMSNorm()
    #   )
    print("Getting activations")
    dir = args.output_activation_dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    for index, prompt in enumerate(questions_id[args.restore_chunk_index * args.chunk_size:]):
        layer_wise_activations, head_wise_activations, mlp_wise_activations = get_llama_activations_bau(model, prompt, device) # (33, seq, hidden_size), (32, seq, hidden_size)
        chunk_layer_wise_activations.append(layer_wise_activations[:,-1,:])
        chunk_head_wise_activations.append(head_wise_activations[:,-1,:])

        if index % 100 == 0:
            print(f"index, {index + args.restore_chunk_index * args.chunk_size}/{len(questions_id)}", flush=True)
        if len(chunk_layer_wise_activations) >= args.chunk_size:
            print("Begin chunk split!!!")
            print(f"Before: {len(chunk_layer_wise_activations)}")
            layer_index = chunk_save(layer_index, chunk_layer_wise_activations, args.chunk_size, args.output_activation_dir, \
                                    False)
            print(f"End: {len(chunk_layer_wise_activations)}")
    
        if len(chunk_head_wise_activations) >= args.chunk_size:
            head_index = chunk_save(head_index, chunk_head_wise_activations, args.chunk_size, args.output_activation_dir, \
                                    True)


        
    # aggregte data
    all_layer_wise_activations = []
    all_head_wise_activations = []
    for index in range(layer_index):
        all_layer_wise_activations.extend(chunk_load(index, args.output_activation_dir, False))
        all_head_wise_activations.extend(chunk_load(index, args.output_activation_dir, True))

    all_layer_wise_activations.extend(chunk_layer_wise_activations)
    all_head_wise_activations.extend(chunk_head_wise_activations)

    final_layer_path = os.path.join(dir, f"layer_wise.npy")
    final_head_path = os.path.join(dir, f"head_wise.npy")
    final_label_path = os.path.join(dir, f"label.npy")
    final_index_path = os.path.join(dir, f"index.json")
    print("Finally Saving layer wise activations")
    np.save(final_layer_path, all_layer_wise_activations) # (idx, (33,  hidden_size))
    print("Finally Saving head wise activations", flush=True)
    np.save(final_head_path, all_head_wise_activations) # (idx, (32, hidden_size)) 没有经过MLP层
    print("Finally Saving label", flush=True)
    np.save(final_label_path, np.array(labels)) # (idx, (32, hidden_size)) 没有经过MLP层

    indexBysymptom = defaultdict(list)
    for index, (symptom, layer_wise_activations, head_wise_activations) in \
                                                    enumerate(zip(symptoms, all_layer_wise_activations, all_head_wise_activations)):
        indexBysymptom[symptom].append(index)
    print("Finally Saving index", flush=True)
    with open(final_index_path, 'w') as final_index_file:
        json.dump(indexBysymptom, final_index_file)

if __name__ == '__main__':
    main()