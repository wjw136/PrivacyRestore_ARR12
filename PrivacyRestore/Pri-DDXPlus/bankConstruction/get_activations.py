import os
import sys
sys.path.append('../')

import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import  tokenized_med_question
from common_utils import get_llama_activations_bau, load_json, \
    pickle_save, pickle_load
import llama
import pickle
import argparse
from collections import defaultdict
import json
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


TIME = datetime.now()
TIME = TIME.strftime("%Y-%m-%d-%H:%M:%S")


def chunk_save(index, dataList, chunk_size, output_activation_dir, HeadOrLayer=True, mode="question"):
    dir = output_activation_dir
    if HeadOrLayer:
        path = os.path.join(dir, f"tmp_{mode}_head_wise_chunk:{index}.pkl")
    else:
        path = os.path.join(dir, f"tmp_{mode}_layer_wise_chunk:{index}.pkl")
    pickle_save(dataList[0:chunk_size], path)
    del dataList[0:chunk_size]
    return index + 1

def chunk_load(index, output_activation_dir, HeadOrLayer=True, mode="question"):
    dir = output_activation_dir
    if HeadOrLayer:
        path = os.path.join(dir, f"tmp_{mode}_head_wise_chunk:{index}.pkl")
    else:
        path = os.path.join(dir, f"tmp_{mode}_layer_wise_chunk:{index}.pkl")
    
    return pickle_load(path)

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--output_activation_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--restore_chunk_index", type=int, default=0, help='local directory with model data')
    parser.add_argument("--chunk_size", type=int, default=5000, help='local directory with model data')
    parser.add_argument("--data_type", type=str, default="quetsion_end_q", help='local directory with model data')

    parser.add_argument("--train_data_path", type=str, default="", \
                        help='local directory with model data')
    args = parser.parse_args()
    print(args)

    MODEL = args.model_dir

    device= torch.device('cuda:{}'.format(args.device))
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = llama.LlamaForCausalLM.from_pretrained(args.model_dir, device_map=device,
                                                   bank_intervention_dir = ""
                                                    ) # 32全精度 16半精度 8int推理

    dataset = load_json(args.train_data_path)
    formatter = tokenized_med_question

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
    layer_path = os.path.join(dir, f"tmp_layer_wise_{TIME}.npy")
    head_path = os.path.join(dir, f"tmp_head_wise_{TIME}.npy")

    for index, prompt in enumerate(questions_id[args.restore_chunk_index * args.chunk_size:]):
        layer_wise_activations, head_wise_activations, mlp_wise_activations = get_llama_activations_bau(model, prompt, device) # (33, seq, hidden_size), (32, seq, hidden_size)
        # print(layer_wise_activations.shape)
        # print(raw_quesions[index])
        chunk_layer_wise_activations.append(layer_wise_activations[:,-1,:])
        chunk_head_wise_activations.append(head_wise_activations[:,-1,:])

        if index % 100 == 0:
            print(f"index, {index + args.restore_chunk_index * args.chunk_size}/{len(questions_id)}", flush=True)

        if len(chunk_layer_wise_activations) >= args.chunk_size:
            print("Begin chunk split!!!")
            print(f"Before: {len(chunk_layer_wise_activations)}")
            layer_index = chunk_save(layer_index, chunk_layer_wise_activations, args.chunk_size, args.output_activation_dir, \
                                    False, mode=args.data_type)
            print(f"End: {len(chunk_layer_wise_activations)}")
    
        if len(chunk_head_wise_activations) >= args.chunk_size:
            head_index = chunk_save(head_index, chunk_head_wise_activations, args.chunk_size, args.output_activation_dir, \
                                    True, mode=args.data_type)


        
    # aggregte data
    all_layer_wise_activations = []
    all_head_wise_activations = []
    for index in range(layer_index):
        all_layer_wise_activations.extend(chunk_load(index, args.output_activation_dir, False, mode=args.data_type))
        all_head_wise_activations.extend(chunk_load(index, args.output_activation_dir, True, mode=args.data_type))

    all_layer_wise_activations.extend(chunk_layer_wise_activations)
    all_head_wise_activations.extend(chunk_head_wise_activations)

    final_layer_path = os.path.join(dir, f"layer_wise_{args.dataset_name}.npy")
    final_head_path = os.path.join(dir, f"head_wise_{args.dataset_name}.npy")
    final_label_path = os.path.join(dir, f"label_{args.dataset_name}.npy")
    final_index_path = os.path.join(dir, f"index_{args.dataset_name}.json")
    print("Finally Saving layer wise activations")
    np.save(final_layer_path, all_layer_wise_activations) # (idx, (33,  hidden_size))
    
    print("Finally Saving head wise activations", flush=True)
    np.save(final_head_path, all_head_wise_activations) # (idx, (32, hidden_size)) 没有经过MLP层

    print("Finally Saving label", flush=True)
    np.save(final_label_path, np.array(labels)) # (idx, (32, hidden_size)) 没有经过MLP层

    indexBysymptom = defaultdict(list)
    for index, (question, label, symptom, layer_wise_activations, head_wise_activations) in \
                                                    enumerate(zip(raw_quesions, labels, symptoms, all_layer_wise_activations, all_head_wise_activations)):
        indexBysymptom[symptom].append(index)
        
    print("Finally Saving index", flush=True)
    with open(final_index_path, 'w') as final_index_file:
        json.dump(indexBysymptom, final_index_file)
    
    

if __name__ == '__main__':
    main()