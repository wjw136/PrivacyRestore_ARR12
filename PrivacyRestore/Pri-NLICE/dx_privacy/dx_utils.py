import numpy as np
import copy
from scipy.stats import gamma
import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import numpy as np
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
import json

class DX_privacy:
    def __init__(self, epsilon = 1.0) -> None:
        self.epsilon = epsilon
    
    # 添加差分隐私噪声到句子的嵌入表示
    def add_noise_to_embedding(self, interventions):
        # 使用拉普拉斯分布生成噪声
        # noisy_interventions = copy.deepcopy(interventions)
    
        vector = np.random.randn(interventions.shape[0])
        # print(vector)
        noise = gamma.rvs(interventions.shape[0], scale= 1.0/self.epsilon, size=1) * vector / np.linalg.norm(vector)
        noisy_interventions = interventions + torch.tensor(noise).to(interventions.device) # 加噪声
            
        return noisy_interventions
    
class DX_privacy_NLP:
    def __init__(self, model, tokenizer, epsilon = 1.0) -> None:

        self.tokenizer = tokenizer
        self.model = model
        
        self.embedding = self.model.get_input_embeddings()
        # 获取预训练BERT模型的嵌入矩阵
        self.embedding_matrix = self.model.get_input_embeddings().weight.detach().cpu().numpy()

        self.epsilon = epsilon

    # 定义一个函数来获取单词的嵌入表示
    def get_word_embedding(self, word):
        word_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
        word_embedding = self.embedding_matrix[word_id]
        return word_embedding
    
    def get_sim_embedding(self, embeddings):
        # 遍历整个嵌入空间并找到最接近的单词嵌入
        # （这里仅为演示目的，实际应用可能需要更复杂的搜索方法）
        # 初始化最小距离和最近的单词
        min_distance = float('inf')
        nearest_word = None
        nearest_embedding = None

        # 计算矩阵每行的范数
        norm_matrix1 = np.linalg.norm(embeddings, axis=1, keepdims=True)  # matrix1每行的范数
        norm_matrix2 = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)  # matrix2每行的范数

        # 计算点积
        dot_product = np.dot(embeddings, self.embedding_matrix.T)

        # 计算余弦相似度
        similarities = dot_product / (norm_matrix1 @ norm_matrix2.T)
        similarities[np.isinf(similarities)] = -1
        similarities[np.isnan(similarities)] = -1

        nearest_word_indexs = np.argmax(similarities, axis=1)
        nearest_embeddings = self.embedding_matrix[nearest_word_indexs]
        nearest_words = self.tokenizer.convert_ids_to_tokens(nearest_word_indexs)

        return nearest_embeddings, nearest_words
    
    # 添加差分隐私噪声到句子的嵌入表示
    def add_noise_to_embedding(self, embedding):
        # 使用拉普拉斯分布生成噪声
        noisy_embedding = copy.deepcopy(embedding)
        for subindex in range(len(embedding)):
            # noise = np.random.laplace(0, 1.0/self.epsilon, embedding.shape[1:])
            vector = np.random.randn(embedding.shape[1])
            # print(embedding.shape)
            noise = gamma.rvs(embedding.shape[1], scale= 1.0/self.epsilon, size=1) * vector / np.linalg.norm(vector)
            noisy_embedding[subindex] = embedding[subindex] + noise # 加噪声
        
        nearest_embedding, nearest_word = self.get_sim_embedding(noisy_embedding)
        # print(f"与目标嵌入最接近的单词是 '{nearest_word}'")
        # noisy_embedding[subindex] = nearest_embedding
            
        return nearest_embedding, nearest_word
    

    def detokenize(self, tokens):
        sentence = ' '.join(tokens)
        sentence = sentence.replace(' ##', '')  # 处理特殊字符的连接
        sentence = sentence.replace(' .', '.')  # 处理句号前面的空格
        sentence = sentence.replace(' ,', ',')  # 处理逗号前面的空格
        sentence = sentence.replace(" ' ", "'")  # 处理单引号周围的空格
        sentence = sentence.replace(" - ", "-")  # 处理连字符周围的空格
        sentence = sentence.replace(" ? ", "?")  # 处理问号周围的空格
        sentence = sentence.replace(" ! ", "!")  # 处理感叹号周围的空格
        return sentence

    def add_noise_to_tokens(self, sentence):
        # print(f"Add noise into {sentence}")
        # return sentence
        init_embedding = self.get_word_embedding(sentence)
        # 使用拉普拉斯分布生成噪声
        noisy_embedding = copy.deepcopy(init_embedding)
        for subindex in range(len(init_embedding)):
            vector = np.random.randn(init_embedding.shape[1])
            noise = gamma.rvs(init_embedding.shape[1], scale= 1.0/self.epsilon, size=1) * vector / np.linalg.norm(vector)
            noisy_embedding[subindex] = init_embedding[subindex] + noise # 加噪声
        
        nearest_embeddings, nearest_words = self.get_sim_embedding(noisy_embedding)

        return self.detokenize(nearest_words)