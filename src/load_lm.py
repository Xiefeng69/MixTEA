import os
import json
import numpy as np
import pickle
from tqdm import *

def load_lm(task, node_size):
    print("[loading Language Model...]\n")
    
    save_dict_path = f"data/{task}.pkl"
    if os.path.exists(save_dict_path):
        with open(save_dict_path, "rb") as f:
            ent_vec = pickle.load(f)
    else:
        # loading glove language model
        word_vecs = {}
        with open("data/glove.6B.300d.txt", encoding='UTF-8') as f:
            for line in tqdm(f.readlines()):
                line = line.split()
                try:
                    word_vecs[line[0]] = np.array([float(x) for x in line[1:]])
                except:
                    continue
        # mapping name to embedding
        ent_names = json.load(open('data/' + task + "/name.json","r"))
        d = {}
        count = 0
        for _, name in ent_names:
            for word in name:
                word = word.lower()
                for idx in range(len(word)-1):
                    if word[idx:idx+2] not in d:
                        d[word[idx:idx+2]] = count
                        count += 1
        
        ent_vec = np.zeros((node_size,300))
        for i, name in ent_names:
            k = 0
            for word in name:
                word = word.lower()
                if word in word_vecs:
                    ent_vec[i] += word_vecs[word]
                    k += 1
            if k:
                ent_vec[i]/=k
            else:
                ent_vec[i] = np.random.random(300)-0.5
            ent_vec[i] = ent_vec[i]/ np.linalg.norm(ent_vec[i])
        
        with open(save_dict_path, "wb") as f:
            pickle.dump(ent_vec, f)
    
    return np.array(ent_vec, dtype=np.float32)