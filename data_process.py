import torch
import json
import os
from transformers import AutoTokenizer


def load_data(arg_mode):
    # """用来生成训练、测试数据"""
    # train_df = pd.read_csv("bert_example.csv", header=None)
    # sentences = train_df[0].values
    # targets = train_df[1].values
    # train_inputs, test_inputs, train_targets, test_targets = train_test_split(sentences, targets)
    if arg_mode == 'train':
        data_dir = '/data/wangyuanchun/NLP_course/dataset/post_processed/train_mse_2_line.json'
        data_dir = '/data/wangyuanchun/NLP_course/codes/train_example.json'
    elif arg_mode == 'val':
        data_dir = '/data/wangyuanchun/NLP_course/dataset/post_processed/val_mse_2_line.json'
        data_dir = '/data/wangyuanchun/NLP_course/codes/val_example.json'

    train_inputs = []
    train_targets = []
    with open(data_dir, 'r') as file_src:
        train_src = json.load(file_src)
    file_src.close()

    for each_train in train_src:
        train_inputs.append(each_train['text'])
        train_targets.append(torch.tensor(each_train['label'], dtype=torch.float))
    # with open(data_dir + 'dev.json', 'r') as file_src:
    #     val_src = json.load(file_src)
    # file_src.close()
    
    # train_inputs = train_src['text']
    # val_inputs = val_src['text']
    # train_targets = train_src['label']
    # val_targets = val_src['label']
    return train_inputs, train_targets