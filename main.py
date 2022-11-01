# coding=utf-8
# Copyright (c) 2022, Frederick Yuanchun Wang. wyc99@mail.nwpu.edu.cn 
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#             佛祖保佑   🙏   永无BUG 
#
#                    _ooOoo_
#                   o8888888o
#                   88" . "88
#                   (| -_- |)
#                   O\  =  /O
#                ____/`---'\____
#              .'  \\|     |//  `.
#             /  \\|||  :  |||//  \
#            /  _||||| -:- |||||-  \
#            |   | \\\  -  /// |   |
#            | \_|  ''\---/''  |   |
#            \  .-\__  `-`  ___/-. /
#          ___`. .'  /--.--\  `. . __
#       ."" '<  `.___\_<|>_/___.'  >'"".
#      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#      \  \ `-.   \_ __\ /__ _/   .-` /  /
# ======`-.____`-.___\_____/___.-`____.-'======
#                    `=---='
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# happy 1024 hh

from asyncio import AbstractEventLoop
from re import L
from symbol import testlist_comp
import torch
from torch import nn
from torch import optim
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.utils.data as Data
import numpy as np
from loguru import logger
import argparse 
import json
import xlrd
import os
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from model import WangYC_Model
from train import train

parser = argparse.ArgumentParser() # 初始化

parser.add_argument('--data_dir', default='/data/wangyuanchun/NLP_course/dataset/post_processed',
                       help='data_root')
parser.add_argument('--batch_size', type=int, default=1,
                       help='size of one batch') 
parser.add_argument('--epoch', type=int, default=100,
                       help='epoch')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='learning_rate')
parser.add_argument('--val_per_ite', type=int, default=100,
                       help='validation per how many iterations')
parser.add_argument('--model_save', default='/data/wangyuanchun/NLP_course/saved_models/',
                       help='validation per how many iterations')
parser.add_argument('--weight', type = int, default=5000,
                       help='validation per how many iterations')
parser.add_argument('--device', default='cuda:0',
                       help='the device you want to use to train')
parser.add_argument('--local_rank', type = int, default=-1)


if __name__ == '__main__':
    args, others_list = parser.parse_known_args() # 解析已知参数
    time = datetime.datetime.now()
    time_str = str(time.month) + '-' + str(time.day) + '-' + str(time.hour) + '-' + str(time.minute)
    writer = SummaryWriter('./runs/' + time_str)
    args.model_save = args.model_save + time_str
    os.system("mkdir " + args.model_save)
    train(args)
    # eval(args)
    # test()
    print('done :-)')