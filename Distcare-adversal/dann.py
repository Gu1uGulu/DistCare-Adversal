import numpy as np
import argparse
import os
import imp
import re
import pickle5 as pickle
import datetime
import random
import math
import logging
import copy
import matplotlib.pyplot as plt
import sklearn
import logging
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neighbors import kneighbors_graph
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter

from utils import utils
from utils.readers import InHospitalMortalityReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils
from torch.autograd import Function
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 定义常量

target_dataset = 'HM' 
RANDOM_SEED = 43
np.random.seed(RANDOM_SEED) #numpy
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED) # cpu
torch.cuda.manual_seed(RANDOM_SEED) #gpu
torch.backends.cudnn.deterministic=True # cudnn

# Use CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
# print("available device: {}".format(device))
reverse = False
model_name = 'dann'

if reverse:
    file_name = 'log_file' + '_' + model_name + '_' + target_dataset + '_' + 'reverse' + '.log'
else:
    file_name = 'log_file' + '_' + model_name + '_' + target_dataset + '.log'
def get_logger(name, file_name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 以下两行是为了在jupyter notebook 中不重复输出日志
    if logger.root.handlers:
        logger.root.handlers[0].setLevel(logging.WARNING)
 
    handler_stdout = logging.StreamHandler()
    handler_stdout.setLevel(logging.INFO)
    handler_stdout.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # logger.addHandler(handler_stdout)
 
    handler_file = logging.FileHandler(filename=file_name, mode='w', encoding='utf-8')
    handler_file.setLevel(logging.DEBUG)
    handler_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler_file)
 
    return logger

logger = get_logger(__name__,file_name)

logger.debug('这是希望输出的debug内容')
logger.info('这是希望输出的info内容')
logger.warning('这是希望输出的warning内容')


# 获取源域数据集
def get_n2n_data(x, y, x_len):
    length = len(x)
    assert length == len(y)
    assert length == len(x_len)
    new_x = []
    new_y = []
    new_x_len = []
    for i in range(length):
        for j in range(len(x[i])):
            new_x.append(x[i][:j+1])
            new_y.append(y[i][j])
            new_x_len.append(j+1)
    return new_x, new_y, new_x_len


source_data_path = './data/Challenge/'
small_part = False
arg_timestep = 1.0
batch_size = 256
epochs = 100
all_x_source = pickle.load(open(source_data_path + 'new_x_front_fill.dat', 'rb'))
all_y_source = pickle.load(open(source_data_path + 'new_y_front_fill.dat', 'rb'))
all_names_source = pickle.load(open(source_data_path + 'new_name.dat', 'rb'))
static_source = pickle.load(open(source_data_path + 'new_demo_front_fill.dat', 'rb'))
mask_x_source = pickle.load(open(source_data_path + 'new_mask_x.dat', 'rb'))
mask_demo_source = pickle.load(open(source_data_path + 'new_mask_demo.dat', 'rb'))
all_x_len_source = [len(i) for i in all_x_source]

if target_dataset == 'PD':
    subset_idx_source = [31, 29, 28, 33, 25, 18, 7, 21, 16, 15, 19, 17, 24, 3, 5, 0]
elif target_dataset == 'TJ':
    subset_idx_source = [27, 29, 18, 16, 26, 33, 28, 31, 32, 15, 11, 25, 21, 20, 9, 17, 30, 19]
elif target_dataset == 'HM':
    subset_idx_source = [0, 1, 2, 3, 5, 9, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

subset_cnt = len(subset_idx_source)
other_idx = []
for i in range(len(all_x_source[0][0])):
    if i not in subset_idx_source:
        other_idx.append(i)

for i in range(len(all_x_source)): #将共同特征移动到最开始，非共同特征移动到末尾
    cur = np.array(all_x_source[i], dtype=float)
    cur_mask = np.array(mask_x_source[i])
    cur_subset = cur[:, subset_idx_source]
    cur_other = cur[:, other_idx]
    cur_mask_subset = cur_mask[:, subset_idx_source]
    cur_mask_other = cur_mask[:, other_idx]
    all_x_source[i] = np.concatenate((cur_subset, cur_other), axis=1).tolist()
    mask_x_source[i] = np.concatenate((cur_mask_subset, cur_mask_other), axis=1).tolist()


train_num_source =int( len(all_x_source) * 0.8) + 1
logger.info(train_num_source)
dev_num_source =int( len(all_x_source) * 0.1) + 1
logger.info(dev_num_source)
test_num_source =int( len(all_x_source) * 0.1)
logger.info(test_num_source)
assert(train_num_source+dev_num_source+test_num_source == len(all_x_source))

train_x_source = []
train_y_source = []
train_names_source = []
train_static_source = []
train_x_len_source = []
train_mask_x_source = []
for idx in range(train_num_source):
    train_x_source.append(all_x_source[idx])
    train_y_source.append(int(all_y_source[idx][-1]))
    train_names_source.append(all_names_source[idx])
    train_static_source.append(static_source[idx])
    train_x_len_source.append(all_x_len_source[idx])
    train_mask_x_source.append(mask_x_source[idx])

dev_x_source = []
dev_y_source = []
dev_names_source = []
dev_static_source = []
dev_x_len_source = []
dev_mask_x_source = []
for idx in range(train_num_source, train_num_source + dev_num_source):
    dev_x_source.append(all_x_source[idx])
    dev_y_source.append(int(all_y_source[idx][-1]))
    dev_names_source.append(all_names_source[idx])
    dev_static_source.append(static_source[idx])
    dev_x_len_source.append(all_x_len_source[idx])
    dev_mask_x_source.append(mask_x_source[idx])


test_x = []
test_y = []
test_names = []
test_static = []
test_x_len = []
test_mask_x = []
for idx in range(train_num_source + dev_num_source, train_num_source + dev_num_source + test_num_source):
    test_x.append(all_x_source[idx])
    test_y.append(int(all_y_source[idx][-1]))
    test_names.append(all_names_source[idx])
    test_static.append(static_source[idx])
    test_x_len.append(all_x_len_source[idx])
    test_mask_x.append(mask_x_source[idx])


assert(len(train_x_source) == train_num_source)
assert(len(dev_x_source) == dev_num_source)
assert(len(test_x) == test_num_source)

long_x_source = all_x_source
long_y_source = [y[-1] for y in all_y_source]



def get_loss(y_pred, y_true):
    loss = torch.nn.BCELoss()
    return loss(y_pred, y_true)

def get_re_loss(y_pred, y_true):
    loss = torch.nn.MSELoss()
    return loss(y_pred, y_true)

def get_kl_loss(x_pred, x_target):
    loss = torch.nn.KLDivLoss(reduce=True, size_average=True)
    return loss(x_pred, x_target)

def get_wass_dist(x_pred, x_target):
    m1 = torch.mean(x_pred, dim=0)
    m2 = torch.mean(x_target, dim=0)
    v1 = torch.var(x_pred, dim=0)
    v2 = torch.var(x_target, dim=0)
    p1 = torch.sum(torch.pow((m1 - m2), 2))
    p2 = torch.sum(torch.pow(torch.pow(v1, 1/2) - torch.pow(v2, 1/2), 2))
    return torch.pow(p1+p2, 1/2)

def pad_sents(sents, pad_token):
#     print(f'len(pad_token) is {len(pad_token)}')
#     print(f'sents is {sents}')

    sents_padded = []

    max_length = max([len(_) for _ in sents])
    for i in sents:
        padded = list(i) + [pad_token]*(max_length-len(i))
#         print(f'padded is {padded}')
        sents_padded.append(np.array(padded))
        
    return np.array(sents_padded)

def batch_iter(x, y, lens, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    # batch_num = math.ceil(len(x) / batch_size) # 向下取整
    batch_num = len(x) // batch_size if len(x) % batch_size == 0 else len(x) // batch_size + 1
    # print(f"len(x) is {len(x)}, len(y) is {len(y)}, len(lens) is {len(lens)}, batch_size is {batch_size}, batch_num is {batch_num}")
    index_array = list(range(len(x)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        if (i + 1) * batch_size  < len(x):
            indices = index_array[i * batch_size: (i + 1) * batch_size] #  fetch out all the induces
        else:
            indices = index_array[i * batch_size: ]
        examples = []
        for idx in indices:
            examples.append((x[idx], y[idx],lens[idx]))
       
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
    
        batch_x = [e[0] for e in examples]
        batch_y = [e[1] for e in examples]
        batch_lens = [e[2] for e in examples]

        yield batch_x, batch_y, batch_lens

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class dann(nn.Module):
    def __init__(self, common_dim, hidden_dim, d_model,  MHD_num_head, d_ff, output_dim, keep_prob=0.5):
        super(dann, self).__init__()

        # hyperparameters
        self.input_dim = common_dim
        self.hidden_dim = hidden_dim  # d_model
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob

        # layers
        self.GRUs = clones(nn.GRU(1, self.hidden_dim, batch_first = True), self.input_dim)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=0.1)

        self.demo_proj_main = nn.Linear(12, self.hidden_dim)
        self.demo_proj = nn.Linear(12, self.hidden_dim)
        self.Linear = nn.Linear(self.hidden_dim, 1)
        self.output = nn.Linear(self.input_dim + self.input_diff_dim, self.output_dim)

        # adversal方法中的域分类器  
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_dim, self.hidden_dim))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(self.hidden_dim))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(hidden_dim, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        self.dropout = nn.Dropout(p = 1 - self.keep_prob)
        self.FC_embed = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tanh=nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.to_MMD = nn.Linear(self.hidden_dim, 1)

    def forward(self, input, input_diff, lens, alpha, is_teacher):
        lens = lens.to('cpu')
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        feature_dim_diff = input_diff.size(2)
        assert(feature_dim == self.input_dim)# input Tensor : 256 * 48 * 76
        assert(self.d_model % self.MHD_num_head == 0)
        GRU_embeded_input = self.GRUs[0](pack_padded_sequence(input[:,:,0].unsqueeze(-1), lens, batch_first=True))[1].squeeze().unsqueeze(1) # b 1 h
        for i in range(feature_dim-1):
            embeded_input = self.GRUs[i+1](pack_padded_sequence(input[:,:,i+1].unsqueeze(-1), lens, batch_first=True))[1].squeeze().unsqueeze(1) # b 1 h
            GRU_embeded_input = torch.cat((GRU_embeded_input, embeded_input), 1)

        if is_teacher: # 来自源数据集
            common_input = GRU_embeded_input[:, 0, :]
            for i in range(1, feature_dim):
                common_input = common_input + GRU_embeded_input[:, i, :]  
            # print(f"common_input1.shape is {common_input.shape}")
            common_input = torch.squeeze(common_input, 1) # batch * hidden
            reverse_input = ReverseLayerF.apply(common_input, alpha)
            # print(f"common_input2.shape is {common_input.shape}")
            domain_output = self.domain_classifier(reverse_input)

            posi_input = self.dropout(torch.cat((GRU_embeded_input, General_GRU_embeded_input), 1)) # batch_size * d_input + d_input_diff * hidden_dim
            
            contexts = self.Linear(posi_input).squeeze()# b i
            output = self.output(self.dropout(contexts))# b 1
            output = self.sigmoid(output)
            return output, domain_output, contexts
        else: # 来自目标数据集，主要是为了混淆domain classifier
            common_input = GRU_embeded_input[:, 0, :]
            for i in range(1, feature_dim):
                common_input = common_input + GRU_embeded_input[:, i, :]  
            common_input = torch.squeeze(common_input, 1) # batch * hidden
            reverse_input = ReverseLayerF.apply(common_input, alpha)
            domain_output = self.domain_classifier(reverse_input)
            return domain_output

        
def getSplitData(x, lens, y):
    train_num =int( len(x) * 0.8) + 1
    dev_num =int( len(x) * 0.1) + 1
    test_num = len(x) - train_num - dev_num
    train_x = []
    train_y = []
    train_len = []
    for idx in range(train_num):
        train_x.append(x[idx])
        train_y.append(int(y[idx][-1]))
        train_len.append(lens[idx])

    dev_x = []
    dev_y = []
    dev_len = []
    for idx in range(train_num, train_num + dev_num):
        dev_x.append(x[idx])
        dev_y.append(int(y[idx][-1]))
        dev_len.append(lens[idx])

    test_x = []
    test_y = []
    test_len = []

    for idx in range(train_num + dev_num, train_num + dev_num + test_num):
        test_x.append(x[idx])
        test_y.append(int(y[idx][-1]))
        test_len.append(lens[idx])
    return train_x, train_y, train_len, dev_x, dev_y, dev_len, test_x, test_y, test_len

logger.info("load target data")
if target_dataset == 'PD':
    data_path = './data/PD/'
    all_x_target = pickle.load(open(data_path + 'x.pkl', 'rb'))
    all_time_target = pickle.load(open(data_path + 'y_z.pkl', 'rb'))
    all_x_len_target = [len(i) for i in all_x_target]

    subset_idx_target = [0, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 19, 20, 56, 57, 58]
    other_idx_target = list(range(69))
    for i in subset_idx_target:
        other_idx_target.remove(i)
    for i in range(len(all_x_target)):
        cur = np.array(all_x_target[i], dtype=float)
        cur_subset = cur[:, subset_idx_target]
        cur_other = cur[:, other_idx_target]
        all_x_target[i] = np.concatenate((cur_subset, cur_other), axis=1).tolist()
elif target_dataset == 'TJ':
    data_path = './data/Tongji/'
    all_x_target = pickle.load(open(data_path + 'x.pkl', 'rb'))
    all_y_target = pickle.load(open(data_path + 'y.pkl', 'rb'))
    all_time_target = pickle.load(open(data_path + 'y.pkl', 'rb'))
    all_x_len_target = [len(i) for i in all_x_target]

    for i in range(len(all_time_target)):
        for j in range(len(all_time_target[i])):
            all_time_target[i][j] = all_time_target[i][j][-1]
            all_y_target[i][j] = all_y_target[i][j][0]

    subset_idx_target = [2, 3, 4, 9, 13, 14, 26, 27, 30, 32, 34, 38, 39, 41, 52, 53, 66, 74]
    other_idx_target = list(range(75))
    for i in subset_idx_target:
        other_idx_target.remove(i)
    for i in range(len(all_x_target)):
        cur = np.array(all_x_target[i], dtype=float)
        cur_subset = cur[:, subset_idx_target]
        cur_other = cur[:, other_idx_target]
        all_x_target[i] = np.concatenate((cur_subset, cur_other), axis=1).tolist()
elif target_dataset == 'HM':
    data_path = './data/CDSL/'
    all_x_target = pickle.load(open(data_path + 'x.pkl', 'rb'))
    all_y_target = pickle.load(open(data_path + 'y.pkl', 'rb'))
    all_time_target = pickle.load(open(data_path + 'y.pkl', 'rb'))
    all_x_len_target = [len(i) for i in all_x_target]

    for i in range(len(all_time_target)):
        for j in range(len(all_time_target[i])):
            all_time_target[i][j] = all_time_target[i][j][-1]
            all_y_target[i][j] = all_y_target[i][j][0]

    subset_idx_target = [5, 6, 4, 2, 3, 48, 79, 76, 87, 25, 30, 31, 18, 43, 58, 66, 40, 57, 23, 92, 50, 54, 91, 60, 39, 81]
    other_idx_target= list(range(99))
    for i in subset_idx_target:
        other_idx_target.remove(i)
    for i in range(len(all_x_target)):
        cur = np.array(all_x_target[i], dtype=float)
        cur_subset = cur[:, subset_idx_target]
        cur_other = cur[:, other_idx_target]
    #     tar_all_x[i] = np.concatenate((cur_subset, cur_other), axis=1).tolist()
        all_x_target[i] = np.concatenate((cur_subset, cur_other), axis=1).tolist()
    
if target_dataset == 'PD':
    all_x_target = all_x_target
    all_y_target = all_time_target
elif  target_dataset == 'HM' or target_dataset == 'TJ':
    examples = []
    for idx in range(len(all_x_target)):
        examples.append((all_x_target[idx], all_y_target[idx], all_time_target[idx], all_x_len_target[idx]))
    examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
    all_x_target = [e[0] for e in examples]
    all_y_target = [e[1] for e in examples]
    all_time_target = [e[2] for e in examples]
    all_x_len_target = [e[3] for e in examples]

num_source = len(all_x_source)
num_target = len(all_x_target)
# print(target_dataset,len(all_x_target), len(all_x_target[0]),len(all_x_target[0][0]))
all_x_target_confuse = []
all_x_len_target_confuse = []
all_y_target_confuse = []
all_x_source_confuse = []
all_x_len_source_confuse = []
all_y_source_confuse = []
repeat_times = 0

if num_source < num_target:
    all_x_target_confuse = all_x_target
    all_y_target_confuse = all_y_target
    all_x_len_target_confuse = all_x_len_target
    while repeat_times * num_source < num_target:
        all_x_source_confuse = all_x_source_confuse + all_x_source
        all_x_len_source_confuse = all_x_len_source_confuse + all_x_len_source
        all_y_source_confuse =  all_y_source_confuse + all_y_source
        repeat_times = repeat_times + 1
    all_x_source_confuse = all_x_source_confuse[:num_target]
    all_x_len_source_confuse = all_x_len_source_confuse[:num_target]
    all_y_source_confuse = all_y_source_confuse[:num_target]
elif num_target < num_source:
    all_x_source_confuse = all_x_source
    all_x_len_source_confuse = all_x_len_source
    all_y_source_confuse = all_y_source
    while repeat_times * num_target < num_source:
        all_x_target_confuse = all_x_target_confuse + all_x_target
        all_x_len_target_confuse = all_x_len_target_confuse + all_x_len_target
        all_y_target_confuse = all_y_target_confuse + all_y_target
        repeat_times = repeat_times + 1
    all_x_target_confuse = all_x_target_confuse[:num_source]
    all_x_len_target_confuse = all_x_len_target_confuse[:num_source]
    all_y_target_confuse = all_y_target_confuse[:num_source]

# print(f"len(all_x_source_confuse) is {len(all_x_source_confuse)}, len(all_x_target_confuse) is {len(all_x_target_confuse)}")

#todo 划分train、dev、test 
# all_x_source_confuse = pad_sents(all_x_source_confuse, pad_token_source)
# all_x_target_confuse = pad_sents(all_x_target_confuse, pad_token_target)
train_x_source_confuse, train_y_source_confuse, train_len_source_confuse, dev_x_source_confuse, dev_y_source_confuse, dev_len_source_confuse, test_x_source_confuse,\
test_y_source_confuse, test_len_source_confuse = getSplitData(all_x_source_confuse, all_x_len_source_confuse, all_y_source_confuse)

train_x_target_confuse, train_y_target_confuse, train_len_target_confuse, dev_x_target_confuse, dev_y_target_confuse, dev_len_target_confuse, test_x_target_confuse,\
test_y_target_confuse, test_len_target_confuse = getSplitData(all_x_target_confuse, all_x_len_target_confuse, all_y_target_confuse)

# long_x_source = all_x_source
# long_y_source = [y[-1] for y in all_y_source]

epochs = 50
batch_size = 256
common_dim = subset_cnt 

diff_dim = input_dim - subset_cnt
hidden_dim = 64
d_model = 64
MHD_num_head = 4
d_ff = 64
output_dim = 1
model_student = distcare_student(input_dim = common_dim, input_diff_dim = diff_dim, hidden_dim = hidden_dim, d_model=d_model, MHD_num_head=MHD_num_head, d_ff=d_ff, output_dim = output_dim).to(device)
optimizer_student = torch.optim.Adam(model_student.parameters(), lr=1e-3)

class MultitaskLoss(nn.Module):
    def __init__(self, task_num=2):
        super(MultitaskLoss, self).__init__()
        self.task_num = task_num
        self.alpha = nn.Parameter(torch.ones((task_num)), requires_grad=True)
        self.bce = nn.BCELoss()
        self.kl = nn.KLDivLoss(reduce=True, size_average=True)

    def forward(self, opt_student, batch_y, emb_student, emb_teacher, tar_source, tar_tar):
        BCE_Loss = self.bce(opt_student, batch_y)
        emb_Loss = self.kl(emb_student, emb_teacher)
        return BCE_Loss * self.alpha[0] + emb_Loss * self.alpha[1]

def get_multitask_loss(opt_student, batch_y, emb_student, emb_teacher):
    mtl = MultitaskLoss(task_num=3)
    return mtl(opt_student, batch_y, emb_student, emb_teacher)




# # Training Student
# # If you don't want to train Student Model:
# # - The pretrained student model is in direcrtory './model/', and can be directly loaded, 
# # - Simply skip this cell and load the model to validate on Dev Dataset.

# logger.info('Training Student')
# teacher_flag = True
# epochs = 30
# total_train_loss = []
# total_valid_loss = []
# global_best = 0
# auroc = []
# auprc = []
# minpse = []
# history = []
# # begin_time = time.time()
# best_auroc = 0
# best_auprc = 0
# best_minpse = 0
# best_total_loss = 0x3f3f3f3f
# loss_domain = torch.nn.NLLLoss()
# loss_predict = torch.nn.MSELoss()
# loss_embed = nn.KLDivLoss(reduce=True, size_average=True)




# print(f'len(train_source_iter) is {len(train_x_source_confuse)}, len(train_target_iter) is {len(train_x_target_confuse)}, steps is {len(train_x_source_confuse) // batch_size + 1}')

# if target_dataset == 'PD':
#     data_str = 'pd'
# elif target_dataset == 'TJ':
#     data_str = 'covid'
# elif target_dataset == 'HM':
#     data_str = 'spain'


# if teacher_flag:
#     file_name = './model/pretrained-challenge-front-fill-2'+ data_str
# else: 
#     file_name = './model/pretrained-challenge-front-fill-2'+ data_str + '-noteacher'

# for each_epoch in range(epochs):
#     train_source_iter = batch_iter(train_x_source_confuse, train_y_source_confuse, train_len_source_confuse, batch_size=batch_size)
#     dev_source_iter = batch_iter(dev_x_source_confuse, dev_y_source_confuse, dev_len_source_confuse, batch_size=batch_size)
#     test_source_iter = batch_iter(test_x_source_confuse, test_y_source_confuse, test_len_source_confuse, batch_size=batch_size)
#     train_target_iter = batch_iter(train_x_target_confuse, train_y_target_confuse, train_len_target_confuse, batch_size=batch_size)
#     dev_target_iter = batch_iter(dev_x_target_confuse, dev_y_target_confuse, dev_len_target_confuse, batch_size=batch_size)
#     test_target_iter = batch_iter(test_x_target_confuse, test_y_target_confuse, test_len_target_confuse, batch_size=batch_size)
#     epoch_loss = []
#     counter_batch = 0
#     model_student.train()  
#     model.eval()
#     steps = len(train_x_source_confuse) // batch_size + 1 if len(train_x_source_confuse) % batch_size != 0 else len(train_x_source_confuse) // batch_size
#     for step in range(steps):
#         # -----source_domain--------
#         batch_x, batch_y, batch_lens= next(train_source_iter)
#         p = float(step + each_epoch * steps) / epochs / steps
#         alpha = 2. / (1. + np.exp(-10 * p)) - 1
#         optimizer_student.zero_grad()
#         batch_x = torch.tensor(pad_sents(batch_x, pad_token_source), dtype=torch.float32).to(device)
#         batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
#         batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
#         # batch_mask_x = torch.tensor(pad_sents(batch_mask_x, pad_token), dtype=torch.float32).to(device)
#         # opt_student, decov_loss_student, emb_student, tar_result = model_student(batch_x[:,:,:subset_cnt], batch_x[:,:,subset_cnt:], batch_lens, [tar_all_x, tar_all_x_len], True)
#         domain_label = torch.zeros(min(batch_size, batch_x.shape[0])).long().to(device)
#         opt_student, opt_domain, emb_student = model_student(batch_x[:,:,:subset_cnt], batch_x[:,:,subset_cnt:], batch_lens, alpha, True)
#         emb_teacher = torch.tensor(train_teacher_emb[step], dtype=torch.float32).to(device)
#         emb_student = F.log_softmax(emb_student, dim=1)
#         emb_teacher = F.softmax(emb_teacher.detach(), dim=1)
#         err_emb = loss_embed(emb_student, emb_teacher)
#         err_predict = loss_predict(opt_student, batch_y)
#         err_domain1 = loss_domain(opt_domain, domain_label)
#             # loss = get_multitask_loss(opt_student, batch_y.unsqueeze(-1), emb_student, emb_teacher)

#         # -----target_domain--------
#         batch_x, batch_y, batch_lens = next(train_target_iter)
#         p = float(step + each_epoch * len(train_x_source)) / epochs / len(train_x_len_source)
#         alpha = 2. / (1. + np.exp(-10 * p)) - 1
#         optimizer_student.zero_grad()
#         batch_x = torch.tensor(pad_sents(batch_x, pad_token_target), dtype=torch.float32).to(device)
#         batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
#         batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
#         # batch_mask_x = torch.tensor(pad_sents(batch_mask_x, pad_token), dtype=torch.float32).to(device)
#         # opt_student, decov_loss_student, emb_student, tar_result = model_student(batch_x[:,:,:subset_cnt], batch_x[:,:,subset_cnt:], batch_lens, [tar_all_x, tar_all_x_len], True)
#         domain_label = torch.ones(min(batch_size, batch_x.shape[0])).long().to(device)
#         opt_domain = model_student(batch_x[:,:,:subset_cnt], batch_x[:,:,subset_cnt:], batch_lens, alpha, False)
#         err_domain2 = loss_domain(opt_domain, domain_label)

#         # -----common--------
#         loss = err_emb + err_predict + err_domain1 + err_domain2
#         epoch_loss.append(loss.cpu().detach().numpy())
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model_student.parameters(), 20)
#         optimizer_student.step()

#         if step % 20 == 0:
#             print('Epoch %d Batch %d: Train Loss = %.4f'%(each_epoch, step, loss.cpu().detach().numpy()))
#             logger.info('Epoch %d Batch %d: Train Loss = %.4f'%(each_epoch, step, loss.cpu().detach().numpy()))

#     epoch_loss = np.mean(epoch_loss)
#     total_train_loss.append(epoch_loss)


#     # dev_source_dataset = MyDataset(dev_x_source_confuse, dev_len_source_confuse, dev_y_source_confuse)
#     # dev_target_dataset = MyDataset(dev_x_target_confuse, dev_len_target_confuse, dev_y_target_confuse)
#     # dev_source_dataloader = DataLoader(dev_source_dataset, batch_size= batch_size)
#     # dev_target_dataloader = DataLoader(dev_target_dataset, batch_size=batch_size)
#     #Validation

#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         steps = len(dev_x_source_confuse) // batch_size + 1 if len(dev_x_source_confuse) % batch_size != 0 else len(dev_x_source_confuse) // batch_size
#         for step in range(steps):
#             # -----source_domain--------
#             batch_x, batch_y, batch_lens= next(dev_source_iter)
#             p = float(step + each_epoch * steps) / epochs / steps
#             alpha = 2. / (1. + np.exp(-10 * p)) - 1
#             optimizer_student.zero_grad()
#             batch_x = torch.tensor(pad_sents(batch_x, pad_token_source), dtype=torch.float32).to(device)
#             batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
#             batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
#             # batch_mask_x = torch.tensor(pad_sents(batch_mask_x, pad_token), dtype=torch.float32).to(device)
#             # opt_student, decov_loss_student, emb_student, tar_result = model_student(batch_x[:,:,:subset_cnt], batch_x[:,:,subset_cnt:], batch_lens, [tar_all_x, tar_all_x_len], True)
#             domain_label = torch.zeros(min(batch_size, batch_x.shape[0])).long().to(device)
#             opt_student, opt_domain, emb_student = model_student(batch_x[:,:,:subset_cnt], batch_x[:,:,subset_cnt:], batch_lens, alpha, True)
#             # emb_teacher = torch.tensor(dev_teacher_emb[step], dtype=torch.float32).to(device)
#             emb_student = F.log_softmax(emb_student, dim=1)
#             emb_teacher = F.softmax(emb_teacher.detach(), dim=1)
#             # err_emb = loss_embed(emb_student, emb_teacher) #todo 是否考虑它
#             err_predict = loss_predict(opt_student, batch_y)
#             err_domain1 = loss_domain(opt_domain, domain_label)
#                 # loss = get_multitask_loss(opt_student, batch_y.unsqueeze(-1), emb_student, emb_teacher)

#             # -----target_domain--------
#             batch_x, batch_y, batch_lens = next(dev_target_iter)
#             optimizer_student.zero_grad()
#             batch_x = torch.tensor(pad_sents(batch_x, pad_token_target), dtype=torch.float32).to(device)
#             batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
#             batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
#             # batch_mask_x = torch.tensor(pad_sents(batch_mask_x, pad_token), dtype=torch.float32).to(device)
#             # opt_student, decov_loss_student, emb_student, tar_result = model_student(batch_x[:,:,:subset_cnt], batch_x[:,:,subset_cnt:], batch_lens, [tar_all_x, tar_all_x_len], True)
#             domain_label = torch.ones(min(batch_size, batch_x.shape[0])).long().to(device)
#             opt_domain = model_student(batch_x[:,:,:subset_cnt], batch_x[:,:,subset_cnt:], batch_lens, alpha, False)
#             err_domain2 = loss_domain(opt_domain, domain_label)

#             # -----common--------
#             loss = err_predict + err_domain1 + err_domain2
#             if loss < best_total_loss:
#                 best_total_loss = loss
#                 state = {
#                     'net': model_student.state_dict(),
#                     'optimizer': optimizer_student.state_dict(),
#                     'epoch': each_epoch
#                 }
#                 torch.save(state, file_name)
#                 print('------------ Save best model - TOTAL_LOSS: %.4f ------------'%best_total_loss)
#                 logger.info('------------ Save best model - TOTAL_LOSS: %.4f ------------'%best_total_loss)


teacher_flag = True

if target_dataset == 'PD':    
    data_str = 'pd'
elif target_dataset == 'TJ':    
    data_str = 'covid'
elif target_dataset == 'HM':
    data_str = 'spain'

if teacher_flag:
    file_name = './model/pretrained-challenge-front-fill-2'+ data_str
else: 
    file_name = './model/pretrained-challenge-front-fill-2'+ data_str + '-noteacher'

checkpoint = torch.load(file_name, \
                        map_location=torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu') )
save_epoch = checkpoint['epoch']
print("last saved model is in epoch {}".format(save_epoch))
logger.info("last saved model is in epoch {}".format(save_epoch))
model_student.load_state_dict(checkpoint['net'])
optimizer_student.load_state_dict(checkpoint['optimizer'])
model_student.eval()
