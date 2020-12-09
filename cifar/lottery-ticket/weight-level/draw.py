import pandas as pd
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
sns.set()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

parser.add_argument('--ori', default='', type=str, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--inverse', default='0', type=str, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--name', default="pruned.png", type=str)
args = parser.parse_args()
#
# #import pdb; pdb.set_trace()
# ori_data = pd.read_table(args.ori, sep='\t')
# inverse_data = pd.read_table(args.inverse, sep='\t')
# print(ori_data)
#
# sns.lineplot(x=range(ori_data['Valid Acc.'].shape[0]), y=ori_data['Valid Acc.'], color='red',dashes=True)
# sns.lineplot(x=range(inverse_data['Valid Acc.'].shape[0]), y=inverse_data['Valid Acc.'], color='green',dashes=True)
# plt.xlabel("episode")
# plt.ylabel("accuracy")
# plt.legend()
# plt.savefig(args.name)
low_mask = open(args.ori,'rb')
low_mask = pickle.load(low_mask)
high_mask = open(args.inverse, 'rb')
high_mask = pickle.load(high_mask)
mask_out = {"sim":[], "diff":[]}
for key in high_mask.keys():
    #import pdb; pdb.set_trace()
    # #print(key)
    # high_mask[key]
    total_num = high_mask[key].size
    and_out = high_mask[key]&low_mask[key]
    sim_out = np.sum(and_out) / total_num
    mask_out["sim"].append(sim_out)
    mask_out["diff"].append(1-sim_out)

len = np.arange(len(mask_out["sim"]))
width = 0.25
fig, ax = plt.subplots(figsize=(5,4))
rects1 = ax.bar(len - width/2, mask_out['sim'], width, label='sim')
rects2 = ax.bar(len + width/2, mask_out['diff'], width, label='diff')
ax.set_ylabel('proportion')
ax.set_xlabel('layer index')
ax.legend()
plt.savefig(args.name)