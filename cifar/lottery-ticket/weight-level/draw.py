import pandas as pd
import argparse
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

parser.add_argument('--ori', default='', type=str, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--inverse', default='0', type=str, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--name', default="pruned.png", type=str)
args = parser.parse_args()

import pdb; pdb.set_trace()
ori_data = pd.read_table(args.ori, sep='\t')
print(ori_data)

sns.lineplot(x=range(len(ori_data['Valid Acc.'])), y=ori_data['Valid Acc.'], color='red',dashes=True)
plt.xlabel("episode")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(args.name)