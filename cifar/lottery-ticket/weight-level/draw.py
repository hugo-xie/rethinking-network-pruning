import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

parser.add_argument('--ori', default='', type=str, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--inverse', default='0', type=str, metavar='N',
                    help='manual epoch number (useful on restarts)')
args = parser.parse_args()

ori_data = pd.read_table(args.ori, sep='\t')
print(ori_data)