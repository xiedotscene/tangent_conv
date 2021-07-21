import argparse
parser = argparse.ArgumentParser(description='check data')
parser.add_argument('--path', type=str, default='raw/semantic3d', help='relative path of data')
args = parser.parse_args()

import os
import numpy as np
import sys

root_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(root_dir, args.path)
folders = os.listdir(data_dir)
print(root_dir)

ws_dir = os.path.dirname(os.getcwd())
sys.path.append(ws_dir)
from util.path_config import *

for f in folders:
    data_path = os.path.join(data_dir, f, 'scan.pcd')
    label_path = os.path.join(data_dir, f, 'scan.labels')
    pcd = read_point_cloud(data_path)
    data = np.asarray(pcd.points)
    num_data = data.shape[0]
    num_label = sum(1 for line in open(label_path))

    # check if number of points corresponds to the number of labels
    if not num_data == num_label:
        print("Data number error with %s" %f)
        print("Points number: %d, Labels number: %d" %(num_data, num_label))
    else:
        print("Right data %s" %f)