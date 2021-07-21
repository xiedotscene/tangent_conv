import argparse
parser = argparse.ArgumentParser(description='Tangent convolutions')
parser.add_argument('config', type=str, metavar='config', help='config json file')
parser.add_argument('scan', type=str, metavar='scan', help='scan name')
parser.add_argument('label', type=str, metavar='label', help='using label in [g (GT), p (prediction)]')
parser.add_argument('--raw', action='store_true', help='operating raw data')
args = parser.parse_args()

from util.path_config import *
from util.dataset_params import *
from util.cloud import *

class param:
    def __init__(self, config):
        self.dataset_dir = get_tc_path() + config['pre_dataset_dir']
        self.parametrized_dir = get_tc_path() + config['pre_output_dir']
        self.output_dir = os.path.join(get_tc_path(), config['co_experiment_dir'], config['co_output_dir'])
        self.vis_dir = os.path.join(get_tc_path(), config['co_vis_dir'])
        dataset_type = config['pre_dataset_param']
        if dataset_type == "stanford":
            self.d_par = stanford_params()
        elif dataset_type == "scannet":
            self.d_par = scannet_params()
        elif dataset_type == "semantic3d":
            self.d_par = semantic3d_params()
        elif dataset_type == "dotsceneIn":
            self.d_par = dotsceneIn_params()
        elif dataset_type == "dotsceneOut":
            self.d_par = dotsceneOut_params()

# reading configuration files
import os, sys
sys.path.append('util')
from config_reader import *
from cloud import *
config = config_reader(args.config)
par = param(config)
scan_name = args.scan
label = args.label

if args.raw:
    cloud_file_name = os.path.join(par.dataset_dir, scan_name, "scan.pcd")
    pcd = read_point_cloud(cloud_file_name)
    estimate_normals(pcd, search_param=KDTreeSearchParamHybrid(radius=1.0, max_nn=100))
    if label == 'g':
        labels_file_name = os.path.join(par.dataset_dir, scan_name, "scan.labels")
    elif label == 'p':
        labels_file_name = os.path.join(par.output_dir, scan_name, "extrapolated.labels")
    lbl = read_txt_labels(labels_file_name)
    remapped = remap_colors(np.asarray(lbl, dtype='int32'), color_map=par.d_par.color_map) / 255.0
    idx = np.in1d(lbl, np.asarray([0]), invert=True)

    pcd.points = Vector3dVector(np.asarray(pcd.points)[idx])#[nonzero])
    pcd.colors = Vector3dVector(remapped[idx])
    # vis path
    if label == 'g':
        vis_cloud_path = os.path.join(par.vis_dir, "dense", "gt")
        scan_vis_name = scan_name + "_gt.pcd"
    elif label == 'p':
        vis_cloud_path = os.path.join(par.vis_dir, "dense", "pred")
        scan_vis_name = scan_name + "_pred.pcd"
else:
    if label == 'p':
        cloud_file_path = os.path.join(par.output_dir, scan_name)
    else:
        cloud_file_path = os.path.join(par.parametrized_dir, scan_name, '0')
    scan = ScanData()
    scan.load(cloud_file_path, 3)

    if label == 'g':
        lbl = scan.labels_gt[0]
    elif label == 'p':
        lbl = scan.labels_pr[0]
    remapped = remap_colors(np.copy(np.asarray(lbl, dtype='int32')), color_map=par.d_par.color_map) / 255.0
    idx = np.in1d(lbl, np.asarray([0]), invert=True)
    pcd = PointCloud()
    pcd.points = Vector3dVector(np.copy(np.asarray(scan.clouds[0].points)[idx]))
    pcd.colors = Vector3dVector(remapped[idx])
    estimate_normals(pcd)
    # vis path
    if label == 'g':
        vis_cloud_path = os.path.join(par.vis_dir, "sparse", "gt")
        scan_vis_name = scan_name + "_gt.pcd"
    elif label == 'p':
        vis_cloud_path = os.path.join(par.vis_dir, "sparse", "pred")
        scan_vis_name = scan_name + "_pred.pcd"

vis_cloud_name = os.path.join(vis_cloud_path, scan_vis_name)
# TODO: create folder if it does not exists
os.makedirs(vis_cloud_path, exist_ok=True)
# TODO: save pcd
write_point_cloud(vis_cloud_name, pcd)