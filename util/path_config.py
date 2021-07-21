import sys

open3d_path = '/home/xie/Open3D/build/lib/'
tc_path = '/home/xie/tangent_conv/'

sys.path.append(open3d_path)
from py3d import *

def get_tc_path():
	return tc_path

import random