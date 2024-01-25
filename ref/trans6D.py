# encoding: utf-8
"""This file includes necessary params, info."""
import os
import os.path as osp
import mmcv
import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")
# ---------------------------------------------------------------- #
# TLESS DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "trans6D")
train_real_dir = osp.join(dataset_root, "train_pbr")
#train_render_dir = osp.join(dataset_root, "train_render_reconst")
test_dir = osp.join(dataset_root, "test")

# model_dir = osp.join(dataset_root, "models_reconst")  # use recon models as default
model_dir = osp.join(dataset_root, "models")
#model_cad = osp.join(dataset_root, "models_cad")
#model_reconst_dir = osp.join(dataset_root, "models_reconst")
model_eval_dir = osp.join(dataset_root, "models")
vertex_scale = 0.001
# object info
objects = [str(i) for i in range(1, 11)]
id2obj = {i: str(i) for i in range(1, 11)}

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)]  # for renderer

# Camera info
width = 640
height = 480
zNear = 0.25
zFar = 6.0
center = (height / 2, width / 2)

# NOTE: for tless, the camera matrix is not fixed!
camera_matrix = np.array([568.8888888888889, 0.0, 320.0, 0.0, 568.8888888888889, 240.0, 0.0, 0.0, 1.0]).reshape(3, 3)


diameters = (
    np.array(
        [   0.11163869500160217,
            0.11056650429964066,
            0.09451869130134583,
            0.10559443384408951,
            0.12061885744333267,
            0.07563385367393494,
            0.08112307637929916,
            0.10718723386526108,
            0.10594763606786728,
            0.08775851130485535
        ]
    )
)


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


# ref core/gdrn_modeling/tools/tless/tless_1_compute_fps.py
def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


# ref core/gdrn_modeling/tools/tless/tless_1_compute_keypoints_3d.py
def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    kpts_dict = mmcv.load(keypoints_3d_path)
    return kpts_dict
