from __future__ import annotations

import pathlib
import sys

#if os.getenv('SYSTEM') == 'spaces':
#    import mim

    #mim.uninstall('mmcv-full', confirm_yes=True)
    #mim.install('mmcv-full==1.5.0', is_yes=True)

    #subprocess.run('pip uninstall -y opencv-python'.split())
    #subprocess.run('pip uninstall -y opencv-python-headless'.split())
    #subprocess.run('pip install opencv-python-headless==4.5.5.64'.split())

import numpy as np
import torch
import torch.nn as nn

app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / 'ViTPose/'
sys.path.insert(0, submodule_dir.as_posix())

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)


class DetModel:
    MODEL_DICT = {
        'YOLOX-tiny': {
            'config':
            'mmdet_configs/configs/yolox/yolox_tiny_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        },
        'YOLOX-s': {
            'config':
            'mmdet_configs/configs/yolox/yolox_s_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
        },
        'YOLOX-l': {
            'config':
            'mmdet_configs/configs/yolox/yolox_l_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        },
        'YOLOX-x': {
            'config':
            'mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
        },
        'Hand': {   # Hand Detector
            'config':
            'mmdet_configs/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1class.py',
            'model':
            'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth',
        },
    }

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        #self._load_all_models_once()
        self.model_name = 'YOLOX-l'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        return init_detector(dic['config'], dic['model'], device=self.device)

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def detect_and_visualize(
            self, image: np.ndarray,
            score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        out = inference_detector(self.model, image)
        return out

    def visualize_detection_results(
            self,
            image: np.ndarray,
            detection_results: list[np.ndarray],
            score_threshold: float = 0.3) -> np.ndarray:
        person_det = [detection_results[0]] + [np.array([]).reshape(0, 5)] * 79

        image = image[:, :, ::-1]  # RGB -> BGR
        vis = self.model.show_result(image,
                                     person_det,
                                     score_thr=score_threshold,
                                     bbox_color=None,
                                     text_color=(200, 200, 200),
                                     mask_color=None)
        return vis[:, :, ::-1]  # BGR -> RGB


class _DetModel(DetModel):
    def run(self, model_name: str, image: np.ndarray,
            score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        self.set_model(model_name)
        return self.detect_and_visualize(image, score_threshold)


class PoseModel:
    MODEL_DICT = {
        'ViTPose-B': {
            'config':
            'vitpose_configs/ViTPose_base_coco_256x192.py',
            'model': 'vitpose_models/vitpose-b.pth',
        },
        'ViTPose-L': {
            'config':
            'vitpose_configs/ViTPose_large_coco_256x192.py',
            'model': 'vitpose_models/vitpose-l.pth',
        },
        'ViTPose-B*': {
            'config':
            'vitpose_configs/ViTPose_base_coco_256x192.py',
            'model': 'vitpose_models/vitpose-b-multi-coco.pth',
        },
        'ViTPose-L*': {
            'config':
            'vitpose_configs/ViTPose_large_coco_256x192.py',
            'model': 'vitpose_models/vitpose-l-multi-coco.pth',
        },
        'Hand': { # Hand Pose Estimator
            'config':
            'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py',
            'model': 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth',
        },
    }

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model_name = 'ViTPose-B*'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        model = init_pose_model(dic['config'], dic['model'], device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB


class _PoseModel(PoseModel):
    def run(
        self, model_name: str, image: np.ndarray,
        det_results: list[np.ndarray], box_score_threshold: float,
        kpt_score_threshold: float, vis_dot_radius: int,
        vis_line_thickness: int
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        self.set_model(model_name)
        return self.predict_pose_and_visualize(image, det_results,
                                               box_score_threshold,
                                               kpt_score_threshold,
                                               vis_dot_radius,
                                               vis_line_thickness)