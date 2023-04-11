from __future__ import annotations

from PIL import Image, ImageDraw
import numpy as np
import cv2

import argparse

from model import _DetModel, _PoseModel
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')

    ### Input Parameters
    parser.add_argument('--image-filename', type=str, default='test_images/diver01.jpg')

    ### Output Parameters
    parser.add_argument('--save-outputs', type=bool, default=True)

    ### Human Detector and Model Parameters
    parser.add_argument('--detector-name', type=str, default='YOLOX-x')
    parser.add_argument('--pose-model-name', type=str, default='ViTPose-B*')

    ### Hand Detector and Model Parameters

    ### Bounding Box Visualization Parameters
    parser.add_argument('--vis-det-score-threshold', default=0.3, help='Bounding box score threshold to visualize.')

    ### Pose Visualization Parameters
    parser.add_argument('--det-score-threshold', default=0.5, help='Bounding box score threshold to estimate pose.')
    parser.add_argument('--vis-kpt-score-threshold', default=0.7)
    parser.add_argument('--vis-dot-radius', default=4)
    parser.add_argument('--vis-line-thickness', default=2)

    ### Pose Augmentation Parameters
    parser.add_argument('--kpt-number', type=int, default=11, help='17 will extract all the keypoints. 11 will extract the upper body.')

    return parser.parse_args()

def main():
    args = parse_args()
    det_model = _DetModel(device='cpu')
    det_model.set_model(args.detector_name)
    poseModel = _PoseModel(device='cpu')
    #handDet_model = _DetModel(device='cuda')
    #handDet_model.set_model('Hand2')

    #handDet_preds, handDetection_visualization = handDet_model.run('Hand', input_image, 0.5)


    testVideoPath = 'croppedVideos/Underwater06.mp4'
    outputFile = 'croppedVideos/handTests/Underwater06_.mp4'
    cap = cv2.VideoCapture(testVideoPath)
    frameCounter = 0
    counter = 0

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    fps = int(cap.get(5))
    output = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, frame_size)

    handDetected = False

    while(cap.isOpened()):
        frameCounter+=1
        ret, frame = cap.read()

        if ret:            
            det_preds, detection_visualization = det_model.run(args.detector_name, np.asarray(frame), args.vis_det_score_threshold)
            posePredictions, poseVisualization = poseModel.run('WholeBody-V+S',             #PoseModel
                                                                np.asarray(frame),       #Input
                                                                det_preds,    #Detected Human Box
                                                                0.3,                     #Detection Threshold
                                                                0.1,          #Keypoint Visualization Threshold
                                                                4,                       #Keypoint Radius
                                                                2)                       #Line Thickness
            
            cv2.imshow('Frame',poseVisualization)
            output.write(poseVisualization)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
        else:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()