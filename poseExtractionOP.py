from __future__ import annotations

from PIL import Image, ImageDraw
import numpy as np
import cv2

import argparse

from openpose import pyopenpose as op

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
    parser.add_argument('--vis-det-score-threshold', default=0.5, help='Bounding box score threshold to visualize.')

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

    params = dict()
    params["body"] = 1
    params["face"] = False
    params["face_detector"] = 1
    params["hand"] = False
    params["hand_detector"] = 1
    params["disable_blending"] = True
    params["model_folder"] = "/home/alejandro/openpose_folder/openpose/models/"

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    testVideoPath = 'test_videos/test2.mp4'
    cap = cv2.VideoCapture(testVideoPath)
    frameCounter = 0
    counter = 0

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    fps = 20
    output = cv2.VideoWriter('/home/alejandro/Documents/GitHub/simpleViTPose/test_videos/PoseDetection_01.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    while(cap.isOpened()):
        frameCounter+=1
        ret, frame = cap.read()

        if frameCounter < 510:
            pass
        elif frameCounter > 1470:
            break
        else:
            if ret:
                cv2.waitKey(1)
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                frame = datum.cvOutputData
                #cv2.imshow('Frame',handDetection_visualization)
                output.write(frame)
            else:
                break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()