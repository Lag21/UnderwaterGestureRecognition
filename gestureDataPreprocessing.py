import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import mediapipe as mp
from torch import nn
from PIL import Image, ImageDraw
import poseAugmentation as poseAug
from model import _DetModel, _PoseModel
from customModels import DepthEstimator, StaticGestureRecognizer
from spatialAttentionModule import GestureRecognition

def main(detectionModel, poseModel, depthModel, staticGestureModel, mediaPipe):
    datasetPath = '../../Datasets/ChaLearnLSSI/'

    data = pd.read_csv(datasetPath+'train_labels.csv',header=None)

    counter = 0
    videoLengths = []

    for idx, row in tqdm(data.iterrows()):
        file = row[0]
        gestureLabel = row[1]
        counter+=1
        if counter >= 2:
            break

        pathfile = 'train/'+file+'_color.mp4'

        leftHandVector, rightHandVector, augmentedPoseVector, xDyn = GestureRecognition(datasetPath+pathfile, None, detectionModel, poseModel, depthModel, staticGestureModel, mediaPipe, device=device)

        print(len(leftHandVector), len(rightHandVector), len(augmentedPoseVector), len(xDyn))
        print(type(leftHandVector), type(rightHandVector), type(augmentedPoseVector), type(xDyn))
        #cap = cv2.VideoCapture(datasetPath+pathfile)
        #width = int(cap.get(3))
        #height = int(cap.get(4))
        #fps = int(cap.get(5))

        #while(cap.isOpened()):
        #    ret, frame = cap.read()

        #    if ret:
                #cv2.imshow('Video',frame)
        #        pressedKey = cv2.waitKey(1) & 0xFF
        #        if pressedKey == ord('q'):
        #            break
        #    else:
        #        break

        #cap.release()
        #cv2.destroyAllWindows()

if __name__=="__main__":
    device = 'cuda'

    # Initializing detector and pose estimator.
    detectionModel = _DetModel(device=device)
    poseModel = _PoseModel(device=device)
    detectionModel.set_model("YOLOX-x")

    # Initializing the depth estimator.
    depthModel = DepthEstimator()
    depthModel.load_state_dict(torch.load("depthEstimatorModels/depthEstimator02.pth"))
    depthModel.eval()

    # Initializing the static gesture recognizer.
    staticGestureModel = StaticGestureRecognizer(device="cuda")
    staticGestureModel.LoadModel(pathfile='trainedModels/inception-v3-tlearning02.pth')

    # Initializing MediaPipe.
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mediaPipe = (mp_drawing,mp_drawing_styles,mp_hands)

    main(detectionModel, poseModel, depthModel, staticGestureModel, mediaPipe)
