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
    currentSigner = 'signer2_'

    data = pd.read_csv(datasetPath+'train_labels.csv',header=None)

    dataBySigner = data[data[0].str.contains(currentSigner)]

    if dataBySigner.empty:
        print("This signer does not have any train data.")
        return

    counter = 0
    videoLengths = []

    processedSamples = glob('DatasetSamples/ChaLearnLSSIEmbeddings/*.npy')

    for idx, row in tqdm(dataBySigner.iterrows()):
        sampleStatus = False
        file = row[0]
        gestureLabel = row[1]
        pathfile = 'train/'+file+'_color.mp4'
        
        for processedSample in processedSamples:
            if file in processedSample:
                sampleStatus=True
                break
        
        if sampleStatus:
            continue

        try:
            leftHandVector, rightHandVector, augmentedPoseVector, xDyn = GestureRecognition(datasetPath+pathfile,
                                                                                            None,
                                                                                            detectionModel,
                                                                                            poseModel,
                                                                                            depthModel,
                                                                                            staticGestureModel,
                                                                                            mediaPipe,
                                                                                            device=device,
                                                                                            visualize=False,
                                                                                            verbose=False)

            #print(len(leftHandVector), len(rightHandVector), len(augmentedPoseVector), len(xDyn))
            #print(type(leftHandVector), type(rightHandVector), type(augmentedPoseVector), type(xDyn))
            np.save('DatasetSamples/ChaLearnLSSIEmbeddings/'+file+'.npy',xDyn,allow_pickle=True)
        except Exception as e:
            print(f'Had issues with file {file}:')
            print(f"Unexpected {e}, {type(e)}")

        counter+=1
        if counter >= 150:
            break


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
