import cv2
import numpy as np
from PIL import Image, ImageDraw
import poseAugmentation as poseAug
from model import _DetModel, _PoseModel


def main(windowSize=7,device="cpu",visualize = True):
    windowFrames = []

    videoFile = "croppedVideos/Normal01.mp4"

    detectionModel = _DetModel(device=device)
    poseModel = _PoseModel(device=device)
    detectionModel.set_model("YOLOX-x")

    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))

    for i in range(windowSize):
        windowFrames.append(np.zeros((height,width,3),dtype=np.uint8))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            detectionPredictions, detectionVisualization = detectionModel.run("YOLOX-x", np.asarray(frame), 0.3) #(DetectionModel, Image, DetectionThreshold)
            posePredictions, poseVisualization = poseModel.run('ViTPose-B*',             #PoseModel
                                                                np.asarray(frame),       #Input
                                                                detectionPredictions,    #Detected Human Box
                                                                0.3,                     #Detection Threshold
                                                                0.7,                     #Keypoint Visualization Threshold
                                                                4,                       #Keypoint Radius
                                                                2)                       #Line Thickness
            windowFrames.pop(0)
            windowFrames.append(poseVisualization)
            row1 = np.concatenate(windowFrames[:3], axis=1)
            row2 = np.concatenate([np.zeros((height,width,3),dtype=np.uint8),windowFrames[3],np.zeros((height,width,3),dtype=np.uint8)], axis=1)
            row3 = np.concatenate(windowFrames[4:], axis=1)

            completeDisplay = np.concatenate([row1, row2, row3], axis=0)

            resizedScale = 0.8
            resizedWidth = int(width*resizedScale)
            resizedHeight = int(height*resizedScale)

            resizedDisplay = cv2.resize(completeDisplay, (resizedWidth,resizedHeight), interpolation=cv2.INTER_AREA)
            cv2.imshow('Frame',resizedDisplay)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main(device="cuda")