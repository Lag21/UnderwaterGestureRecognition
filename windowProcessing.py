import cv2
import numpy as np
from PIL import Image, ImageDraw
import poseAugmentation as poseAug
from model import _DetModel, _PoseModel


def main(windowSize=7,device="cpu", visualize = True, performCorrection = True, saveOutput = False):
    windowFrames = []
    windowPoses = []
    keypointNumber = 11
    replacementCounter = np.zeros((11,1))
    maxConsecutiveReplacements = 7
    jointThreshold = 0.3

    videoFile = "croppedVideos/Underwater02.mp4"
    outputFile = "croppedVideos/windowTests/Underwater02_P.mp4"

    detectionModel = _DetModel(device=device)
    poseModel = _PoseModel(device=device)
    detectionModel.set_model("YOLOX-x")

    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    frame_size = (width,height)

    output = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, frame_size)

    for i in range(windowSize):
        windowFrames.append(np.zeros((height,width,3),dtype=np.uint8))
        windowPoses.append(np.zeros((17,3)))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            detectionPredictions, detectionVisualization = detectionModel.run("YOLOX-x", np.asarray(frame), 0.3) #(DetectionModel, Image, DetectionThreshold)
            posePredictions, poseVisualization = poseModel.run('ViTPose-B*',             #PoseModel
                                                                np.asarray(frame),       #Input
                                                                detectionPredictions,    #Detected Human Box
                                                                0.3,                     #Detection Threshold
                                                                jointThreshold,          #Keypoint Visualization Threshold
                                                                4,                       #Keypoint Radius
                                                                2)                       #Line Thickness
            
            windowFrames.pop(0)
            windowPoses.pop(0)
            try:
                posePredictions[0]['keypoints'][keypointNumber:,2] = 0    # Cleaning up pose keypoints to exclude lower body.
                windowPoses.append(posePredictions[0]['keypoints'])

                # Check if there are any missing / below threshold keypoints
                if performCorrection:
                    for i, (xCoord, yCoord, jointConf) in enumerate(windowPoses[-1][:keypointNumber,:]):
                        if jointConf < jointThreshold:
                            prevXCoord, prevYCoord, prevJointConf = windowPoses[-2][i]
                            if prevJointConf >= jointThreshold and replacementCounter[i] < maxConsecutiveReplacements:
                                windowPoses[-1][i] = windowPoses[-2][i]
                                replacementCounter[i] += 1
                                print(f"Replaced joint {i} for {replacementCounter[i]} consecutive times.")
                            else:
                                replacementCounter[i] = 0
                        else:
                            replacementCounter[i] = 0

                    # Update pose results for visualization:
                    posePredictions[0]['keypoints'] = windowPoses[-1]
                        

                poseVis = poseModel.visualize_pose_results(np.asarray(frame),
                                                           posePredictions,
                                                           jointThreshold,
                                                           4,
                                                           2
                                                           )
                windowFrames.append(poseVis)
            except Exception as e:
                print(f"Unexpected {e}, {type(e)}")
                windowFrames.append(poseVisualization)
                windowPoses.append(np.zeros((17,3)))
            
            row1 = np.concatenate(windowFrames[:3], axis=1)
            row2 = np.concatenate([np.zeros((height,width,3),dtype=np.uint8),windowFrames[3],np.zeros((height,width,3),dtype=np.uint8)], axis=1)
            row3 = np.concatenate(windowFrames[4:], axis=1)

            completeDisplay = np.concatenate([row1, row2, row3], axis=0)

            resizedScale = 0.8
            resizedWidth = int(width*resizedScale)
            resizedHeight = int(height*resizedScale)

            resizedDisplay = cv2.resize(completeDisplay, (resizedWidth,resizedHeight), interpolation=cv2.INTER_AREA)
            cv2.imshow('Frame',resizedDisplay)
            output.write(poseVis)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
        else:
            break
    
    cap.release()
    output.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main(device="cuda")