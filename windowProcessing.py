import cv2
import numpy as np
from PIL import Image, ImageDraw
import poseAugmentation as poseAug
from model import _DetModel, _PoseModel

def GaussianSmoothPose(windowPoses,b,keypointNumber=11):
    smoothXCoord = []
    smoothYCoord = []
    for i, (xCoordRef, yCoordRef, _) in enumerate(windowPoses[3][:keypointNumber,:]):
        xCoords = []
        yCoords = []
        for windowPose in windowPoses:
            xCoords.append(windowPose[i][0])
            yCoords.append(windowPose[i][1])
        gkvX = np.exp(-(((3-np.array([0,1,2,3,4,5,6]))**2)/(2*(b**2))))
        gkvY = np.exp(-(((3-np.array([0,1,2,3,4,5,6]))**2)/(2*(b**2))))

        gkvX = gkvX/gkvX.sum()
        gkvY = gkvY/gkvY.sum()

        smoothXCoord.append((xCoords*gkvX).sum())
        smoothYCoord.append((yCoords*gkvY).sum())

    newWindowPose = windowPoses[3]
    newWindowPose[:keypointNumber,0] = smoothXCoord
    newWindowPose[:keypointNumber,1] = smoothYCoord
    
    return newWindowPose

def main(windowSize=7,device="cpu", visualize = True, jointReplacement = True, poseFilter = True, saveOutput = True, outputType = 'compare'):
    rawWindowFrames = []
    rawPoseVisualizations = []
    windowFrames = []
    windowPoses = []
    posePreds = []
    keypointNumber = 11
    replacementCounter = np.zeros((11,1))
    misdetectionCounter = 0
    maxConsecutiveReplacements = 7
    jointThreshold = 0.3

    videoFile = "croppedVideos/Normal04.mp4"
    outputFile = "croppedVideos/windowTests/Normal04_vs.mp4"

    detectionModel = _DetModel(device=device)
    poseModel = _PoseModel(device=device)
    detectionModel.set_model("YOLOX-x")

    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    

    if saveOutput:
        if outputType == 'solo':
            frame_size = (width,height)
            output = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, frame_size)
        elif outputType == 'compare':
            frame_size = (width*3,height)
            output = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, frame_size)

    for i in range(windowSize):
        rawWindowFrames.append(np.zeros((height,width,3),dtype=np.uint8))
        windowFrames.append(np.zeros((height,width,3),dtype=np.uint8))
        windowPoses.append(np.zeros((17,3)))
        posePreds.append(None)
        rawPoseVisualizations.append(np.zeros((height,width,3),dtype=np.uint8))

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
            
            rawWindowFrames.pop(0)
            rawWindowFrames.append(frame)
            rawPoseVisualizations.pop(0)
            rawPoseVisualizations.append(poseVisualization)
            windowFrames.pop(0)
            windowPoses.pop(0)
            posePreds.pop(0)

            # Dealing with the lack of human detection in the frame:
            if len(posePredictions) == 0 and misdetectionCounter < maxConsecutiveReplacements:
                posePredictions = posePreds[-1]
                misdetectionCounter += 1
                print(f"Replaced human detection and whole skeleton for {misdetectionCounter} consecutive times.")
            elif len(posePredictions) == 0 and misdetectionCounter >= maxConsecutiveReplacements:
                posePredictions = posePreds[-1]
                posePredictions[0]['keypoints'][:,2] = 0    # If there were too many consecutive misdetections, set all keypoint confidence values to 0
                misdetectionCounter += 1
                print(f"{misdetectionCounter} consecutive misdetections, setting pose predictions to 0.")
            else:
                misdetectionCounter = 0

            try:
                posePredictions[0]['keypoints'][keypointNumber:,2] = 0    # Cleaning up pose keypoints to exclude lower body.
                windowPoses.append(posePredictions[0]['keypoints'])
                posePreds.append(posePredictions)

                # Check if there are any missing / below threshold keypoints on Last Frame:
                if jointReplacement:
                    for i, (xCoord, yCoord, jointConf) in enumerate(windowPoses[-1][:keypointNumber,:]):
                        if jointConf < jointThreshold:
                            prevXCoord, prevYCoord, prevJointConf = windowPoses[-2][i]
                            if prevJointConf >= jointThreshold and replacementCounter[i] < maxConsecutiveReplacements and misdetectionCounter < maxConsecutiveReplacements:
                                windowPoses[-1][i] = windowPoses[-2][i]
                                replacementCounter[i] += 1
                                print(f"Replaced joint {i} for {replacementCounter[i]} consecutive times.")
                            else:
                                replacementCounter[i] = 0
                        else:
                            replacementCounter[i] = 0

                    # Update pose results on Last Frame for visualization:
                    posePredictions[0]['keypoints'] = windowPoses[-1]

                if poseFilter:
                    # Gaussian Smoothing on Middle Frame
                    if posePreds[3] is not None:
                        newWindowPose = GaussianSmoothPose(windowPoses,3)
                        middleFrame = rawWindowFrames[3]
                        posePredictionsMiddle = posePreds[3]
                        posePredictionsMiddle[0]['keypoints'] = newWindowPose
                        poseVisMiddle = poseModel.visualize_pose_results(np.asarray(middleFrame),
                                                           posePredictionsMiddle,
                                                           jointThreshold,
                                                           4,
                                                           2
                                                           )
                        windowFrames[3] = poseVisMiddle

                poseVisLast = poseModel.visualize_pose_results(np.asarray(frame),
                                                           posePredictions,
                                                           jointThreshold,
                                                           4,
                                                           2
                                                           )
                windowFrames.append(poseVisLast)
            except Exception as e:
                print(f"Unexpected {e}, {type(e)}")
                windowFrames.append(poseVisualization)
                windowPoses.append(np.zeros((17,3)))
                posePreds.append(posePredictions)
            
            row1 = np.concatenate(windowFrames[:3], axis=1)
            row2 = np.concatenate([np.zeros((height,width,3),dtype=np.uint8),windowFrames[3],np.zeros((height,width,3),dtype=np.uint8)], axis=1)
            row3 = np.concatenate(windowFrames[4:], axis=1)

            completeDisplay = np.concatenate([row1, row2, row3], axis=0)

            resizedScale = 0.8
            resizedWidth = int(width*resizedScale)
            resizedHeight = int(height*resizedScale)

            resizedDisplay = cv2.resize(completeDisplay, (resizedWidth,resizedHeight), interpolation=cv2.INTER_AREA)
            cv2.imshow('Frame',resizedDisplay)

            if saveOutput:
                if outputType == 'solo':
                    focusFrame = 3
                    frameToSave = windowFrames[focusFrame]
                    output.write(frameToSave)
                elif outputType == 'compare':
                    focusFrame = 3
                    rawFr = rawWindowFrames[focusFrame]
                    poseFr = rawPoseVisualizations[focusFrame]
                    procFr = windowFrames[focusFrame]
                    frameToSave = np.concatenate([rawFr,poseFr,procFr], axis=1)
                    output.write(frameToSave)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
        else:
            break
    
    cap.release()
    if saveOutput:
        output.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main(device="cuda")