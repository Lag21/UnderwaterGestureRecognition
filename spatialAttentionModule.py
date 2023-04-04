import cv2
import copy
import numpy as np
import torch
###
import mediapipe as mp
###
from torch import nn
from PIL import Image, ImageDraw
import poseAugmentation as poseAug
from model import _DetModel, _PoseModel
from depthEstimator import DepthEstimator

def GestureRecognition(windowSize=7, device="cpu", visualize = True, jointReplacement = True, poseFilter = True, saveOutput = False, outputType = 'compare'):
    # Initialization of parameters and arrays
    rawWindowFrames = []
    rawPoseVisualizations = []
    windowFrames = []
    windowPoses = []
    posePreds = []
    augmentedPoses = []
    leftHandLocations = []
    leftHandCorners = []
    rightHandLocations = []
    rightHandCorners = []
    keypointNumber = 11
    replacementCounter = np.zeros((11,1))
    misdetectionCounter = 0
    maxConsecutiveReplacements = 7
    jointThreshold = 0.3

    # Input and output paths.
    videoFile = "croppedVideos/Underwater01.mp4"
    outputFile = "croppedVideos/windowTests/Underwater01_vs.mp4"

    # Initializing detector and pose estimator.
    detectionModel = _DetModel(device=device)
    poseModel = _PoseModel(device=device)
    detectionModel.set_model("YOLOX-x")

    # Initializing MediaPipe.
    ###
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    ###

    # Initializing the depth estimator.
    depthModel = DepthEstimator()
    depthModel.load_state_dict(torch.load("depthEstimatorModels/depthEstimator02.pth"))
    depthModel.eval()

    # Initializing video input.
    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))

    # Initializing video output.
    if saveOutput:
        if outputType == 'solo':
            frame_size = (width,height)
            output = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, frame_size)
        elif outputType == 'compare':
            frame_size = (width*3,height)
            output = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('m','p','4','v'), fps, frame_size)

    # Initializing the window.
    for i in range(windowSize):
        rawWindowFrames.append(np.zeros((height,width,3),dtype=np.uint8))
        windowFrames.append(np.zeros((height,width,3),dtype=np.uint8))
        windowPoses.append(np.zeros((17,3)))
        posePreds.append(None)
        rawPoseVisualizations.append(np.zeros((height,width,3),dtype=np.uint8))
        augmentedPoses.append(None)
        leftHandLocations.append(None)
        leftHandCorners.append(None)
        rightHandLocations.append(None)
        rightHandCorners.append(None)
        

    ###
    with mp_hands.Hands(model_complexity=1,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    ###

        # Reading the video. 
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
                
                # Removing the information from the first frame of the window (now outside).
                rawWindowFrames.pop(0)
                rawPoseVisualizations.pop(0)
                windowFrames.pop(0)
                windowPoses.pop(0)
                posePreds.pop(0)
                leftHandLocations.pop(0)
                leftHandCorners.pop(0)
                rightHandLocations.pop(0)
                rightHandCorners.pop(0)
                #augmentedPoses.pop(0)  # maybe not needed.

                # Adding newly acquired data.
                rawWindowFrames.append(frame)
                rawPoseVisualizations.append(poseVisualization)

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
                            middleFrame = rawWindowFrames[3].copy()
                            posePredictionsMiddle = copy.deepcopy(posePreds[3])
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

                # Pose Augmentation.
                ## This is done in the middle frame, after gaussian smoothing.
                if posePreds[3] is not None:
                    augmentedPose = poseAug.augmentPose(copy.deepcopy(posePredictionsMiddle), jointStructure="body")
                    
                    augPose = []
                    for array in augmentedPose:
                        for subarray in array:
                            augPose.append(subarray)

                # Depth Estimation.
                # This is done only after the pose is augmented.
                    estDepths = depthModel.forward(augPose)
                    neckDepth = estDepths[0].item()
                    nleftHandDepth = estDepths[1].item()
                    nrightHandDepth = estDepths[2].item()

                    if np.isnan(nleftHandDepth):
                        pass
                    else:
                        leftHandDepth = nleftHandDepth

                    if np.isnan(nrightHandDepth):
                        pass
                    else:
                        rightHandDepth = nrightHandDepth

                # Hand Extraction.
                # This is done in the middle frame, after gaussian smoothing.
                    handL = posePredictionsMiddle[0]['keypoints'][9][0:2]       # Coordinates from left wrist.
                    handR = posePredictionsMiddle[0]['keypoints'][10][0:2]      # Coordinates from right wrist.

                    resModifier = 40    ## NEEDS TO CHANGE FOR EACH VIDEO RESOLUTION
                    modL_ = int(resModifier*leftHandDepth/width)                         # Applying estimated depth modifier for left hand.
                    modR_ = int(resModifier*rightHandDepth/width)                        # Applying estimated depth modifier for right hand.

                    handL_ULx = int(handL[0])-modL_
                    handL_ULy = int(handL[1])-modL_
                    handR_ULx = int(handR[0])-modR_
                    handR_ULy = int(handR[1])-modR_

                    curLeftHandCorner = (handL_ULx,handL_ULy)
                    curRightHandCorner = (handR_ULx,handR_ULy)

                    handL_ = rawWindowFrames[3].copy()[int(handL[1])-modL_:int(handL[1])+modL_,int(handL[0])-modL_:int(handL[0])+modL_]
                    handR_ = rawWindowFrames[3].copy()[int(handR[1])-modR_:int(handR[1])+modR_,int(handR[0])-modR_:int(handR[0])+modR_]

                    cv2.imshow('Right Hand Raw',handR_)
                    cv2.imshow('Left Hand Raw',handL_)

                    if leftHandLocations[-1] is not None:
                        prevXMin, prevXMax, prevYMin, prevYMax = leftHandLocations[-1]
                        prev_ULx, prev_ULy = leftHandCorners[-1]   # Corner of left hand frame in previous frame
                        handL_ = rawWindowFrames[3].copy()[int(prev_ULy+prevYMin):int(prev_ULy+prevYMax),int(prev_ULx+prevXMin):int(prev_ULx+prevXMax)]
                        if np.array(handL_).shape[0] == 0 or np.array(handL_).shape[1] == 0:
                            handL_ = rawWindowFrames[3].copy()[int(handL[1])-modL_:int(handL[1])+modL_,int(handL[0])-modL_:int(handL[0])+modL_]
                        else:
                            curLeftHandCorner = (int((prev_ULx+prevXMin)*0.9),int((prev_ULy+prevYMin)*0.9))

                    if rightHandLocations[-1] is not None:
                        prevXMin, prevXMax, prevYMin, prevYMax = rightHandLocations[-1]
                        prev_ULx, prev_ULy = rightHandCorners[-1]
                        handR_ = rawWindowFrames[3].copy()[int(prev_ULy+prevYMin):int(prev_ULy+prevYMax),int(prev_ULx+prevXMin):int(prev_ULx+prevXMax)]
                        if np.array(handR_).shape[0] == 0 or np.array(handR_).shape[1] == 0:
                            handR_ = rawWindowFrames[3].copy()[int(handR[1])-modR_:int(handR[1])+modR_,int(handR[0])-modR_:int(handR[0])+modR_]
                        else:
                            curRightHandCorner = (int((prev_ULx+prevXMin)*0.9),int((prev_ULy+prevYMin)*0.9))
                        
                    #print(np.array(handR_).shape)
                    handL_MP = copy.deepcopy(handL_)
                    handR_MP = copy.deepcopy(handR_)
                    xHandSizeL = np.array(handL_MP).shape[0]
                    yHandSizeL = np.array(handL_MP).shape[1]
                    xHandSizeR = np.array(handR_MP).shape[1]
                    yHandSizeR = np.array(handR_MP).shape[0]
                    
                    ###
                    handL_ = cv2.cvtColor(handL_, cv2.COLOR_BGR2RGB)
                    handR_ = cv2.cvtColor(handR_, cv2.COLOR_BGR2RGB)

                    resultsL_ = hands.process(handL_)
                    resultsR_ = hands.process(handR_)

                    handL_ = cv2.cvtColor(handL_, cv2.COLOR_RGB2BGR)
                    handR_ = cv2.cvtColor(handR_, cv2.COLOR_RGB2BGR)

                    #xMax, xMin, yMax, yMin = [0]*4

                    if resultsL_.multi_hand_landmarks:
                        for hand_landmarks in resultsL_.multi_hand_landmarks:
                            xMinL, xMaxL, yMinL, yMaxL = ExtractCoordinatesFromLandmark(hand_landmarks)
                            xMinL *= xHandSizeL
                            xMaxL *= xHandSizeL
                            yMinL *= yHandSizeL
                            yMaxL *= yHandSizeL

                            mp_drawing.draw_landmarks(handL_,
                                                      hand_landmarks,
                                                      mp_hands.HAND_CONNECTIONS,
                                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                                      mp_drawing_styles.get_default_hand_connections_style())
                        
                        #handL_MP_ = handL_MP[int(yMinL):int(yMaxL),int(xMinL):int(xMaxL)]
                        #cv2.imshow('Refocused Left Hand', handL_MP_)
                        handL_ = cv2.rectangle(handL_,(int(xMinL),int(yMinL)),(int(xMaxL),int(yMaxL)),(255,0,0),1)
                        leftHandLocations.append((xMinL*0.5,xMaxL*1.3,yMinL*0.5,yMaxL*1.3)) # Increasing the size of the box around the hand coordinates.
                    else:
                        leftHandLocations.append(None)
                            
                    if resultsR_.multi_hand_landmarks:
                        for hand_landmarks in resultsR_.multi_hand_landmarks:
                            xMinR, xMaxR, yMinR, yMaxR = ExtractCoordinatesFromLandmark(hand_landmarks)
                            xMinR *= xHandSizeR
                            xMaxR *= xHandSizeR
                            yMinR *= yHandSizeR
                            yMaxR *= yHandSizeR

                            mp_drawing.draw_landmarks(handR_,
                                                      hand_landmarks,
                                                      mp_hands.HAND_CONNECTIONS,
                                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                                      mp_drawing_styles.get_default_hand_connections_style())
                        
                        #handR_MP_ = handR_MP[int(yMinR):int(yMaxR),int(xMinR):int(xMaxR)]
                        #cv2.imshow('Refocused Right Hand', handR_MP_)
                        handR_ = cv2.rectangle(handR_,(int(xMinR),int(yMinR)),(int(xMaxR),int(yMaxR)),(0,255,0),1)
                        rightHandLocations.append((xMinR*0.5,xMaxR*1.3,yMinR*0.5,yMaxR*1.3)) # Increasing the size of the box around the hand coordinates.
                    else:
                        rightHandLocations.append(None)
                    
                    leftHandCorners.append(curLeftHandCorner)
                    rightHandCorners.append(curRightHandCorner)

                    cv2.imshow('Left Hand',handL_)
                    cv2.imshow('Right Hand',handR_)
                    ###


                # Pose Normalization.
                # This is done only after the depth is estimated.
                    neckX = (posePredictionsMiddle[0]['keypoints'][5][0]+posePredictionsMiddle[0]['keypoints'][6][0])/2
                    neckY = (posePredictionsMiddle[0]['keypoints'][5][1]+posePredictionsMiddle[0]['keypoints'][6][1])/2

                    # Calculating X distance from left shoulder to right shoulder, from left shoulder to right hip,
                    # from left hip to right shoulder, and from left hip to right hip.
                    xDistLSRS = abs(posePredictionsMiddle[0]['keypoints'][5][0]-posePredictionsMiddle[0]['keypoints'][6][0])
                    xDistLSRH = abs(posePredictionsMiddle[0]['keypoints'][5][0]-posePredictionsMiddle[0]['keypoints'][12][0])
                    xDistLHRS = abs(posePredictionsMiddle[0]['keypoints'][11][0]-posePredictionsMiddle[0]['keypoints'][6][0])
                    xDistLHRH = abs(posePredictionsMiddle[0]['keypoints'][11][0]-posePredictionsMiddle[0]['keypoints'][12][0])
                    # Selecting the biggest distance.
                    torsoWidth = max(xDistLSRS,xDistLSRH,xDistLHRS,xDistLHRH)
                    # Calculating Y distance from left shoulder to left hip, from left shoulder to right hip,
                    # from right shoulder to left hip, and from right shoulder to right hip.
                    yDistLSLH = abs(posePredictionsMiddle[0]['keypoints'][5][0]-posePredictionsMiddle[0]['keypoints'][11][0])
                    yDistLSRH = abs(posePredictionsMiddle[0]['keypoints'][5][0]-posePredictionsMiddle[0]['keypoints'][12][0])
                    yDistRSLH = abs(posePredictionsMiddle[0]['keypoints'][6][0]-posePredictionsMiddle[0]['keypoints'][11][0])
                    yDistRSRH = abs(posePredictionsMiddle[0]['keypoints'][6][0]-posePredictionsMiddle[0]['keypoints'][12][0])
                    # Selecting the biggest distance.
                    torsoHeight = max(yDistLSLH,yDistLSRH,yDistRSLH,yDistRSRH)
                    # Calculating the width-to-height torso ratio.
                    torsoRatio = torsoWidth/torsoHeight # == xScale/yScale

                    #bboxWidth = abs(int(posePredictionsMiddle[0]['bbox'][2])-int(posePredictionsMiddle[0]['bbox'][1]))
                    #bboxHeight = abs(int(posePredictionsMiddle[0]['bbox'][1])-int(posePredictionsMiddle[0]['bbox'][3]))
                    xScale = torsoWidth/(width/4)
                    yScale = xScale/torsoRatio

                    posePredictionsMiddle[0]['keypoints'] = (posePredictionsMiddle[0]['keypoints']-np.array([neckX-width/2,neckY-height/3,0]))#/[xScale, yScale, 1]

                    bboxFrame = np.zeros((height,width,3),dtype=np.uint8)
                    posePredictionsMiddle[0].pop('bbox')
                    test = poseModel.visualize_pose_results(bboxFrame,
                                                            posePredictionsMiddle,
                                                            jointThreshold,
                                                            4,
                                                            2
                                                            )
                    cv2.imshow('Test',cv2.resize(test,(int(width*0.5),int(height*0.5)),interpolation=cv2.INTER_AREA))

                # Visualization.
                row1 = np.concatenate(windowFrames[:3], axis=1)
                row2 = np.concatenate([np.zeros((height,width,3),dtype=np.uint8),windowFrames[3],np.zeros((height,width,3),dtype=np.uint8)], axis=1)
                row3 = np.concatenate(windowFrames[4:], axis=1)
                completeDisplay = np.concatenate([row1, row2, row3], axis=0)
                resizedScale = 0.8
                resizedWidth = int(width*resizedScale)
                resizedHeight = int(height*resizedScale)
                resizedDisplay = cv2.resize(completeDisplay, (resizedWidth,resizedHeight), interpolation=cv2.INTER_AREA)
                cv2.imshow('Frame',resizedDisplay)

                # Saving output.
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

def ExtractCoordinatesFromLandmark(hand_landmarks):
    xMax, xMin, yMax, yMin = None, None, None, None
    for coordinates in hand_landmarks.landmark:
        if xMax is None or xMax < coordinates.x:
            xMax = coordinates.x
        if xMin is None or xMin > coordinates.x:
            xMin = coordinates.x #if coordinates.x > 0 else xMin
        if yMax is None or yMax < coordinates.y:
            yMax = coordinates.y
        if yMin is None or yMin > coordinates.y:
            yMin = coordinates.y #if coordinates.y > 0 else yMin
    #xMin = xMin-0.05 if xMin >0.05 else xMin
    #xMax = xMax+0.05 if xMax <0.95 else xMax
    #yMin = yMin-0.05 if yMin >0.05 else yMin
    #yMax = yMax+0.05 if yMax <0.95 else yMax
    #return xMin, xMax, yMin, yMax   # 5% padding is added to the coordinates to avoid cropping the hand.
    return xMin-0.1, xMax+0.1, yMin-0.1, yMax+0.1   # 5% padding is added to the coordinates to avoid cropping the hand.

if __name__=="__main__":
    
    GestureRecognition(device="cuda")