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
from customModels import DepthEstimator, StaticGestureRecognizer

## TO DO - TUESDAY:
## Test hand extraction improvements on more videos of new dataset.
## Finish implementing dynamic model.
## Perform sample training on dynamic model.

def GestureRecognition(videoFile, outputFile, detectionModel, poseModel, depthModel, staticGestureModel, mediaPipe,
                       windowSize=7,
                       device="cpu",
                       visualize = True,
                       verbose = True,
                       jointReplacement = True,
                       poseFilter = True,
                       saveOutput = False,
                       outputType = 'compare',
                       handTracking = 'whole_image'):
    # Initialization of parameters and arrays
    rawWindowFrames = []
    rawPoseVisualizations = []
    windowFrames = []
    windowPoses = []
    posePreds = []
    normalizedPosePreds = []
    normalizedAugmentedPoses = []
    leftHandLocations = []
    leftHandCorners = []
    rightHandLocations = []
    rightHandCorners = []
    keypointNumber = 11
    replacementCounter = np.zeros((11,1))
    misdetectionCounter = 0
    maxConsecutiveReplacements = 7
    jointThreshold = 0.3

    if handTracking=='whole_image':
        leftHandCoordinates = []
        rightHandCoordinates = []
        leftHandMisdetections = 0
        rightHandMisdetections = 0

    # Initializing detector and pose estimator.
    #detectionModel = _DetModel(device=device)
    #poseModel = _PoseModel(device=device)
    #detectionModel.set_model("YOLOX-x")

    # Initializing MediaPipe.
    ###
    #mp_drawing = mp.solutions.drawing_utils
    #mp_drawing_styles = mp.solutions.drawing_styles
    #mp_hands = mp.solutions.hands
    mp_drawing, mp_drawing_styles, mp_hands = mediaPipe
    ###

    # Initializing the depth estimator.
    if handTracking == 'individual_image':
        depthModel = DepthEstimator()
        depthModel.load_state_dict(torch.load("depthEstimatorModels/depthEstimator02.pth"))
        depthModel.eval()

    # Initializing the static gesture recognizer.
    #staticGestureModel = StaticGestureRecognizer(device="cuda")
    #staticGestureModel.LoadModel(pathfile='trainedModels/inception-v3-tlearning02.pth')

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
        leftHandLocations.append(None)
        leftHandCorners.append(None)
        rightHandLocations.append(None)
        rightHandCorners.append(None)

    hands = 2 if handTracking == 'whole_image' else 1

    # To extract hands and augmented pose.
    leftHandVector = []
    rightHandVector = []
    augmentedPoseVector = []
    xDyn = []

    ###
    with mp_hands.Hands(model_complexity=1,
                    max_num_hands=hands,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    ###

        # Reading the video. 
        while(cap.isOpened()):
            ret, frame = cap.read()

            curLeftHand = None
            curLeftHandEmbed = None
            curRightHand = None
            curRightHandEmbed = None
            curAugmentedPose = None

            if ret:
                detectionPredictions, detectionVisualization = detectionModel.run("YOLOX-x", np.asarray(frame), 0.3) #(DetectionModel, Image, DetectionThreshold)
                posePredictions, poseVisualization = poseModel.run('ViTPose-B*',             #PoseModel
                                                                    np.asarray(frame),       #Input
                                                                    detectionPredictions,    #Detected Human Box
                                                                    0.3,                     #Detection Threshold
                                                                    jointThreshold,          #Keypoint Visualization Threshold
                                                                    4,                       #Keypoint Radius
                                                                    2)                       #Line Thickness
                
                # Checking if the window for frames needs to be initialized.
                if len(rawWindowFrames) == 0:
                    rawWindowFrames = [frame]*7
                    rawPoseVisualizations = [poseVisualization]*7
                # If not, remove the first element from the window and add the current information at the end.
                else:
                    rawWindowFrames.pop(0)
                    rawWindowFrames.append(frame)

                    rawPoseVisualizations.pop(0)
                    rawPoseVisualizations.append(poseVisualization)

                leftHandGestureFrame = None
                rightHandGestureFrame = None

                # Dealing with the lack of human detection in the frame:
                if len(posePredictions) == 0 and misdetectionCounter < maxConsecutiveReplacements:
                    posePredictions = posePreds[-1]
                    misdetectionCounter += 1
                    if verbose: print(f"Replaced human detection and whole skeleton for {misdetectionCounter} consecutive times.")
                elif len(posePredictions) == 0 and misdetectionCounter >= maxConsecutiveReplacements:
                    posePredictions = posePreds[-1]
                    posePredictions[0]['keypoints'][:,2] = 0    # If there were too many consecutive misdetections, set all keypoint confidence values to 0
                    misdetectionCounter += 1
                    if verbose: print(f"{misdetectionCounter} consecutive misdetections, setting pose predictions to 0.")
                else:
                    misdetectionCounter = 0

                try:
                    posePredictions[0]['keypoints'][keypointNumber:,2] = 0    # Cleaning up pose keypoints to exclude lower body.

                    # Checking if the window for poses needs to be initialized:
                    if len(windowPoses) == 0:
                        windowPoses = [posePredictions[0]['keypoints']]*7
                        posePreds = [posePredictions]*7
                    # If not, remove the first element from the window and add the current information at the end.
                    else:
                        windowPoses.pop(0)
                        windowPoses.append(posePredictions[0]['keypoints'])

                        posePreds.pop(0)
                        posePreds.append(posePredictions)

                    # Check if there are any missing / below threshold keypoints on Last Frame:
                    if jointReplacement:
                        for i, (xCoord, yCoord, jointConf) in enumerate(windowPoses[-1][:keypointNumber,:]):
                            if jointConf < jointThreshold:
                                prevXCoord, prevYCoord, prevJointConf = windowPoses[-2][i]
                                if prevJointConf >= jointThreshold and replacementCounter[i] < maxConsecutiveReplacements and misdetectionCounter < maxConsecutiveReplacements:
                                    windowPoses[-1][i] = windowPoses[-2][i]
                                    replacementCounter[i] += 1
                                    if verbose: print(f"Replaced joint {i} for {replacementCounter[i]} consecutive times.")
                                else:
                                    replacementCounter[i] = 0
                            else:
                                replacementCounter[i] = 0

                        # Update pose results on Last Frame for visualization:
                        posePredictions[0]['keypoints'] = windowPoses[-1]

                    poseVisLast = poseModel.visualize_pose_results(np.asarray(frame),
                                                            posePredictions,
                                                            jointThreshold,
                                                            4,
                                                            2
                                                            )
                    
                    if len(windowFrames) == 0:
                        windowFrames = [poseVisLast]*7
                    else:
                        windowFrames.pop(0)
                        windowFrames.append(poseVisLast)

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

                except Exception as e:
                    print(f"Unexpected {e}, {type(e)}")
                    
                    # Checking if the window for poses needs to be initialized:
                    if len(windowPoses) == 0:
                        windowPoses = [np.zeros((17,3))]*7
                        posePreds = [posePredictions]*7
                        windowFrames = [poseVisualization]*7
                    # If not, remove the first element from the window and add the current information at the end.
                    else:
                        windowPoses.pop(0)
                        windowPoses.append(np.zeros((17,3)))

                        posePreds.pop(0)
                        posePreds.append(posePredictions)

                        windowFrames.pop(0)
                        windowFrames.append(poseVisualization)

                    #windowPoses.append(np.zeros((17,3)))
                    #posePreds.append(posePredictions)

                # POSE AUGMENTATION.
                ## This is done in the middle frame, after gaussian smoothing.
                if posePreds[3] is not None:
                    augmentedPose = poseAug.augmentPose(copy.deepcopy(posePredictionsMiddle), jointStructure="body")
                    
                    augPose = []
                    for array in augmentedPose:
                        for subarray in array:
                            augPose.append(subarray)

                # HAND EXTRACTION.
                # This is done in the middle frame, after gaussian smoothing.
                    if handTracking == 'whole_image':
                        wholeImage = copy.deepcopy(rawWindowFrames[3])
                        wholeImage = cv2.cvtColor(wholeImage, cv2.COLOR_BGR2RGB)
                        resultsWholeImage = hands.process(wholeImage)
                        wholeImage = cv2.cvtColor(wholeImage, cv2.COLOR_RGB2BGR)

                        width_, height_, _ = np.shape(wholeImage)

                        leftHandDetection = False
                        rightHandDetection = False
                        leftHandCloseness = 1
                        rightHandCloseness = 1
                        
                        if resultsWholeImage.multi_hand_landmarks:
                            for hand_landmarks, handedness in zip(resultsWholeImage.multi_hand_landmarks, resultsWholeImage.multi_handedness):
                                # Note: Handedness is switched in MediaPipe.
                                xMin, xMax, yMin, yMax, xCenter, yCenter = ExtractCoordinatesFromLandmark(hand_landmarks, handTracking)

                                if xMax-xMin > yMax-yMin:
                                    yMin = yCenter - (xMax-xMin)/2
                                    yMax = yCenter + (xMax-xMin)/2
                                elif xMax-xMin < yMax-yMin:
                                    xMin = xCenter - (yMax-yMin)/2
                                    xMax = xCenter + (yMax-yMin)/2

                                # Check closeness  to previous left/right hand detections.
                                if len(leftHandCoordinates) > 0:
                                    xMin_, xMax_, yMin_, yMax_, xCenter_, yCenter_ = leftHandCoordinates[-1]
                                    leftHandCloseness = 1-(abs(xCenter_-xCenter))/xCenter_
                                if len(rightHandCoordinates) > 0:
                                    xMin_, xMax_, yMin_, yMax_, xCenter_, yCenter_ = rightHandCoordinates[-1]
                                    rightHandCloseness = 1-(abs(xCenter_-xCenter))/xCenter_

                                if handedness.classification[0].label == 'Right' and leftHandDetection == False and leftHandCloseness > 0.9:
                                    if len(leftHandCoordinates) == 0:
                                        leftHandCoordinates = [[xMin, xMax, yMin, yMax, xCenter, yCenter]] * 3
                                        leftHandMisdetections = 0
                                        leftHandLabel = handedness.classification[0].label
                                    else:
                                        leftHandCoordinates.pop(0)
                                        leftHandCoordinates.append([xMin, xMax, yMin, yMax, xCenter, yCenter])
                                        leftHandDetection = True
                                        leftHandMisdetections = 0
                                        leftHandLabel = handedness.classification[0].label

                                    if visualize:
                                        mp_drawing.draw_landmarks(wholeImage,
                                                                hand_landmarks,
                                                                mp_hands.HAND_CONNECTIONS,
                                                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                                                mp_drawing_styles.get_default_hand_connections_style())
                                        
                                    #wholeImage = cv2.rectangle(wholeImage,(int(xMin*height_),int(yMin*width_)),(int(xMax*height_),int(yMax*width_)),(255,255,255),1)
                                    
                                if handedness.classification[0].label == 'Left' and rightHandDetection == False and rightHandCloseness > 0.9:
                                    if len(rightHandCoordinates) == 0:
                                        rightHandCoordinates = [[xMin, xMax, yMin, yMax, xCenter, yCenter]] * 3
                                        rightHandMisdetections = 0
                                        rightHandLabel = handedness.classification[0].label
                                    else:
                                        rightHandCoordinates.pop(0)
                                        rightHandCoordinates.append([xMin, xMax, yMin, yMax, xCenter, yCenter])
                                        rightHandDetection = True
                                        rightHandMisdetections = 0
                                        rightHandLabel = handedness.classification[0].label

                                    if visualize:
                                        mp_drawing.draw_landmarks(wholeImage,
                                                                hand_landmarks,
                                                                mp_hands.HAND_CONNECTIONS,
                                                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                                                mp_drawing_styles.get_default_hand_connections_style())
                                
                                    
                            if rightHandDetection == False and rightHandMisdetections < 3 and len(rightHandCoordinates) > 0:
                                rightHandCoordinates.pop(0)
                                deltaHand = np.array(rightHandCoordinates[1])-np.array(rightHandCoordinates[0])
                                rightHandCoordinates.append(list(np.array(rightHandCoordinates[-1])+deltaHand))
                                rightHandMisdetections += 1
                                if verbose: print(f'Replaced right hand coordinates for {rightHandMisdetections} consecutive times:')
                            elif rightHandMisdetections >= 3:
                                rightHandCoordinates = []

                            if leftHandDetection == False and leftHandMisdetections < 3 and len(leftHandCoordinates) > 0:
                                leftHandCoordinates.pop(0)
                                deltaHand = np.array(leftHandCoordinates[1])-np.array(leftHandCoordinates[0])
                                leftHandCoordinates.append(list(np.array(leftHandCoordinates[-1])+deltaHand))
                                leftHandMisdetections += 1
                                if verbose: print(f'Replaced left hand coordinates for {leftHandMisdetections} consecutive times.')
                            elif leftHandMisdetections >= 3:
                                leftHandCoordinates = []

                            if len(rightHandCoordinates) > 0:
                                xMin, xMax, yMin, yMax, _, _ = rightHandCoordinates[-1]
                                wholeImage = cv2.rectangle(wholeImage,(int(xMin*height_),int(yMin*width_)),(int(xMax*height_),int(yMax*width_)),(255,255,255),1)
                                wholeImage = cv2.putText(wholeImage, rightHandLabel,(int(xMin*height_),int(yMin*width_)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2,cv2.LINE_AA)
                                curRightHand = copy.deepcopy(wholeImage[int(yMin*width_):int(yMax*width_),int(xMin*height_):int(xMax*height_)])

                            if len(leftHandCoordinates) > 0:
                                xMin, xMax, yMin, yMax, _, _ = leftHandCoordinates[-1]
                                wholeImage = cv2.rectangle(wholeImage,(int(xMin*height_),int(yMin*width_)),(int(xMax*height_),int(yMax*width_)),(255,255,255),1)
                                wholeImage = cv2.putText(wholeImage, leftHandLabel,(int(xMin*height_),int(yMin*width_)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2,cv2.LINE_AA)
                                curLeftHand = copy.deepcopy(wholeImage[int(yMin*width_):int(yMax*width_),int(xMin*height_):int(xMax*height_)])

                        if visualize:
                            cv2.imshow('Whole Image Hand Tracking', wholeImage)

                    elif handTracking == 'individual_image':

                        # DEPTH ESTIMATION.
                        # This is done only after the pose is augmented and when individual hand tracking is being performed.
                        estDepths = depthModel.forward(augPose)
                        neckDepth = estDepths[0].item()
                        nleftHandDepth = estDepths[1].item()
                        nrightHandDepth = estDepths[2].item()

                        if np.isnan(nleftHandDepth):
                            leftHandDepth = 100
                        else:
                            leftHandDepth = nleftHandDepth

                        if np.isnan(nrightHandDepth):
                            rightHandDepth = 100
                        else:
                            rightHandDepth = nrightHandDepth


                        leftHandLocations.pop(0)
                        leftHandCorners.pop(0)
                        rightHandLocations.pop(0)
                        rightHandCorners.pop(0)
                
                        handL = posePredictionsMiddle[0]['keypoints'][9][0:2]       # Coordinates from left wrist.
                        handR = posePredictionsMiddle[0]['keypoints'][10][0:2]      # Coordinates from right wrist.

                        resModifier = 300    ## NEEDS TO CHANGE FOR EACH VIDEO RESOLUTION
                        modL_ = int(resModifier*leftHandDepth/width)                         # Applying estimated depth modifier for left hand.
                        modR_ = int(resModifier*rightHandDepth/width)                        # Applying estimated depth modifier for right hand.

                        handL_ULx = int(handL[0])-modL_
                        handL_ULy = int(handL[1])-modL_
                        handR_ULx = int(handR[0])-modR_
                        handR_ULy = int(handR[1])-modR_

                        curLeftHandCorner = (handL_ULx,handL_ULy)
                        curRightHandCorner = (handR_ULx,handR_ULy)
                        handL_ = rawWindowFrames[3].copy()[max(int(handL[1])-modL_,0):int(handL[1])+modL_,max(int(handL[0])-modL_,0):int(handL[0])+modL_]
                        handR_ = rawWindowFrames[3].copy()[max(int(handR[1])-modR_,0):int(handR[1])+modR_,max(int(handR[0])-modR_,0):int(handR[0])+modR_]

                        cv2.imshow('Right Hand Raw',handR_)
                        cv2.imshow('Left Hand Raw',handL_)

                        # 'Smart' tracking of hands.

                        if leftHandLocations[-1] is not None:
                            prevXMin, prevXMax, prevYMin, prevYMax = leftHandLocations[-1]
                            prev_ULx, prev_ULy = leftHandCorners[-1]   # Corner of left hand frame in previous frame
                            #handL_ = rawWindowFrames[3].copy()[int(prev_ULy+prevYMin):int(prev_ULy+prevYMax),int(prev_ULx+prevXMin):int(prev_ULx+prevXMax)]
                            if np.array(handL_).shape[0] == 0 or np.array(handL_).shape[1] == 0:
                                if verbose: print(f"Left hand image collapsed while zooming on hand. Zooming out.")
                                handL_ = rawWindowFrames[3].copy()[int(handL[1])-modL_:int(handL[1])+modL_,int(handL[0])-modL_:int(handL[0])+modL_]
                            else:
                                curLeftHandCorner = (int(prev_ULx+prevXMin*0.2),int(prev_ULy+prevYMin*0.2))

                        if rightHandLocations[-1] is not None:
                            prevXMin, prevXMax, prevYMin, prevYMax = rightHandLocations[-1]
                            prev_ULx, prev_ULy = rightHandCorners[-1]
                            #handR_ = rawWindowFrames[3].copy()[int(prev_ULy+prevYMin):int(prev_ULy+prevYMax),int(prev_ULx+prevXMin):int(prev_ULx+prevXMax)]
                            if np.array(handR_).shape[0] == 0 or np.array(handR_).shape[1] == 0:
                                if verbose: print(f"Right hand image collapsed while zooming on hand. Zooming out.")
                                handR_ = rawWindowFrames[3].copy()[int(handR[1])-modR_:int(handR[1])+modR_,int(handR[0])-modR_:int(handR[0])+modR_]
                            else:
                                curRightHandCorner = (int(prev_ULx+prevXMin*0.2),int(prev_ULy+prevYMin*0.2))
                            
                        handL_MP = copy.deepcopy(handL_)
                        handR_MP = copy.deepcopy(handR_)
                        xHandSizeL = np.array(handL_MP).shape[0]
                        yHandSizeL = np.array(handL_MP).shape[1]
                        xHandSizeR = np.array(handR_MP).shape[1]
                        yHandSizeR = np.array(handR_MP).shape[0]
                        
                        ###
                        handL_ = cv2.cvtColor(handL_, cv2.COLOR_BGR2RGB)
                        handR_ = cv2.cvtColor(handR_, cv2.COLOR_BGR2RGB)

                        #image_bw = cv2.cvtColor(handL_, cv2.COLOR_BGR2GRAY)
                        #clahe = cv2.createCLAHE(clipLimit = 5)  ##
                        #final_img = clahe.apply(image_bw) + 30  ##
                        #print(np.shape(final_img))
                        #stacked_img = np.stack((final_img,)*3, axis=-1)
                        #print(np.shape(stacked_img))
                        #resultsL_ = hands.process(stacked_img)    ##
                        #handL_ = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

                        resultsL_ = hands.process(handL_)
                        resultsR_ = hands.process(handR_)

                        handL_ = cv2.cvtColor(handL_, cv2.COLOR_RGB2BGR)
                        handR_ = cv2.cvtColor(handR_, cv2.COLOR_RGB2BGR)

                        skipFrame = False

                        if resultsL_.multi_hand_landmarks:
                            leftHandGestureFrame = copy.deepcopy(handL_)
                            for hand_landmarks in resultsL_.multi_hand_landmarks:
                                xMinL, xMaxL, yMinL, yMaxL = ExtractCoordinatesFromLandmark(hand_landmarks, handTracking)
                                xMinL *= xHandSizeL
                                xMaxL *= xHandSizeL
                                yMinL *= yHandSizeL
                                yMaxL *= yHandSizeL

                                mp_drawing.draw_landmarks(handL_,
                                                        hand_landmarks,
                                                        mp_hands.HAND_CONNECTIONS,
                                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                                        mp_drawing_styles.get_default_hand_connections_style())
                            
                            print(xMinL, xMaxL, yMinL, yMaxL)
                            print(np.shape(leftHandGestureFrame))
                            xBound = np.shape(leftHandGestureFrame)[1]
                            yBound = np.shape(leftHandGestureFrame)[0]
                            print(max([int(yMinL),0]))
                            print(min([int(yMaxL),yBound]))
                            print(max([int(xMinL),0]))
                            print(min([int(xMaxL),xBound]))
                            
                            if max([int(yMinL),0]) >= min([int(yMaxL),yBound]) or max([int(xMinL),0]) >= min([int(xMaxL),xBound]):
                                leftHandLocations.append((xMinL*0.5,xMaxL*1.5,yMinL*0.5,yMaxL*1.5))
                            else:
                                leftHandGestureFrame = leftHandGestureFrame[max([int(yMinL),0]):min([int(yMaxL),yBound]),max([int(xMinL),0]):min([int(xMaxL),xBound])]
                                #leftHandGestureFrame = leftHandGestureFrame[max([int(xMinL),0]):min([int(xMaxL),xBound]),max([int(yMinL),0]):min([int(yMaxL),yBound])]
                                print(np.shape(leftHandGestureFrame))
                                cv2.imshow('Left Hand Gesture Frame',leftHandGestureFrame)
                                handL_ = cv2.rectangle(handL_,(int(xMinL),int(yMinL)),(int(xMaxL),int(yMaxL)),(255,0,0),1)
                                leftHandLocations.append((xMinL*0.5,xMaxL*1.5,yMinL*0.5,yMaxL*1.5)) # Increasing the size of the box around the hand coordinates.
                        else:
                            leftHandLocations.append(None)
                                
                        if resultsR_.multi_hand_landmarks:
                            rightHandGestureFrame = copy.deepcopy(handR_)
                            for hand_landmarks in resultsR_.multi_hand_landmarks:
                                xMinR, xMaxR, yMinR, yMaxR = ExtractCoordinatesFromLandmark(hand_landmarks, handTracking)
                                xMinR *= xHandSizeR
                                xMaxR *= xHandSizeR
                                yMinR *= yHandSizeR
                                yMaxR *= yHandSizeR

                                mp_drawing.draw_landmarks(handR_,
                                                        hand_landmarks,
                                                        mp_hands.HAND_CONNECTIONS,
                                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                                        mp_drawing_styles.get_default_hand_connections_style())
                            

                            rightHandGestureFrame = rightHandGestureFrame[max([int(yMinR),0]):int(yMaxR),max([int(xMinR),0]):int(xMaxR)]
                            #rightHandGestureFrame = rightHandGestureFrame[max([int(xMinR),0]):int(xMaxR),max([int(yMinR),0]):int(yMaxR)]
                            cv2.imshow('Right Hand Gesture Frame',rightHandGestureFrame)
                            handR_ = cv2.rectangle(handR_,(int(xMinR),int(yMinR)),(int(xMaxR),int(yMaxR)),(0,255,0),1)
                            rightHandLocations.append((xMinR*0.5,xMaxR*1.5,yMinR*0.5,yMaxR*1.5)) # Increasing the size of the box around the hand coordinates.
                        else:
                            rightHandLocations.append(None)
                        
                        leftHandCorners.append(curLeftHandCorner)
                        rightHandCorners.append(curRightHandCorner)

                        cv2.imshow('Left Hand',handL_)
                        cv2.imshow('Right Hand',handR_)
                        ###

                # STATIC HAND GESTURE RECOGNITION.

                #if leftHandGestureFrame is not None:
                if curLeftHand is not None:
                    staticGestureModel.LoadArray(curLeftHand)
                    curLeftHandEmbed = staticGestureModel.GetEmbedding()

                #if rightHandGestureFrame is not None:
                if curRightHand is not None:
                    staticGestureModel.LoadArray(curRightHand)
                    curRightHandEmbed = staticGestureModel.GetEmbedding()

                # POSE NORMALIZATION.
                # This is done only after the depth is estimated.
                if posePreds[3] is not None:

                    #normalizedPosePreds.pop(0)
                    #normalizedAugmentedPoses.pop(0)

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
                    #yScale = torsoHeight/(height/4)

                    #posePredictionsMiddle[0]['keypoints'] = (posePredictionsMiddle[0]['keypoints']-np.array([neckX-width/2,neckY-height/3,0]))#/[xScale, yScale, 1]

                    posePredictionsMiddle[0]['keypoints'] = ((posePredictionsMiddle[0]['keypoints']-np.array([neckX,neckY,0]))/[xScale, yScale, 1])+np.array([width/2,height/3,0])

                    NormalizedAugmentedPose = poseAug.augmentPose(copy.deepcopy(posePredictionsMiddle), jointStructure="body")
                    
                    normAugPose = []
                    for array in NormalizedAugmentedPose:
                        for subarray in array:
                            normAugPose.append(subarray)

                    if len(normalizedPosePreds) == 0:
                        normalizedPosePreds = [posePredictionsMiddle[0]['keypoints']]*7
                        normalizedAugmentedPoses = [normAugPose]*7
                    else:
                        normalizedPosePreds.pop(0)
                        normalizedPosePreds.append(posePredictionsMiddle[0]['keypoints'])

                        normalizedAugmentedPoses.pop(0)
                        normalizedAugmentedPoses.append(normAugPose)

                    bboxFrame = np.zeros((height,width,3),dtype=np.uint8)
                    posePredictionsMiddle[0].pop('bbox')
                    test = poseModel.visualize_pose_results(bboxFrame,
                                                            posePredictionsMiddle,
                                                            jointThreshold,
                                                            4,
                                                            2
                                                            )
                    if visualize:
                        cv2.imshow('Normalized Pose',cv2.resize(test,(int(width*0.5),int(height*0.5)),interpolation=cv2.INTER_AREA))
                
                # DYNAMIC POSE AUGMENTATION.
                # 
                if normalizedPosePreds[0] is not None:
                    curAugmentedPose = poseAug.augmentPoseDynamic(normalizedAugmentedPoses)    # vector size: 303
                    #curAugmentedPose = copy.deepcopy(normalizedAugmentedPoses[3])

                # EXTRACTING OUTPUTS
                if curLeftHandEmbed is None:
                    curLeftHandEmbed = torch.zeros(2048)
                if curRightHandEmbed is None:
                    curRightHandEmbed = torch.zeros(2048)
                if curAugmentedPose is None:
                    curAugmentedPose = torch.zeros(303)
                
                leftHandVector.append(curLeftHandEmbed)
                rightHandVector.append(curRightHandEmbed)
                augmentedPoseVector.append(curAugmentedPose)
                curRightHandEmbed = curRightHandEmbed.squeeze(0).numpy()
                curAugmentedPose = np.array(curAugmentedPose)
                curLeftHandEmbed = curLeftHandEmbed.squeeze(0).numpy()
                xDyn.append(np.concatenate((curRightHandEmbed,curAugmentedPose,curLeftHandEmbed)))

                # VISUALIZATION.
                if visualize:
                    row1 = np.concatenate(windowFrames[:3], axis=1)
                    row2 = np.concatenate([np.zeros((height,width,3),dtype=np.uint8),windowFrames[3],np.zeros((height,width,3),dtype=np.uint8)], axis=1)
                    row3 = np.concatenate(windowFrames[4:], axis=1)
                    completeDisplay = np.concatenate([row1, row2, row3], axis=0)
                    resizedScale = 0.8
                    resizedWidth = int(width*resizedScale)
                    resizedHeight = int(height*resizedScale)
                    resizedDisplay = cv2.resize(completeDisplay, (resizedWidth,resizedHeight), interpolation=cv2.INTER_AREA)
                    cv2.imshow('Frame',resizedDisplay)

                # OUTPUT.
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
    return (leftHandVector, rightHandVector, augmentedPoseVector, np.array(xDyn))

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

def ExtractCoordinatesFromLandmark(hand_landmarks, handTracking):
    xMax, xMin, yMax, yMin = None, None, None, None
    allCoordinatesX = []
    allCoordinatesY = []
    for coordinates in hand_landmarks.landmark:
        allCoordinatesX.append(coordinates.x)
        allCoordinatesY.append(coordinates.y)
        if xMax is None or xMax < coordinates.x:
            xMax = coordinates.x
        if xMin is None or xMin > coordinates.x:
            xMin = coordinates.x #if coordinates.x > 0 else xMin
        if yMax is None or yMax < coordinates.y:
            yMax = coordinates.y
        if yMin is None or yMin > coordinates.y:
            yMin = coordinates.y #if coordinates.y > 0 else yMin
    if handTracking=='whole_image':
        return xMin-0.02, xMax+0.02, yMin-0.02, yMax+0.02, np.mean(allCoordinatesX), np.mean(allCoordinatesY) # 2% padding
    if handTracking=='individual_image':
        return xMin-0.1, xMax+0.1, yMin-0.1, yMax+0.1   # 5% padding is added to the coordinates to avoid cropping the hand.

if __name__=="__main__":
    device = 'cuda'
    # Input and output paths.
    videoFile = "croppedVideos/ChaLearn04.mp4"
    outputFile = "croppedVideos/windowTests/ChaLearn04_psNorm.mp4"

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

    leftHandVector, rightHandVector, augmentedPoseVector, xDyn = GestureRecognition(videoFile, outputFile, detectionModel, poseModel, depthModel, staticGestureModel, mediaPipe, device=device)

    print('-----')
    print(len(leftHandVector), len(rightHandVector), len(augmentedPoseVector))
    print('-----')
    print(len(leftHandVector[-10][0]),leftHandVector)
    print('-----')
    print(len(rightHandVector[-10][0]),rightHandVector)
    print('-----')
    #print(augmentedPoseVector)