#!/usr/bin/env python

from __future__ import annotations

from PIL import Image, ImageDraw
import numpy as np
import argparse
from model import _DetModel, _PoseModel

from poseAugmentation import calcAnglesConnectedJoints, calcAnglesLineBestFit, calcBestFit, calcVectors, drawVectors, getConfKeypoints, drawBestFit, augmentPose

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
    parser.add_argument('--detector-name', type=str, default='YOLOX-tiny')
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

#def augmentPose(pose_preds, jointStructure='body'):
    posePlus = []

    # Include estimated keypoints in the augmented pose vector
    if jointStructure=='body':
        for keypoint in pose_preds[0]['keypoints']:
            posePlus.append([keypoint[0],keypoint[1]])

        x = pose_preds[0]['keypoints'][:,0]
        y = pose_preds[0]['keypoints'][:,1]
    elif jointStructure=='hand':
        # If we're augmenting the hand, we remove the [0] since we're already iterating it outside.
        for keypoint in pose_preds['keypoints']:
            posePlus.append([keypoint[0],keypoint[1]])
        
        x = pose_preds['keypoints'][:,0]
        y = pose_preds['keypoints'][:,1]

    a,b = calcBestFit(x,y)

    # Include the parameters of  the line of best fit in the augmented pose vector
    posePlus.append([a,b])

    # Calculate all vectors between joints
    allVectors = calcVectors(x,y)

    # Calculate angles between connected joints
    connectedAngles = calcAnglesConnectedJoints(x,y,jointStructure)

    # Calculate angles between vectors and line of best fit
    bestFitAngles = calcAnglesLineBestFit(allVectors,a,b)

    # Include the vectors between all joints joints in the augmented pose vector
    for vec in allVectors:
        posePlus.append([vec[0],vec[1]])
    # Include the length of all vectors in the augmented pose vector
    for vec in allVectors:
        posePlus.append(np.linalg.norm(vec))
    # Include the angles between the connected joints in the augmented pose vector
    for ang in connectedAngles:
        #print(ang)
        posePlus.append(ang)
    #print("-----")
    for ang in bestFitAngles:
        #print(ang)
        posePlus.append(ang)

    return posePlus

#def calcBestFit(x,y):
    a, b = np.polyfit(y,x,1)
    return a, b

#def drawBestFit(im, a, b, distratio=0.1):
    draw = ImageDraw.Draw(im)
    y1 = im.height*(0.5-distratio)
    y2 = im.height*(0.5+distratio)
    x1 = a*y1+b
    x2 = a*y2+b
    draw.line((x1,y1,x2,y2), fill='blue', width=3)
    return im

#def calcVectors(x,y):
    connectedVectors = []

    for i in range(len(x)):
        for j in range(len(x)):
            if i < j:
                connectedVectors.append([x[i]-x[j],y[i]-y[j]])

    return connectedVectors

#def calcAnglesConnectedJoints(x,y,jointStructure):
    if jointStructure=='body':
        connectedJoints={
            '0':[1,2],      # Nose
            '1':[2,3],      # Left Eye
            '2':[4],        # Right Eye
            '3':[5],        # Left Ear
            '4':[6],        # Right Ear
            '5':[6,7,11],   # Left Shoulder
            '6':[8,12],     # Right Shoulder
            '7':[9],        # Left Elbow
            '8':[10],       # Right Elbow
            '9':[],         # Left Hand
            '10':[],        # Right Hand
            '11':[13],      # Left Hip
            '12':[14],      # Right Hip
            '13':[15],      # Left Knee
            '14':[16],      # Right Knee
            '15':[],        # Left Foot
            '16':[]         # Right Foot
        }
    elif jointStructure=='hand':
        connectedJoints={
            '0':[1,5,9,13,17],      # Wrist
            '1':[2],                # Thumb 1st Joint
            '2':[3],                # Thumb 2nd Joint
            '3':[4],                # Thumb 3rd Joint
            '4':[],                 # Thumb End
            '5':[6],                # Index Finger 1st Joint
            '6':[7],                # Index Finger 2nd Joint
            '7':[8],                # Index Finger 3rd Joint
            '8':[],                 # Index Finger End
            '9':[10],               # Middle Finger 1st Joint
            '10':[11],              # Middle Finger 2nd Joint
            '11':[12],              # Middle Finger 3rd Joint
            '12':[],                # Middle Finger End
            '13':[14],              # Ring Finger 1st Joint
            '14':[15],              # Ring Finger 2nd Joint
            '15':[16],              # Ring Finger 3rd Joint
            '16':[],                # Ring Finger End
            '17':[18],              # Little Finger 1st Joint
            '18':[19],              # Little Finger 2nd Joint
            '19':[20],              # Little Finger 3rd Joint
            '20':[],                # Little Finger End
        }
        
    vecs = []
    nodes = []
    angles = []
    
    for i in range(len(x)):
        for j in connectedJoints[str(i)]:
            if j >= len(x):
                continue
            vec = [x[i]-x[j],y[i]-y[j]]
            vecs.append(vec)
            nodes.append([i,j])

    for i, (vec, node) in enumerate(zip(vecs,nodes)):
        #print(i,vec,node)
        for n in node:
            for j, nod in enumerate(nodes):
                if n in nod and node is not nod:
                    #print('->',nod)
                    unitV1 = vec / np.linalg.norm(vec)
                    unitV2 = vecs[j] / np.linalg.norm(vecs[j])
                    angle = np.arccos(np.dot(unitV1,unitV2))
                    if [angle,min(i,j),max(i,j)] in angles:
                        continue
                    angles.append([angle,min(i,j),max(i,j)])

    return np.array(angles)[:,0]

#def calcAnglesLineBestFit(vectors,a,b):
    y1 = 250*(0.5-0.1)
    y2 = 250*(0.5+0.1)
    x1 = a*y1+b
    x2 = a*y2+b

    lobfV = [x1-x2,y1-y2]
    lobfUnitV = lobfV / np.linalg.norm(lobfV)

    vecs = []
    nodes = []
    angles = []
    
    for vec in vectors:
        unitV2 = vec / np.linalg.norm(vec)
        angle = np.arccos(np.dot(lobfUnitV,unitV2))
        angles.append(angle)
    
    return np.array(angles)

#def drawVectors(im,pose_preds,objectNumber=0):
    x = pose_preds[objectNumber]['keypoints'][:,0]
    y = pose_preds[objectNumber]['keypoints'][:,1]
    draw = ImageDraw.Draw(im)
    for i in range(len(x)):
        for j in range(len(x)):
            if i < j:
                draw.line((x[i],y[i],x[j],y[j]), fill='red')
    return im

#def getConfKeypoints(pose_preds):
    global args
    print(f'Before removing kpts: \n{pose_preds}')
    acceptedKeypoints = []
    for i, keypoint in enumerate(pose_preds[0]['keypoints']):
        if keypoint[2] >= args.vis_kpt_score_threshold:
            acceptedKeypoints.append(keypoint)
    pose_preds[0]['keypoints'] = np.array(acceptedKeypoints)
    args.kpt_number = len(acceptedKeypoints)
    print(f'After removing kpts: \n{pose_preds}')
    return pose_preds

def main():
    global args

    np.set_printoptions(suppress=True)

    args = parse_args()

    det_model = _DetModel(device=args.device)
    pose_model = _PoseModel(device=args.device)
    raw_image = Image.open(args.image_filename)
    input_image = np.asarray(raw_image)

    det_model.set_model(args.detector_name)
    det_preds, detection_visualization = det_model.run(args.detector_name, input_image, args.vis_det_score_threshold)

    ### Human Detection has been performed at this point.

    bbres = Image.fromarray(detection_visualization)

    pose_preds, pose_visualization = pose_model.run(
        args.pose_model_name,
        input_image,
        det_preds,
        args.det_score_threshold,
        args.vis_kpt_score_threshold,
        args.vis_dot_radius,
        args.vis_line_thickness)

    poseres = Image.fromarray(pose_visualization)

    ### Human Pose Estimation has been performed at this point.
    if args.kpt_number < 17:
        print(args.kpt_number)
        pose_preds[0]['keypoints'] = pose_preds[0]['keypoints'][0:args.kpt_number]

    #pose_preds = getConfKeypoints(pose_preds)   # NEED TO FIX THIS, how to properly remove underconfident keypoints to perform everything else without issues.

    posePlus = augmentPose(pose_preds, jointStructure='body')

    print(f'The length of the augmented pose vector for the initial {args.kpt_number} keypoints is: {len(posePlus)}')

    # Flag to save images of processing steps.
    if(args.save_outputs):
        bbres.save('test_results/bb_result.jpg')
        poseres.save('test_results/pose_result.jpg')
        imBestFit = drawBestFit(poseres,posePlus[args.kpt_number][0],posePlus[args.kpt_number][1],0.3)
        imBestFit.save('test_results/bf_result.jpg')
        imAugmentedVectors = drawVectors(Image.open('test_results/pose_result.jpg'),pose_preds)
        imAugmentedVectors.save('test_results/av_result.jpg')

    # Testing Hand Detection Model

    handDet_model = _DetModel(device=args.device)
    handPose_model = _PoseModel(device=args.device)

    raw_image = Image.open(args.image_filename)
    input_image = np.asarray(raw_image)

    handDet_model.set_model('Hand')
    handDet_preds, handDetection_visualization = handDet_model.run('Hand', input_image, 0.3)

    handbbres = Image.fromarray(handDetection_visualization)

    handpose_preds, handpose_visualization = handPose_model.run(
        'Hand',
        input_image,
        handDet_preds,
        0.3,
        args.vis_kpt_score_threshold,
        args.vis_dot_radius,
        args.vis_line_thickness)
    
    print('------')
    for i, handpose in enumerate(handpose_preds):
        handPosePlus = augmentPose(handpose, jointStructure='hand')
        if(args.save_outputs):
            if i == 0:
                handposeres = Image.fromarray(handpose_visualization)
                handbbres.save('test_results/handbbA_result.jpg')
                handposeres.save('test_results/handposeA_result.jpg')
                imBestFit = drawBestFit(handposeres,handPosePlus[21][0],handPosePlus[21][1],0.3)
                imBestFit.save('test_results/handbfA_result.jpg')
                imAugmentedVectors = drawVectors(Image.open('test_results/handposeA_result.jpg'),handpose_preds,i)
                imAugmentedVectors.save('test_results/handavA_result.jpg')
            elif i == 1:
                handposeres = Image.fromarray(handpose_visualization)
                handbbres.save('test_results/handbbB_result.jpg')
                handposeres.save('test_results/handposeB_result.jpg')
                imBestFit = drawBestFit(handposeres,handPosePlus[21][0],handPosePlus[21][1],0.3)
                imBestFit.save('test_results/handbfB_result.jpg')
                imAugmentedVectors = drawVectors(Image.open('test_results/handposeB_result.jpg'),handpose_preds,i)
                imAugmentedVectors.save('test_results/handavB_result.jpg')
        print(f'The length of the augmented pose vector for the hand {i} is: {len(handPosePlus)}')
    print('------')

    return


if __name__ == '__main__':
    main()