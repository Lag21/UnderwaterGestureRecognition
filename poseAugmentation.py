import copy
import numpy as np
from PIL import Image, ImageDraw

def augmentPoseDynamic(augmentedPoseWindow, jointStructure='body'):
    tempPose = copy.deepcopy(augmentedPoseWindow[3])
    for i, joint in enumerate(augmentedPoseWindow[3][:21]):
        velJoint = augmentedPoseWindow[4][i] - augmentedPoseWindow[2][i] # dP_k = P_(k+1) - P_(k-1)
        accJoint = augmentedPoseWindow[5][i] + augmentedPoseWindow[1][i] - 2*joint # d2P_k = P_(k+2) + P_(k-2) - 2P_k
        #augmentedPoseWindow[3].append(velJoint)
        #augmentedPoseWindow[3].append(accJoint)
        tempPose.append(velJoint)
        tempPose.append(accJoint)

    return tempPose


def augmentPose(pose_preds, jointStructure='body'):
    posePlus = []

    # Include estimated keypoints in the augmented pose vector
    if jointStructure=='body':
        pose_preds[0]['keypoints'] = pose_preds[0]['keypoints'][0:11]
        for keypoint in pose_preds[0]['keypoints']:
            posePlus.append(np.array([keypoint[0],keypoint[1]]))

        x = pose_preds[0]['keypoints'][:,0]
        y = pose_preds[0]['keypoints'][:,1]
    elif jointStructure=='hand':
        # If we're augmenting the hand, we remove the [0] since we're already iterating it outside.
        for keypoint in pose_preds['keypoints']:
            posePlus.append(np.array([keypoint[0],keypoint[1]]))
        
        x = pose_preds['keypoints'][:,0]
        y = pose_preds['keypoints'][:,1]

    a,b = calcBestFit(x,y)

    # Include the parameters of  the line of best fit in the augmented pose vector
    posePlus.append(np.array([a,b]))

    # Calculate all vectors between joints
    allVectors = calcVectors(x,y)

    # Calculate angles between connected joints
    connectedAngles = calcAnglesConnectedJoints(x,y,jointStructure)

    # Calculate angles between vectors and line of best fit
    bestFitAngles = calcAnglesLineBestFit(allVectors,a,b)

    # Include the vectors between all joints joints in the augmented pose vector
    for vec in allVectors:
        posePlus.append(np.array([vec[0],vec[1]]))
    # Include the length of all vectors in the augmented pose vector
    for vec in allVectors:
        posePlus.append(np.array([np.linalg.norm(vec)]))
    # Include the angles between the connected joints in the augmented pose vector
    for ang in connectedAngles:
        #print(ang)
        posePlus.append(np.array([ang]))
    #print("-----")
    for ang in bestFitAngles:
        #print(ang)
        posePlus.append(np.array([ang]))
    return posePlus

def calcBestFit(x,y):
    a, b = np.polyfit(y,x,1)
    return a, b

def drawBestFit(im, a, b, distratio=0.1):
    draw = ImageDraw.Draw(im)
    y1 = im.height*(0.5-distratio)
    y2 = im.height*(0.5+distratio)
    x1 = a*y1+b
    x2 = a*y2+b
    draw.line((x1,y1,x2,y2), fill='blue', width=3)
    return im

def calcVectors(x,y):
    connectedVectors = []

    for i in range(len(x)):
        for j in range(len(x)):
            if i < j:
                connectedVectors.append([x[i]-x[j],y[i]-y[j]])

    return connectedVectors

def calcAnglesConnectedJoints(x,y,jointStructure):
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

def calcAnglesLineBestFit(vectors,a,b):
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

def drawVectors(im,pose_preds,objectNumber=0):
    x = pose_preds[objectNumber]['keypoints'][:,0]
    y = pose_preds[objectNumber]['keypoints'][:,1]
    draw = ImageDraw.Draw(im)
    for i in range(len(x)):
        for j in range(len(x)):
            if i < j:
                draw.line((x[i],y[i],x[j],y[j]), fill='red')
    return im

def getConfKeypoints(pose_preds):
    print(f'Before removing kpts: \n{pose_preds}')
    acceptedKeypoints = []
    for i, keypoint in enumerate(pose_preds[0]['keypoints']):
        if keypoint[2] >= 11:
            acceptedKeypoints.append(keypoint)
    pose_preds[0]['keypoints'] = np.array(acceptedKeypoints)
    keypointNumber = len(acceptedKeypoints)
    print(f'After removing kpts: \n{pose_preds}')
    return pose_preds, keypointNumber