import cv2
import glob
import numpy as np
from PIL import Image, ImageDraw
import poseAugmentation as poseAug
from model import _DetModel, _PoseModel
from tqdm import tqdm

## TO DO:
## 1. Add additional txt file to save the name of the processed files.
## 2. Modify code to start from a given .npy file.
## 3. Modify code to append new processed data to the loaded .npy file
## 4. Modify code to check if a file has already been processed and skip it. 

def main():
    visualize = False
    keypointNumber = 11     # To use upper body.

    datasetFolder = "DatasetSamples/"
    scenesFolder = datasetFolder+"RGB/"
    depthsFolder = datasetFolder+"Depth Map/"

    detectionModel = _DetModel(device="cuda")
    poseModel = _PoseModel(device="cuda")
    detectionModel.set_model("YOLOX-x")

    AVIs = sorted(glob.glob(scenesFolder+"/*.avi"))

    aPs = []
    eDs = []

    for AVI in tqdm(AVIs):

        filename = AVI.split('/')[2]
        filename = filename.split('_')[0]
        #print(f"Processing: {filename}")
        
        # Look up list of depth maps for a given video.
        depthMaps = sorted(glob.glob(depthsFolder+filename+"/*.png"))

        cap = cv2.VideoCapture(AVI)
        rgbWidth = int(cap.get(3))
        rgbHeight = int(cap.get(4))

        frameCounter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if ret:
                inputImage = np.asarray(frame)

                detectionPredictions, detectionVisualization = detectionModel.run("YOLOX-x", inputImage, 0.3) #(DetectionModel, Image, DetectionThreshold)
                posePredictions, poseVisualization = poseModel.run('ViTPose-B*',         #PoseModel
                                                                inputImage,              #Input
                                                                detectionPredictions,    #Detected Human Box
                                                                0.3,                     #Detection Threshold
                                                                0.7,                     #Keypoint Visualization Threshold
                                                                4,                       #Keypoint Radius
                                                                2)                       #Line Thickness
                
                try:
                    posePredictions[0]['keypoints'] = posePredictions[0]['keypoints'][0:keypointNumber]    # Cleaning up pose keypoints to exclude lower body.
                except:
                    print(f"Frame {frameCounter} in {filename} had issues with the pose detection.")
                    frameCounter+=1
                    continue
                augmentedPose = poseAug.augmentPose(posePredictions, jointStructure="body")

                # Extracting the coordinates of the middle point between both shoulders.
                # These coordinates will be used to collect the depth value from the depth map.

                rgbX = (posePredictions[0]['keypoints'][5][0]+posePredictions[0]['keypoints'][6][0])/2
                rgbY = (posePredictions[0]['keypoints'][5][1]+posePredictions[0]['keypoints'][6][1])/2
                depthImage = Image.open(depthMaps[frameCounter])
                depthWidth, depthHeight = depthImage.size

                ## Since the videos and the depth maps have different resolution, I need to scale the coordinates
                relativeX = rgbX/rgbWidth
                relativeY = rgbY/rgbHeight

                depthX = int(relativeX*depthWidth)
                depthY = int(relativeY*depthHeight)

                pixelValue = depthImage.getpixel((depthX,depthY))
                
                # Updating the frame counter.
                frameCounter+=1

                # Showing the images.
                if visualize:
                    poseVisualization = cv2.circle(poseVisualization.astype(np.int32), (int(rgbX), int(rgbY)), radius=10, color=(0,0,255), thickness=-1)
                    depthImagenp = np.asarray(depthImage)
                    depthImagenp = (depthImagenp/4500)*255
                    depthImagenp = depthImagenp.astype(np.uint8)
                    depthRGB = cv2.cvtColor(depthImagenp, cv2.COLOR_GRAY2BGR)
                    depthRGB = cv2.circle(depthRGB, (depthX,depthY), radius=5, color=(0,0,255), thickness=-1)
                    resizedPoseVisualization = cv2.resize(poseVisualization.astype(np.uint8), (512,424), interpolation=cv2.INTER_AREA)
                    horizontalDisplay = np.concatenate((resizedPoseVisualization,depthRGB), axis=1)
                    cv2.imshow('Frame',horizontalDisplay)

                # Clean data so it's easier to process by the neural network.
                augPose = []
                for array in augmentedPose:
                    for subarray in array:
                        augPose.append(subarray)

                # Quality checks for the augmented data:
                # 1. Check that the vector is of the expected length.
                if len(augPose) != 261:
                    print(f"Frame {frameCounter} in {filename} did not have the correct length for the augmented pose.")
                    continue
                # 2. Check if there are any NaNs in the data.
                if np.isnan(augPose).any():
                    print(f"Frame {frameCounter} in {filename} had a NaN value in the augmented pose.")
                    continue

                aPs.append(np.array(augPose))
                eDs.append(np.array(pixelValue))

                pressedKey = cv2.waitKey(1) & 0xFF
                if pressedKey == ord('q'):
                    break

            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    # Saving the augmented pose array and depth array as numpy files.

    aPs = np.array(aPs)
    eDs = np.array(eDs)

    np.save(datasetFolder+"aPs.npy",aPs)
    np.save(datasetFolder+"eDs.npy",eDs)

if __name__=="__main__":
    main()