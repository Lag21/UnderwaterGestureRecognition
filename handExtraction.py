from __future__ import annotations

from PIL import Image, ImageDraw
import numpy as np
import cv2

import argparse

from model import _DetModel, _PoseModel

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
    parser.add_argument('--detector-name', type=str, default='YOLOX-x')
    parser.add_argument('--pose-model-name', type=str, default='ViTPose-B*')

    ### Hand Detector and Model Parameters

    ### Bounding Box Visualization Parameters
    parser.add_argument('--vis-det-score-threshold', default=0.3, help='Bounding box score threshold to visualize.')

    ### Pose Visualization Parameters
    parser.add_argument('--det-score-threshold', default=0.5, help='Bounding box score threshold to estimate pose.')
    parser.add_argument('--vis-kpt-score-threshold', default=0.7)
    parser.add_argument('--vis-dot-radius', default=4)
    parser.add_argument('--vis-line-thickness', default=2)

    ### Pose Augmentation Parameters
    parser.add_argument('--kpt-number', type=int, default=11, help='17 will extract all the keypoints. 11 will extract the upper body.')

    return parser.parse_args()

def main():
    args = parse_args()
    det_model = _DetModel(device='cuda')
    det_model.set_model(args.detector_name)
    handDet_model = _DetModel(device='cuda')
    handDet_model.set_model('Hand')

    raw_image = Image.open(args.image_filename)
    input_image = np.asarray(raw_image)

    #handDet_preds, handDetection_visualization = handDet_model.run('Hand', input_image, 0.5)

    #handBoundingBoxes = []
    #for detectedHand in handDet_preds[0]:
    #    handBoundingBoxes.append(detectedHand[0:4])

    #leftHand = raw_image.crop([handBoundingBoxes[0][0]*0.98,handBoundingBoxes[0][1]*0.98,handBoundingBoxes[0][2]*1.02,handBoundingBoxes[0][3]*1.02])
    #rightHand = raw_image.crop([handBoundingBoxes[1][0]*0.98,handBoundingBoxes[1][1]*0.98,handBoundingBoxes[1][2]*1.02,handBoundingBoxes[1][3]*1.02])

    #leftHand.show()
    #rightHand.show()

    testVideoPath = 'test_videos/diver.mp4'
    cap = cv2.VideoCapture(testVideoPath)
    frameCounter = 0
    counter = 0

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    fps = 20
    output = cv2.VideoWriter('/home/alejandro/Documents/GitHub/UnderwaterGestureRecognition/test_videos/HandDetection_03.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    handDetected = False

    while(cap.isOpened()):
        frameCounter+=1
        ret, frame = cap.read()

        #if frameCounter < 510:
        #    pass
        #elif frameCounter > 1470:
        #    break
        #else:
        if ret:
            #cv2.imshow('Frame',frame)
            cv2.waitKey(1)
            #counter+=1
            #if counter > 10:
            #    break
            
            #det_preds, detection_visualization = det_model.run(args.detector_name, np.asarray(frame), args.vis_det_score_threshold)
            
            handDet_preds, handDetection_visualization = handDet_model.run('Hand', np.asarray(frame), args.vis_det_score_threshold)
            #print(handDet_preds)
            #print(len(handDet_preds[0]))
            #if len(handDet_preds[0]) > 0:
            #    if handDet_preds[0][0][4] > args.vis_det_score_threshold:
            #        handDetected = True
            #    else:
            #        handDetected = False
            
            #cv2.imshow('Frame',detection_visualization)
            output.write(handDetection_visualization)
        else:
            break


        #if frame is not None:
        #    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #    handDet_preds, handDetection_visualization = handDet_model.run('Hand', np.asarray(frame2), 0.5)
        #    print(f'------{counter}------')
        #    print(handDet_preds)
        #    #Image.fromarray(handDetection_visualization).show()
        #    handBoundingBoxes = []
        #    for detectedHand in handDet_preds[0]:
        #        handBoundingBoxes.append(detectedHand[0:4])

            
        #    #leftHand = raw_image.crop([handBoundingBoxes[0][0]*0.98,handBoundingBoxes[0][1]*0.98,handBoundingBoxes[0][2]*1.02,handBoundingBoxes[0][3]*1.02])
        #    #leftHand.show()
        #    counter += 1
        #    if counter > 20: break
        #else:
        #    counter += 1
        #    if counter > 20: break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()