import os
import cv2
import pandas as pd
from glob import glob

print(os.getcwd())
os.chdir('../../Datasets/ChaLearnLSSI')
print(os.getcwd())

data = pd.read_csv('train_labels.csv',header=None)

counter = 0
for idx, row in data.iterrows():
    file = row[0]
    gestureLabel = row[1]
    print(row[0],row[1])
    counter+=1
    if counter >= 3:
        break

    pathfile = 'train/'+file+'_color.mp4'
    print(pathfile)

    cap = cv2.VideoCapture(pathfile)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            cv2.imshow('Video',frame)
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()