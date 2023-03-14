import cv2

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = 20

output = cv2.VideoWriter('/home/alejandro/Documents/GitHub/simpleViTPose/test_videos/SampleVideo.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

savingVideo = False

while(True):
    ret, frame = cap.read()
    cv2.imshow('Frame',frame)
    if savingVideo:
        output.write(frame)


    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord('a'):
        savingVideo = not savingVideo
        if savingVideo:
            print('Started recording.')
        else:
            print('Finished recording')


cap.release()
output.release()
cv2.destroyAllWindows()