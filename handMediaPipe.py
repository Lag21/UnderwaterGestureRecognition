import mediapipe as mp

# Initializing MediaPipe.
###
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initializing video input.
cap = cv2.VideoCapture(videoFile)
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

with mp_hands.Hands(model_complexity=1,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while(cap.isOpened()):
        ret, frame = cap.read()
###
