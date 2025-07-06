import cv2
import mediapipe as mp
import numpy as np
import pickle

from pose_detection import is_preparing, is_shooting

from ml.feature_extraction import extract_shot_features
with open('ml/model.pkl', 'rb') as f: 
    clf = pickle.load(f)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# model testing variables
# cap = cv2.VideoCapture('data/videos/steph_curry_2.mp4')
# shot_counter = 9

shooting_buffer = []
lost_tracking_buffer = 0
was_preparaing = False
was_shooting = False

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            if is_preparing(results.pose_landmarks):
                shooting_buffer.append([
                    (lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark
                ])

                was_preparing = True
                cv2.putText(image, 'preparing to shoot', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_shooting(results.pose_landmarks):
                shooting_buffer.append([
                    (lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark
                ])

                # todo: do some logic about if preparing is currently false
                # if the user is shooting but didn't prepare, we can assume 
                # they didn't bend their knees properly
                was_preparing = False

                was_shooting = True
                cv2.putText(image, 'jump shot in motion', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # reset buffer
                if was_shooting and shooting_buffer:
                    features = extract_shot_features(shooting_buffer)
                    # path = f"data/csv_files/shot_{shot_counter}.csv"
                    # np.savetxt(path, np.array(shooting_buffer).reshape(len(shooting_buffer), -1), delimiter=',')
                    # shot_counter += 1
                    # print(f'saved {path}')
                    score = clf.predict([features])[0]
                    print(score)

                    # if score == 1: 
                    #     print('good form')
                    #     cv2.putText(image, 'good form', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # else: 
                    #     print('bad form')
                    #     cv2.putText(image, 'bad form', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                    shooting_buffer = []
                    was_preparaing = False
                    was_shooting = False

        else:
            cv2.putText(image, 'nobody in frame', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow('body tracker', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
