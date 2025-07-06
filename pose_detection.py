import numpy as np

part_to_index = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
} 

def calculate_angle(a, b, c):
    # calcule angle between a, b, and c
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def is_knee_bent(landmarks, side='left'):
    # check if knees are bent below 180
    hip = landmarks.landmark[part_to_index[f'{side}_hip']]
    knee = landmarks.landmark[part_to_index[f'{side}_knee']]
    ankle = landmarks.landmark[part_to_index[f'{side}_ankle']]
    angle = calculate_angle(hip, knee, ankle)

    return angle < 170

def is_preparing(landmarks): 
    # check if user is preparing to shoot with some criteria

    # wrists closer together than shoulders
    left_wrist = landmarks.landmark[part_to_index['left_wrist']]
    right_wrist = landmarks.landmark[part_to_index['right_wrist']]
    wrists_distance = np.linalg.norm(np.array([left_wrist.x, left_wrist.y]) - np.array([right_wrist.x, right_wrist.y]))

    left_shoulder = landmarks.landmark[part_to_index['left_shoulder']]
    right_shoulder = landmarks.landmark[part_to_index['right_shoulder']]
    shoulders_distance = np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - np.array([right_shoulder.x, right_shoulder.y]))

    if wrists_distance > shoulders_distance:
        return False
    
    # elbows bent
    left_elbow = landmarks.landmark[part_to_index['left_elbow']]
    right_elbow = landmarks.landmark[part_to_index['right_elbow']]
    left_elbow_angle = calculate_angle(landmarks.landmark[part_to_index['left_shoulder']], left_elbow, landmarks.landmark[part_to_index['left_wrist']])
    right_elbow_angle = calculate_angle(landmarks.landmark[part_to_index['right_shoulder']], right_elbow, landmarks.landmark[part_to_index['right_wrist']])

    if left_elbow_angle > 150 and right_elbow_angle > 150:
        return False
    
    # elbows aren't above shoulders (is shooting covers this)
    if (left_elbow.y <= left_shoulder.y) or (right_elbow.y <= right_shoulder.y):
        return False
    
    # knees bent
    if not (is_knee_bent(landmarks, 'left') or is_knee_bent(landmarks, 'right')): 
        return False

    # at this point all criteria ar met
    return True

def is_shooting(landmarks): 
    # check for elbows above shoulders
    left_elbow = landmarks.landmark[part_to_index['left_elbow']]
    right_elbow = landmarks.landmark[part_to_index['right_elbow']]
    elbow_height = (left_elbow.y + right_elbow.y) / 2

    shoulder_height = (landmarks.landmark[part_to_index['left_shoulder']].y +
                       landmarks.landmark[part_to_index['right_shoulder']].y) / 2
    
    return elbow_height <= shoulder_height # keep in mind that y is inverted