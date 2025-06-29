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

def is_shooting(landmarks): 
    # check for jump shot position
    left_elbow = landmarks.landmark[part_to_index['left_elbow']]
    right_elbow = landmarks.landmark[part_to_index['right_elbow']]
    elbow_height = (left_elbow.y + right_elbow.y) / 2

    shoulder_height = (landmarks.landmark[part_to_index['left_shoulder']].y +
                       landmarks.landmark[part_to_index['right_shoulder']].y) / 2
    
    return elbow_height < shoulder_height # keep in mind that y is inverted