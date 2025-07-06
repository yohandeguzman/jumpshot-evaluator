import numpy as np
from pose_detection import calculate_angle, part_to_index

def extract_shot_features(landmarks_sequence):
    left_elbow_angles, right_elbow_angles = [], []
    left_knee_angles, right_knee_angles = [], []

    joint_names = [
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'left_heel', 'right_heel'
    ]

    joint_coords = {name: {'x': [], 'y': [], 'z': []} for name in joint_names}

    for flat_frame in landmarks_sequence:
        coords = np.array(flat_frame).reshape((33, 3)) 

        class LM:
            pass
        lms = [LM() for _ in range(33)]
        for i, (x, y, z) in enumerate(coords):
            lms[i].x, lms[i].y, lms[i].z = x, y, z

        # angles
        left_elbow_angles.append(
            calculate_angle(lms[part_to_index['left_shoulder']], lms[part_to_index['left_elbow']], lms[part_to_index['left_wrist']])
        )
        right_elbow_angles.append(
            calculate_angle(lms[part_to_index['right_shoulder']], lms[part_to_index['right_elbow']], lms[part_to_index['right_wrist']])
        )
        left_knee_angles.append(
            calculate_angle(lms[part_to_index['left_hip']], lms[part_to_index['left_knee']], lms[part_to_index['left_ankle']])
        )
        right_knee_angles.append(
            calculate_angle(lms[part_to_index['right_hip']], lms[part_to_index['right_knee']], lms[part_to_index['right_ankle']])
        )

        # coords
        for name in joint_names:
            idx = part_to_index[name]
            joint_coords[name]['x'].append(lms[idx].x)
            joint_coords[name]['y'].append(lms[idx].y)
            joint_coords[name]['z'].append(lms[idx].z)

    features = []
    for angles in [left_elbow_angles, right_elbow_angles, left_knee_angles, right_knee_angles]:
        features.extend([np.mean(angles), np.min(angles), np.max(angles), np.std(angles)])

    for name in joint_names:
        for axis in ['x', 'y', 'z']:
            coords = joint_coords[name][axis]
            features.extend([np.mean(coords), np.min(coords), np.max(coords), np.std(coords)])

    return features
