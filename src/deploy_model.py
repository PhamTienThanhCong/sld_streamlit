# This file contains the existing sign language recognition model code. It handles video capture, feature extraction, and model prediction. This file may be imported into app.py to utilize the model's functionality.

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import deque
import pandas as pd
from math import sqrt

THRESHOLD = 0.8
CAM_IDX = 0  # Change if connecting to a different webcam

# Initialize MediaPipe Hands and Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

# Load labels from metadata.csv
METADATA_PATH = './data/metadata.csv'
metadata = pd.read_csv(METADATA_PATH)
labels = metadata['label']
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Load model
model = tf.keras.models.load_model('./data/SLR_model_words.h5', compile=False)

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, frame_rgb

def normalize_landmarks(landmarks, frame_shape):
    if landmarks.sum() == 0:
        return landmarks
    landmarks = landmarks.reshape(-1, 3)
    x_coords, y_coords, z_coords = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    width, height = frame_shape[1], frame_shape[0]
    if x_max - x_min > 0 and y_max - y_min > 0:
        x_normalized = (x_coords - x_min) / (x_max - x_min)
        y_normalized = (y_coords - y_min) / (y_max - y_min)
    else:
        x_normalized = x_coords / width
        y_normalized = y_coords / height
    return np.concatenate([x_normalized, y_normalized, z_coords]).flatten()

def compute_hand_movement_distance(hand_landmarks, prev_center):
    if hand_landmarks.sum() == 0 or prev_center is None:
        return 0.0
    hand_landmarks = hand_landmarks.reshape(-1, 3)
    x_mean, y_mean = np.mean(hand_landmarks[:, 0]), np.mean(hand_landmarks[:, 1])
    distance = sqrt((x_mean - prev_center[0])**2 + (y_mean - prev_center[1])**2)
    normalized_distance = min(distance * 10, 1.0)
    return normalized_distance

def compute_hand_to_shoulder_distances(hand_landmarks, shoulder_left, shoulder_right, frame_shape):
    if hand_landmarks.sum() == 0:
        return 0.0, 0.0
    hand_landmarks = hand_landmarks.reshape(-1, 3)
    width, height = frame_shape[1], frame_shape[0]
    
    x_mean, y_mean = np.mean(hand_landmarks[:, 0]), np.mean(hand_landmarks[:, 1])
    center_absolute = (int(x_mean * width), int(y_mean * height))
    
    shoulder_left_absolute = (int(shoulder_left[0] * width), int(shoulder_left[1] * height)) if shoulder_left is not None else None
    shoulder_right_absolute = (int(shoulder_right[0] * width), int(shoulder_right[1] * height)) if shoulder_right is not None else None
    
    dist_to_left_raw = 0.0
    dist_to_right_raw = 0.0
    
    if shoulder_left_absolute:
        dist_to_left_raw = sqrt((center_absolute[0] - shoulder_left_absolute[0])**2 + 
                                (center_absolute[1] - shoulder_left_absolute[1])**2)
    if shoulder_right_absolute:
        dist_to_right_raw = sqrt((center_absolute[0] - shoulder_right_absolute[0])**2 + 
                                 (center_absolute[1] - shoulder_right_absolute[1])**2)
    
    shoulder_distance = 0.0
    if shoulder_left_absolute and shoulder_right_absolute:
        shoulder_distance = sqrt((shoulder_left_absolute[0] - shoulder_right_absolute[0])**2 + 
                                 (shoulder_left_absolute[1] - shoulder_right_absolute[1])**2)
    
    diagonal = sqrt(width**2 + height**2)
    normalization_factor = shoulder_distance if shoulder_distance > 0 else diagonal
    
    dist_to_left = dist_to_left_raw / normalization_factor if normalization_factor > 0 else 0.0
    dist_to_right = dist_to_right_raw / normalization_factor if normalization_factor > 0 else 0.0
    
    return dist_to_left, dist_to_right

def extract_frame_features(frame, prev_right, prev_left, prev_right_center, prev_left_center, 
                         prev_right_shoulder_dists, prev_left_shoulder_dists, prev_shoulder_left, prev_shoulder_right):
    processed_frame, frame_rgb = preprocess_frame(frame)
    frame_shape = frame.shape
    
    hand_results = hands.process(frame_rgb)
    right_hand, left_hand = None, None
    right_score, left_score = 0, 0
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            score = handedness.classification[0].score
            label_hand = handedness.classification[0].label
            if label_hand == 'Right' and (right_hand is None or score > right_score):
                right_hand = hand
                right_score = score
            elif label_hand == 'Left' and (left_hand is None or score > left_score):
                left_hand = hand
                left_score = score
    
    pose_results = pose.process(frame_rgb)
    shoulder_left, shoulder_right = None, None
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
    else:
        shoulder_left = prev_shoulder_left
        shoulder_right = prev_shoulder_right
    
    right_features = np.zeros(21 * 3) if not right_hand else np.array([
        coord for lm in right_hand.landmark for coord in [lm.x, lm.y, lm.z]
    ])
    right_features_normalized = normalize_landmarks(right_features, frame_shape)
    right_movement_distance = compute_hand_movement_distance(right_features, prev_right_center)
    right_shoulder_dists = compute_hand_to_shoulder_distances(right_features, shoulder_left, shoulder_right, frame_shape)
    right_detected = right_features.sum() != 0
    if right_detected:
        right_landmarks = right_features.reshape(-1, 3)
        prev_right_center = (np.mean(right_landmarks[:, 0]), np.mean(right_landmarks[:, 1]))
        prev_right = right_features
        prev_right_shoulder_dists = right_shoulder_dists
    
    left_features = np.zeros(21 * 3) if not left_hand else np.array([
        coord for lm in left_hand.landmark for coord in [lm.x, lm.y, lm.z]
    ])
    left_features_normalized = normalize_landmarks(left_features, frame_shape)
    left_movement_distance = compute_hand_movement_distance(left_features, prev_left_center)
    left_shoulder_dists = compute_hand_to_shoulder_distances(left_features, shoulder_left, shoulder_right, frame_shape)
    left_detected = left_features.sum() != 0
    if left_detected:
        left_landmarks = left_features.reshape(-1, 3)
        prev_left_center = (np.mean(left_landmarks[:, 0]), np.mean(left_landmarks[:, 1]))
        prev_left = left_features
        prev_left_shoulder_dists = left_shoulder_dists
    
    frame_features = np.concatenate([
        right_features_normalized, [right_movement_distance],
        [prev_right_shoulder_dists[0] if shoulder_left is None else right_shoulder_dists[0],
         prev_right_shoulder_dists[1] if shoulder_right is None else right_shoulder_dists[1]],
        left_features_normalized, [left_movement_distance],
        [prev_left_shoulder_dists[0] if shoulder_left is None else left_shoulder_dists[0],
         prev_left_shoulder_dists[1] if shoulder_right is None else left_shoulder_dists[1]]
    ])
    
    hand_detected = right_detected or left_detected
    return (frame_features, hand_detected, prev_right, prev_left, 
            prev_right_center, prev_left_center, right_shoulder_dists, left_shoulder_dists,
            shoulder_left, shoulder_right)