def display_prediction(predictions):
    if predictions:
        return f"Predicted: {predictions[-1]}"
    return "No predictions yet."

def display_progress_bar(current_frames, total_frames=30):
    filled_width = int(200 * current_frames / total_frames)
    return filled_width

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