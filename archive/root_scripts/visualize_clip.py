import os
import sys
import json
import argparse
import math
import numpy as np
import cv2

# Try importing MediaPipe for mouth box detection
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# Constants from the project configuration
FRAME_HEIGHT = 80
FRAME_WIDTH = 120
MAX_FRAMES = 150
LIP_PAD_X = 0.25
LIP_PAD_Y = 0.35

LIP_LANDMARK_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    95, 88, 178, 87, 14, 317, 402, 318, 324
]

def distance2d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_mouth_openness(landmarks):
    # vertical distance (13 to 14) divided by horizontal distance (78 to 308)
    v_dist = distance2d(landmarks[13], landmarks[14])
    h_dist = distance2d(landmarks[78], landmarks[308])
    return v_dist / max(1e-5, h_dist)

def get_lip_bbox_coords(landmarks, w, h):
    xs = [landmarks[i].x * w for i in LIP_LANDMARK_INDICES]
    ys = [landmarks[i].y * h for i in LIP_LANDMARK_INDICES]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    lip_w = x_max - x_min
    lip_h = y_max - y_min
    x_start = max(0, int(x_min - lip_w * LIP_PAD_X))
    x_end   = min(w, int(x_max + lip_w * LIP_PAD_X))
    y_start = max(0, int(y_min - lip_h * LIP_PAD_Y))
    y_end   = min(h, int(y_max + lip_h * LIP_PAD_Y))
    return x_start, y_start, x_end, y_end

def calculate_active_speaker_bbox(video_path):
    """Detects the mouth bounding box in the raw video using MediaPipe (identical to preprocessor)."""
    if not HAS_MEDIAPIPE:
        return None
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        return None
        
    h, w = frames[0].shape[:2]
    
    # Check model path
    model_path = os.path.join("ImConvo", "reports", "face_landmarker.task")
    if not os.path.exists(model_path):
        model_path = os.path.join("reports", "face_landmarker.task")
        if not os.path.exists(model_path):
            print("[MediaPipe] Model task file not found. Bounding box overlay will be skipped.")
            return None
        
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=2,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
    )
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        left_openings = []
        right_openings = []
        left_bboxes = []
        right_bboxes = []
        
        # Sample frames for active speaker decision (check up to 15 frames)
        sample_interval = max(1, len(frames) // 15)
        for i in range(0, len(frames), sample_interval):
            rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if result.face_landmarks:
                for face_lms in result.face_landmarks:
                    xs = [lm.x * w for lm in face_lms]
                    center_x = sum(xs) / len(xs)
                    openness = calculate_mouth_openness(face_lms)
                    bbox = get_lip_bbox_coords(face_lms, w, h)
                    
                    if center_x < w / 2:
                        left_openings.append(openness)
                        left_bboxes.append(bbox)
                    else:
                        right_openings.append(openness)
                        right_bboxes.append(bbox)
                        
        left_var = np.std(left_openings) if len(left_openings) > 2 else 0.0
        right_var = np.std(right_openings) if len(right_openings) > 2 else 0.0
        has_left = len(left_bboxes) > 0
        has_right = len(right_bboxes) > 0
        
        selected_speaker = "none"
        if has_left and has_right:
            if left_var > right_var:
                selected_speaker = "left"
            else:
                selected_speaker = "right"
        elif has_left:
            selected_speaker = "left"
        elif has_right:
            selected_speaker = "right"
            
        print(f"  Detected speaker: '{selected_speaker}' (Left var: {left_var:.4f}, Right var: {right_var:.4f})")
        
        # Collect bboxes of the selected speaker across all frames
        target_bboxes = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            
            found_bbox = None
            if result.face_landmarks:
                for face_lms in result.face_landmarks:
                    xs = [lm.x * w for lm in face_lms]
                    center_x = sum(xs) / len(xs)
                    is_left_face = center_x < w / 2
                    
                    if (selected_speaker == "left" and is_left_face) or (selected_speaker == "right" and not is_left_face):
                        found_bbox = get_lip_bbox_coords(face_lms, w, h)
                        break
            target_bboxes.append(found_bbox)
            
        valid_bboxes = [b for b in target_bboxes if b is not None]
        if not valid_bboxes:
            print(f"  [WARN] No faces matched selected speaker '{selected_speaker}'. Using center fallback.")
            return int(w*0.35), int(h*0.45), int(w*0.65), int(h*0.65)
        else:
            x_start = max(0, int(np.median([b[0] for b in valid_bboxes])))
            y_start = max(0, int(np.median([b[1] for b in valid_bboxes])))
            x_end   = min(w, int(np.median([b[2] for b in valid_bboxes])))
            y_end   = min(h, int(np.median([b[3] for b in valid_bboxes])))
            return x_start, y_start, x_end, y_end

def draw_wrapped_text(img, text, x, y, max_w, font, scale, color, thickness, line_spacing=25):
    """Draws text wrapped within max_w width on the OpenCV image."""
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        if size[0] > max_w:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))
        
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_spacing), font, scale, color, thickness)
    return len(lines) * line_spacing

def main():
    parser = argparse.ArgumentParser(description="Create a storyboard image showing raw frames vs preprocessed mouth crops")
    parser.add_argument("--clip", required=True, help="Clip base name, e.g. 'vid001_sent073'")
    parser.add_argument("--raw_dir", "--raw-dir", dest="raw_dir", default="my_lipreading_dataset/sentence_level", help="Path to raw videos folder")
    parser.add_argument("--preprocessed_dir", "--preprocessed-dir", dest="preprocessed_dir", default="ImConvo/data/preprocessed", help="Path to preprocessed .npy folder")
    parser.add_argument("--output_img", "--output-img", dest="output_img", default=None, help="Name of output image file (default: visualize_{clip}.png)")
    parser.add_argument("--num_frames", "--num-frames", dest="num_frames", type=int, default=10, help="Number of frames to display in the sequence strip")
    args = parser.parse_args()

    clip_name = os.path.splitext(args.clip)[0]
    
    video_path = os.path.join(args.raw_dir, f"{clip_name}.mp4")
    npy_path = os.path.join(args.preprocessed_dir, f"{clip_name}.npy")
    txt_path = os.path.join(args.raw_dir, f"{clip_name}.txt")
    map_path = "dataset_transcript_map.json"
    
    if args.output_img is None:
        args.output_img = f"visualize_{clip_name}.png"

    # 1. Validate files
    if not os.path.exists(video_path):
        print(f"Error: Raw video not found at {video_path}")
        return
        
    if not os.path.exists(npy_path):
        print(f"Error: Preprocessed .npy not found at {npy_path}")
        return

    # 2. Get Transcription Info
    text = ""
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            t_map = json.load(f)
            text = t_map.get(clip_name, "")
            
    char_len = len([c for c in text.lower() if c == " " or c.isalnum()])
    words = text.split()
    word_count = len(words)

    # 3. Load files
    print(f"Loading files for {clip_name}...")
    npy_data = np.load(npy_path)
    if len(npy_data.shape) == 4:
        npy_data = np.squeeze(npy_data, axis=-1)
    
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    
    if not raw_frames:
        print("Error: Could not read raw video frames.")
        return

    # Resample raw frames to 25 FPS exactly like the preprocessor does
    if abs(source_fps - 25.0) > 0.5:
        target_count = max(1, int(round(len(raw_frames) * 25.0 / source_fps)))
        resample_indices = np.linspace(0, len(raw_frames) - 1, target_count).round().astype(int)
        raw_frames_resampled = [raw_frames[i] for i in resample_indices]
    else:
        raw_frames_resampled = list(raw_frames)
        
    T_actual = len(raw_frames_resampled)
    
    # Pad resampled raw frames with black frames to MAX_FRAMES (150)
    if T_actual < MAX_FRAMES:
        black_frame = np.zeros_like(raw_frames_resampled[0])
        raw_frames_resampled += [black_frame] * (MAX_FRAMES - T_actual)
    elif T_actual > MAX_FRAMES:
        resample_indices_2 = np.linspace(0, T_actual - 1, MAX_FRAMES).round().astype(int)
        raw_frames_resampled = [raw_frames_resampled[i] for i in resample_indices_2]

    # 4. Calculate crop box overlay (using precise speaker tracking)
    print("Detecting active speaker mouth box for overlay...")
    bbox = calculate_active_speaker_bbox(video_path)
    
    # 5. Build storyboard arrays
    num_cols = args.num_frames
    indices = np.linspace(0, MAX_FRAMES - 1, num_cols).round().astype(int)
    
    # Display dimensions per sub-frame in the strip
    target_w = 160
    target_h = 120
    
    row1_list = []
    row2_list = []
    
    for idx in indices:
        # Get raw frame and draw bounding box
        raw_frame = raw_frames_resampled[idx].copy()
        # Draw bounding box only if it's not a padded black frame
        if bbox is not None and idx < T_actual:
            cv2.rectangle(raw_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        raw_resized = cv2.resize(raw_frame, (target_w, target_h))
        row1_list.append(raw_resized)
        
        # Get mouth crop frame
        crop_frame = npy_data[idx]
        if crop_frame.dtype != np.uint8:
            crop_frame = (crop_frame * 255).astype(np.uint8)
        crop_colored = cv2.cvtColor(crop_frame, cv2.COLOR_GRAY2BGR)
        crop_resized = cv2.resize(crop_colored, (target_w, target_h))
        row2_list.append(crop_resized)
        
    # Stack columns horizontally
    row1 = np.hstack(row1_list) # Original video frames row
    row2 = np.hstack(row2_list) # Mouth crop frames row
    
    # 6. Create Metadata and Labels Panel
    panel_w = target_w * num_cols
    panel_h = 120
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    # Draw frame index columns labels
    for i, idx in enumerate(indices):
        col_center_x = (i * target_w) + (target_w // 2)
        label_str = f"F:{idx}"
        text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        text_x = col_center_x - (text_size[0] // 2)
        cv2.putText(panel, label_str, (text_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
    # Draw text separator line
    cv2.line(panel, (15, 35), (panel_w - 15, 35), (100, 100, 100), 1)
    
    # Draw metadata and transcription
    draw_wrapped_text(
        panel, 
        f"GT: \"{text}\"", 
        x=20, 
        y=60, 
        max_w=panel_w - 40, 
        font=cv2.FONT_HERSHEY_SIMPLEX, 
        scale=0.5, 
        color=(100, 255, 100), 
        thickness=1
    )
    
    cv2.putText(
        panel, 
        f"Expected Character Tokens: {char_len} | Words: {word_count} | Total Frames: {len(raw_frames)} (resampled to {MAX_FRAMES})",
        (20, 105), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.4, 
        (255, 150, 100), 
        1
    )
    
    # Stack rows vertically
    storyboard = np.vstack([row1, row2, panel])
    
    # Save the output image
    cv2.imwrite(args.output_img, storyboard)
    print(f"\n✓ Storyboard generated successfully and saved to: {args.output_img}")

if __name__ == "__main__":
    main()
