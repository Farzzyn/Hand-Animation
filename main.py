import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerResult

# 1. Check or download model
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        MODEL_PATH
    )
    print("Model downloaded successfully!")

# Global callback variable
latest_result = None

def result_callback(result: HandLandmarkerResult, output_image, timestamp_ms):
    global latest_result
    latest_result = result

def get_tip(landmarks, tip_id, w, h):
    lm = landmarks[tip_id]
    return int(lm.x * w), int(lm.y * h)

def invert(img):
    inverted = cv2.bitwise_not(img)
    b, g, r = cv2.split(inverted)
    r_shift = 4
    b_shift = -4
    r_s = np.roll(r, r_shift, axis=1)
    b_s = np.roll(b, b_shift, axis=1)
    scanlines = np.zeros_like(inverted)
    scanlines[::3, :] = 255
    result = cv2.merge([b_s, g, r_s])
    result = cv2.add(result, scanlines)
    tint = np.array([0, 25, 15], dtype=np.uint8)
    return cv2.add(result, tint)

def red_grain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = np.full_like(img, 255)
    h, w = gray.shape
    shadow_mask = gray < 100
    grid_spacing = 3
    for y in range(0, h, grid_spacing):
        for x in range(0, w, grid_spacing):
            if shadow_mask[y, x]:
                out[y, x] = [0, 0, 200]
                if y + 1 < h and x + 1 < w:
                    out[y + 1, x] = [0, 0, 200]
                    out[y, x + 1] = [0, 0, 200]
    return out

def warp_rect_region(frame, src_pts):
    src = np.array(src_pts, dtype=np.float32)
    w, h = 200, 80
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (w, h))
    return warped, M, dst

def blend_rect_back(frame, warped, src_pts):
    src = np.array(src_pts, dtype=np.float32)
    w, h = warped.shape[1], warped.shape[0]
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(src_pts), 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    warped_back = cv2.warpPerspective(
        warped, M_inv, (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame = np.where(mask_3ch > 0, warped_back, frame)
    return frame

def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=result_callback
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (index 0). Please check your webcam.")
        sys.exit(1)

    window_name = 'Hand Effect'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    print("Camera ready ✅ (Press 'q' or ESC in the video window to quit)")

    timestamp = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detector.detect_async(mp_image, timestamp)
            timestamp += 1

            if latest_result and len(latest_result.hand_landmarks) == 2:
                h1 = latest_result.hand_landmarks[0]
                h2 = latest_result.hand_landmarks[1]

                thumb1 = get_tip(h1, 4, w, h)
                index1 = get_tip(h1, 8, w, h)
                middle1 = get_tip(h1, 12, w, h)

                thumb2 = get_tip(h2, 4, w, h)
                index2 = get_tip(h2, 8, w, h)
                middle2 = get_tip(h2, 12, w, h)

                rects = [
                    [thumb1, thumb2, index2, index1],
                    [index1, index2, middle2, middle1],
                ]

                effects = [invert, red_grain]

                for rect, effect in zip(rects, effects):
                    try:
                        warped, M, dst = warp_rect_region(frame, rect)
                        processed = effect(warped)
                        processed = cv2.GaussianBlur(processed, (3, 3), 0)
                        frame = blend_rect_back(frame, processed, rect)
                    except Exception:
                        pass

            try:
                rect_dim = cv2.getWindowImageRect(window_name)
                win_w, win_h = rect_dim[2], rect_dim[3]
                if win_w > 0 and win_h > 0:
                    frame_h, frame_w = frame.shape[:2]
                    scale = min(win_w / frame_w, win_h / frame_h)
                    new_w = int(frame_w * scale)
                    new_h = int(frame_h * scale)
                    resized = cv2.resize(frame, (new_w, new_h))
                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    x_off = (win_w - new_w) // 2
                    y_off = (win_h - new_h) // 2
                    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
                    frame = canvas
            except Exception:
                pass

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            # If window is closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == '__main__':
    main()
