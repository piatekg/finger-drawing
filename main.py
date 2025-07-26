import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import uuid

model = tf.keras.models.load_model("custom_letters_scratch.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def is_thumbs_up(landmarks): 
    if landmarks.landmark[4].y < landmarks.landmark[3].y < landmarks.landmark[2].y:
        for i in [5,8,9,12,13,16,17,20]:
            if landmarks.landmark[4].y >= landmarks.landmark[i].y:
                return False
        return True
    return False

def is_open_palm(landmarks):
    fingers = [
        (8,6,5),
        (12,10,9),
        (16,14,13),
        (20,18,17)
    ]
    for tip, pip, mcp in fingers:
        if not (landmarks.landmark[tip].y < landmarks.landmark[pip].y < landmarks.landmark[mcp].y):
            return False
    return True

def is_pinky_extended(landmarks):
    return landmarks.landmark[20].y < landmarks.landmark[18].y < landmarks.landmark[17].y

prev_thumb_detected = False

drawing_points = []

letter_added = False
captured_text = ""

while True:
    success, img = cap.read()
    if not success: 
        print("Ignoring empty frame")
        continue

    #Mirror webcam feed to mimic straight on drawing
    img = cv2.flip(img, 1)

    #Convert image from OpenCV default convention to MediaPipe compatible (BGR -> RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    box_x1, box_y1 = 1240, 100
    box_x2, box_y2 = 1780, 580
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (200, 200, 200), 2)

    #Hand detection, list of hand objects
    if results.multi_hand_landmarks:
        left_hand = None
        right_hand = None

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_label = handedness.classification[0].label
            if hand_label == "Left":
                left_hand = hand_landmarks
            elif hand_label == "Right":
                right_hand = hand_landmarks

        if left_hand and is_thumbs_up(left_hand):
            thumb_detected = True
        else:
            thumb_detected = False
        
        if right_hand and thumb_detected:
            index_finger_tip = right_hand.landmark[8]
            h, w, _  = img.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            if box_x1 < cx < box_x2 and box_y1 < cy < box_y2:
                drawing_points.append((cx, cy))


        if thumb_detected and not prev_thumb_detected:
            print("Thumbs up")
        
        prev_thumb_detected = thumb_detected


        if thumb_detected: 
            cv2.putText(img, "Thumbs Up!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        if left_hand and is_open_palm(left_hand):
            drawing_points.clear()
        
        #Prepare capture
        if right_hand and is_pinky_extended(right_hand):
            pinky_detected = True

            if drawing_points:
                canvas = np.zeros((700, 1000), dtype=np.uint8)

                # Compute bounding box of points
                x_coords = [p[0] for p in drawing_points]
                y_coords = [p[1] for p in drawing_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Shift points so bounding box starts at (0,0)
                shifted_points = [(x - x_min, y - y_min) for x, y in drawing_points]

                # Draw shifted points on canvas
                for i in range(1, len(shifted_points)):
                    pt1 = shifted_points[i - 1]
                    pt2 = shifted_points[i]
                    cv2.line(canvas, pt1, pt2, (255), thickness=5)

                # Now crop with a margin around the shifted bounding box
                margin = 10
                crop_x_min = max(0, 0 - margin)
                crop_y_min = max(0, 0 - margin)
                crop_x_max = min(canvas.shape[1], (x_max - x_min) + margin)
                crop_y_max = min(canvas.shape[0], (y_max - y_min) + margin)

                cropped = canvas[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

                if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    print("Empty crop detected")
                else:
                    # Pad to square before resizing
                    h, w = cropped.shape
                    dim_diff = abs(h - w)
                    if h > w:
                        pad_left = dim_diff // 2
                        pad_right = dim_diff - pad_left
                        padded = cv2.copyMakeBorder(cropped, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                    else:
                        pad_top = dim_diff // 2
                        pad_bottom = dim_diff - pad_top
                        padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

                    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
                    cv2.imshow("resized", resized)
                    # Invert colors (white background, black ink)
                    inverted = cv2.bitwise_not(resized)

                    # Normalize and reshape
                    letter_img = inverted / 255.0
                    letter_img = letter_img.reshape(1, 28, 28, 1)

                    # Predict the letter
                    pred = model.predict(letter_img)
                    predicted_label = np.argmax(pred)  # EMNIST letters: labels 1–26

                    if not letter_added and pinky_detected:
                        predicted_char = chr(predicted_label + ord('a'))  # 1 → 'a', 2 → 'b', etc.
                        print("Predicted letter:", predicted_char.upper())
                        captured_text += predicted_char.upper()
                        letter_added = True

                    # Show the captured image for reference
                    cv2.imshow("Captured Drawing", inverted)
                    cv2.waitKey(1)

            else:
                print("No points to capture")
        else:
            letter_added = False

    for point in drawing_points:
        cv2.circle(img, point, 5, (255, 0 , 0), -1)
    
    cv2.putText(img, "Text: " + captured_text, (700,100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), 4)
    cv2.imshow("Hand Tracking", img)

    #Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()