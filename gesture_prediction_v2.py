import cv2
import mediapipe as mp
import joblib
import numpy as np
import math


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("gesture_model_v2.pkl")

print("✅ V2 Model loaded successfully!")


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ==========================================
# NORMALIZE HAND
# ==========================================

def normalize_hand(hand_landmarks):

    points = []

    wrist = hand_landmarks.landmark[0]

    for landmark in hand_landmarks.landmark:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        points.append([x, y, z])

    max_distance = 0

    for point in points:

        distance = math.sqrt(
            point[0] ** 2 +
            point[1] ** 2 +
            point[2] ** 2
        )

        max_distance = max(max_distance, distance)

    if max_distance == 0:
        max_distance = 1

    normalized = []

    for point in points:

        normalized.extend([
            point[0] / max_distance,
            point[1] / max_distance,
            point[2] / max_distance
        ])

    return normalized


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Camera could not open.")
    exit()


# ==========================================
# HAND DETECTION
# ==========================================

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        left_hand = None
        right_hand = None


        # ======================================
        # DETECT HANDS
        # ======================================

        if results.multi_hand_landmarks:

            for i, hand_landmarks in enumerate(
                results.multi_hand_landmarks
            ):

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                if results.multi_handedness:

                    hand_type = results.multi_handedness[
                        i
                    ].classification[0].label

                    if hand_type == "Left":

                        left_hand = hand_landmarks

                    elif hand_type == "Right":

                        right_hand = hand_landmarks


        # ======================================
        # CREATE 126 FEATURES
        # ======================================

        if left_hand is not None:

            left_data = normalize_hand(left_hand)

        else:

            left_data = [0.0] * 63


        if right_hand is not None:

            right_data = normalize_hand(right_hand)

        else:

            right_data = [0.0] * 63


        features = left_data + right_data

        features = np.array(features).reshape(1, -1)


        # ======================================
        # PREDICTION
        # ======================================

        if left_hand is not None or right_hand is not None:

            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]

            confidence = np.max(probabilities) * 100


            # ==================================
            # DISPLAY GESTURE
            # ==================================

            cv2.putText(
                frame,
                f"Gesture: {prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3
            )


            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Show your hand",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )


        # ======================================
        # INSTRUCTIONS
        # ======================================

        cv2.putText(
            frame,
            "Press Q to Quit",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ======================================
        # SHOW CAMERA
        # ======================================

        cv2.imshow(
            "Hand Gesture Recognition",
            frame
        )


        # ======================================
        # QUIT
        # ======================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ==========================================
# CLOSE
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Camera closed.")