import cv2
import mediapipe as mp
import csv
import os
import math

# ==========================================
# SETTINGS
# ==========================================

DATASET_FOLDER = "dataset_v2"
CSV_FILE = os.path.join(DATASET_FOLDER, "gestures.csv")

GESTURES = [
    "HELLO",
    "THANK_YOU",
    "OK",
    "VICTORY",
    "YES",
    "NO",
    "PLEASE",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN"
]

os.makedirs(DATASET_FOLDER, exist_ok=True)


# ==========================================
# CSV HEADER
# ==========================================

if not os.path.exists(CSV_FILE):

    headers = []

    # LEFT HAND: 63 values
    for i in range(21):
        headers.extend([f"L_x{i}", f"L_y{i}", f"L_z{i}"])

    # RIGHT HAND: 63 values
    for i in range(21):
        headers.extend([f"R_x{i}", f"R_y{i}", f"R_z{i}"])

    headers.append("label")

    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    print("New V2 dataset created!")


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ==========================================
# NORMALIZE HAND LANDMARKS
# ==========================================

def normalize_hand(hand_landmarks):

    points = []

    wrist = hand_landmarks.landmark[0]

    # Make wrist the origin
    for landmark in hand_landmarks.landmark:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        points.append([x, y, z])

    # Find scale
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

    # Scale values
    normalized = []

    for point in points:

        normalized.extend([
            point[0] / max_distance,
            point[1] / max_distance,
            point[2] / max_distance
        ])

    return normalized


# ==========================================
# DATA COLLECTION
# ==========================================

while True:

    print("\n==============================")
    print("GESTURE DATA COLLECTION V2")
    print("==============================")

    for gesture in GESTURES:
        print("-", gesture)

    print("- DONE")

    label = input("\nEnter gesture name: ").strip().upper()

    if label == "DONE":
        break

    if label not in GESTURES:
        print("❌ Invalid gesture!")
        continue

    print("\n================================")
    print("Gesture:", label)
    print("================================")

    print("Show the correct hand position.")
    print("S = Save sample")
    print("Q = Next gesture")

    if label == "HELLO":
        print("\nHELLO pose:")
        print("Open palm + fingers clearly spread + hand slightly tilted.")

    elif label == "FIVE":
        print("\nFIVE pose:")
        print("Open palm + fingers straight and close together.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera could not open.")
        break

    count = 0

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

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            left_hand = None
            right_hand = None

            # ======================================
            # DETECT LEFT / RIGHT HAND
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

            landmarks_data = left_data + right_data

            # ======================================
            # DISPLAY
            # ======================================

            cv2.putText(
                frame,
                f"Gesture: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "S = Save | Q = Next",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Hand Gesture Collection V2",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # ======================================
            # SAVE
            # ======================================

            if key == ord("s"):

                if left_hand is not None or right_hand is not None:

                    with open(
                        CSV_FILE,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow(
                            landmarks_data + [label]
                        )

                    count += 1

                    print(
                        f"✅ {label}: {count} samples saved"
                    )

                else:

                    print(
                        "⚠️ No hand detected!"
                    )

            # ======================================
            # NEXT
            # ======================================

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(
        f"\nFinished {label}: {count} samples"
    )


print("\n================================")
print("🎉 V2 DATASET COLLECTION DONE")
print("================================")
print("Saved at:", CSV_FILE)