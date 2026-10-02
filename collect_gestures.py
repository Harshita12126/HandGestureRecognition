import cv2
import mediapipe as mp
import csv
import os

# -----------------------------
# SETTINGS
# -----------------------------

DATASET_FOLDER = "dataset"
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

# Create dataset folder
os.makedirs(DATASET_FOLDER, exist_ok=True)


# -----------------------------
# CREATE CSV FILE
# -----------------------------

if not os.path.exists(CSV_FILE):

    headers = []

    # 21 landmarks × x,y,z
    for i in range(21):
        headers.append(f"x{i}")
        headers.append(f"y{i}")
        headers.append(f"z{i}")

    headers.append("label")

    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    print("New dataset file created!")


# -----------------------------
# MEDIAPIPE
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# -----------------------------
# DATA COLLECTION
# -----------------------------

while True:

    print("\n==============================")
    print("Available Gestures:")
    print("==============================")

    for gesture in GESTURES:
        print("-", gesture)

    print("- DONE")

    label = input("\nEnter gesture name: ").strip().upper()

    # Exit
    if label == "DONE":
        break

    # Check valid gesture
    if label not in GESTURES:
        print("❌ Invalid gesture name!")
        continue

    print(f"\nStarting collection for: {label}")
    print("Show your hand clearly to the camera.")
    print("Press S = Save sample")
    print("Press Q = Stop this gesture\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera could not be opened!")
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
                print("❌ Could not read camera.")
                break

            # Flip camera
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands
            results = hands.process(rgb)

            landmarks_data = []

            if results.multi_hand_landmarks:

                # Take detected hands
                detected_hands = results.multi_hand_landmarks

                # For consistency, use maximum 2 hands
                for hand_landmarks in detected_hands[:2]:

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    for landmark in hand_landmarks.landmark:
                        landmarks_data.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

            # ------------------------------------------------
            # IMPORTANT
            # ------------------------------------------------
            # Our model will use 2 hands = 126 values.
            # If only one hand is detected, add zeros.
            # ------------------------------------------------

            if len(landmarks_data) == 63:
                landmarks_data.extend([0.0] * 63)

            # If no hand / invalid number
            if len(landmarks_data) != 126:
                landmarks_data = None

            # Display information
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
                "S = Save | Q = Next Gesture",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("Hand Gesture Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Save sample
            if key == ord("s"):

                if landmarks_data is not None:

                    with open(CSV_FILE, "a", newline="") as file:

                        writer = csv.writer(file)

                        writer.writerow(
                            landmarks_data + [label]
                        )

                    count += 1

                    print(
                        f"✅ Saved {label} sample: {count}"
                    )

                else:
                    print(
                        "⚠️ Hand not detected properly. "
                        "Try again."
                    )

            # Exit current gesture
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(
        f"\n✅ Finished collecting {label}: "
        f"{count} samples"
    )


print("\n================================")
print("🎉 DATA COLLECTION COMPLETED!")
print("================================")
print(f"Dataset saved at: {CSV_FILE}")