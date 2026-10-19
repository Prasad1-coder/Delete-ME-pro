
import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)


def distance(point1, point2):
    return math.hypot(
        point1.x - point2.x,
        point1.y - point2.y
    )


def detect_gesture(hand_landmarks):
    points = hand_landmarks.landmark

    # Fingertip and middle-joint landmark IDs
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    # Check whether each finger is extended
    fingers_up = []

    for tip, pip in zip(finger_tips, finger_pips):
        fingers_up.append(points[tip].y < points[pip].y)

    # Thumb check: compare tip distance from wrist
    thumb_up = (
        distance(points[4], points[0])
        > distance(points[3], points[0])
    )

    # Fist: all four fingers and thumb folded
    if not any(fingers_up) and not thumb_up:
        return "FIST - INVISIBILITY MODE"

    # Open palm: all fingers extended
    if all(fingers_up) and thumb_up:
        return "OPEN PALM - NORMAL MODE"

    # Peace sign: index and middle fingers extended
    if (
        fingers_up[0]
        and fingers_up[1]
        and not fingers_up[2]
        and not fingers_up[3]
    ):
        return "PEACE - GHOST MODE"

    return "GESTURE NOT RECOGNIZED"


if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit(1)

print("DeleteMe Pro - Gesture Detection Started")
print("Try a fist, open palm, and peace sign.")
print("Press Q to quit.")

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb_frame)
            gesture = "SHOW YOUR HAND"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                gesture = detect_gesture(hand)

            cv2.putText(
                frame,
                gesture,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "DeleteMe Pro - Gesture Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Gesture detection stopped.")

