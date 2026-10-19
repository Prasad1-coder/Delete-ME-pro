import cv2

print("DeleteMe Pro is starting...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam open nahi ho raha!")
    exit()

print("Webcam successfully started!")
print("Camera band karne ke liye Q dabao.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Camera se frame nahi mil raha.")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("DeleteMe Pro - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("DeleteMe Pro closed.")