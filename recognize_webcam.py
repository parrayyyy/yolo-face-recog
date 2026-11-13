
import cv2

from common import (
    detect_faces,
    extract_embedding,
    recognize_face
)


def recognize_webcam():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Could not access webcam.")
        return

    print("Webcam started!")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Detect faces using YOLO
        boxes = detect_faces(frame, image_size=640)

        for box in boxes:

            x1, y1, x2, y2 = box

            embedding = extract_embedding(frame, box)

            if embedding is None:

                name = "Unrecognized face"
                score = 0

            else:

                name, score = recognize_face(embedding)

            color = (
                (0, 255, 0)
                if name != "Unknown"
                else (0, 0, 255)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{name} ({score:.2f})"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("YOLO Live Face Recognition", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()

    cv2.destroyAllWindows()

    print("Webcam closed.")


if __name__ == "__main__":

    recognize_webcam()
