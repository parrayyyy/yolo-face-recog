
import cv2
import time

from common import face_app, load_database, recognize_face


def recognize_webcam():

    # Load database once
    database = load_database()

    if not database:
        print("No registered people found!")
        return

    # Open webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Could not access webcam.")
        return

    # Camera settings
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Live recognition started!")
    print("Press Q to exit.")

    frame_count = 0

    cached_faces = []

    # Performance measurements
    inference_times = []
    display_start = time.time()
    display_frames = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Unable to read camera frame.")
            break

        frame_count += 1

        # Run InsightFace every fifth frame
        if frame_count % 5 == 0:

            inference_start = time.perf_counter()

            # Detection and embedding in one pipeline
            faces = face_app.get(frame)

            cached_faces = []

            for face in faces:

                embedding = face.normed_embedding

                name, score = recognize_face(
                    embedding,
                    database
                )

                # Obtain bounding box
                x1, y1, x2, y2 = face.bbox.astype(int)

                cached_faces.append(
                    (x1, y1, x2, y2, name, score)
                )

            inference_time = (
                time.perf_counter() - inference_start
            )

            inference_times.append(inference_time)

            # Keep only the latest 30 measurements
            inference_times = inference_times[-30:]

        # Draw cached recognition results
        for x1, y1, x2, y2, name, score in cached_faces:

            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

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

        # Measure display FPS separately
        display_frames += 1

        elapsed = time.time() - display_start

        display_fps = display_frames / elapsed if elapsed else 0

        average_ms = (
            sum(inference_times) / len(inference_times) * 1000
            if inference_times else 0
        )

        cv2.putText(
            frame,
            f"Display FPS: {display_fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Inference: {average_ms:.0f} ms",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "Live Face Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Webcam closed.")

    if inference_times:
        print(
            "Average inference time:",
            round(sum(inference_times) / len(inference_times) * 1000, 2),
            "ms"
        )


if __name__ == "__main__":
    recognize_webcam()
