
import cv2

from pathlib import Path

from common import (
    detect_faces,
    extract_embedding,
    recognize_face
)


def recognize_photo(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Could not open image.")
        return

    print("Detecting faces...")

    boxes = detect_faces(image, image_size=1280)

    print(f"Detected {len(boxes)} faces.")

    for box in boxes:

        x1, y1, x2, y2 = box

        embedding = extract_embedding(image, box)

        if embedding is None:

            name = "Unrecognized face"
            score = 0

        else:

            name, score = recognize_face(embedding)

        print(f"{name}: similarity = {score:.3f}")

        # Draw bounding box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Display identity
        label = f"{name} ({score:.2f})"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    output_path = (
        Path(__file__).resolve().parent
        / "output"
        / "recognized_group.jpg"
    )

    output_path.parent.mkdir(exist_ok=True)

    cv2.imwrite(str(output_path), image)

    print("\nRecognition complete!")
    print(f"Output saved at: {output_path}")

    cv2.imshow("Face Recognition", image)

    print("Press any key to close the image.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    path = input("Enter group photo path: ").strip()
    path = path.strip('"')

    recognize_photo(path)
