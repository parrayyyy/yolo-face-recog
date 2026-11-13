
from common import (
    detect_faces,
    extract_embedding,
    load_database,
    save_database
)

import cv2


def register_person(name, image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    boxes = detect_faces(image, image_size=640)

    if len(boxes) == 0:
        print("No face detected. Try a clearer photograph.")
        return

    if len(boxes) > 1:
        print("Multiple faces detected.")
        print("Please provide a photograph containing one person.")
        return

    embedding = extract_embedding(image, boxes[0])

    if embedding is None:
        print("Could not extract facial features.")
        return

    database = load_database()

    if name not in database:
        database[name] = []

    database[name].append(embedding.tolist())

    save_database(database)

    print(f"\n{name} registered successfully!")
    print("Face embedding saved in database.")


if __name__ == "__main__":

    person_name = input("Enter person's name: ").strip()

    image_path = input("Enter reference image path: ").strip()
    image_path = image_path.strip('"')

    if person_name:
        register_person(person_name, image_path)
    else:
        print("Name cannot be empty.")
