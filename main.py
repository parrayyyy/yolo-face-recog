
from register import register_person
from recognize_photo import recognize_photo
from recognize_webcam import recognize_webcam


def main():

    while True:

        print("\n================================")
        print("     YOLO FACE RECOGNITION")
        print("================================")

        print("1. Register a person")
        print("2. Recognize faces in group photo")
        print("3. Start live webcam recognition")
        print("4. Exit")

        choice = input("\nEnter your choice: ").strip()

        try:

            if choice == "1":

                name = input("Enter person's name: ").strip()

                path = input("Enter reference photo path: ").strip()
                path = path.strip('"')

                if name:
                    register_person(name, path)
                else:
                    print("Name cannot be empty.")

            elif choice == "2":

                path = input("Enter group photo path: ").strip()
                path = path.strip('"')

                recognize_photo(path)

            elif choice == "3":

                recognize_webcam()

            elif choice == "4":

                print("Exiting application...")
                break

            else:

                print("Invalid choice. Try again.")

        except Exception as error:

            print("An error occurred:", error)


if __name__ == "__main__":

    main()
