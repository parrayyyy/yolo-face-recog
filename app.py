
import streamlit as st
import cv2
import numpy as np
import json
import subprocess
import sys

from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "known_faces" / "face_db.json"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="YOLO Face Recognition",
    page_icon="👤",
    layout="wide"
)


# ==========================================
# LOAD PROJECT MODULES WHEN REQUIRED
# ==========================================

def get_models():

    from common import (
        detect_faces,
        extract_embedding,
        load_database,
        save_database,
        recognize_face
    )

    return (
        detect_faces,
        extract_embedding,
        load_database,
        save_database,
        recognize_face
    )


def read_database():

    if not DB_PATH.exists():
        return {}

    with open(DB_PATH, "r") as file:
        return json.load(file)


def decode_image(uploaded_file):

    image_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        np.uint8
    )

    return cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #10131c;
}

.stApp {
    background-color: #10131c;
}

section[data-testid="stSidebar"] {
    background-color: #171b29;
}

h1, h2, h3 {
    color: #ffffff;
}

p, label {
    color: #cbd5e1;
}

div[data-testid="stMetric"] {
    background-color: #1c2333;
    padding: 20px;
    border-radius: 12px;
}

.stButton > button {
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("👤 Face Recognition")

st.sidebar.caption("YOLO + InsightFace Computer Vision")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Register Face",
        "Photo Recognition",
        "Live Webcam",
        "Face Database"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Powered by Python, YOLO, OpenCV and InsightFace"
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("Face Recognition Dashboard")

    st.write(
        "Welcome to your AI-powered face recognition system."
    )

    database = read_database()

    total_people = len(database)

    total_embeddings = sum(
        len(items) for items in database.values()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Registered People",
        total_people
    )

    col2.metric(
        "Stored Face Samples",
        total_embeddings
    )

    col3.metric(
        "Recognition Engine",
        "InsightFace"
    )

    st.divider()

    st.subheader("Available Features")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "📸 Photo Recognition\n\n"
            "Upload a group photograph to identify "
            "registered participants."
        )

        st.info(
            "👤 Face Registration\n\n"
            "Add new identities using reference photographs."
        )

    with col2:

        st.info(
            "🎥 Live Webcam\n\n"
            "Launch live face recognition using your webcam."
        )

        st.info(
            "🗂️ Face Database\n\n"
            "View and manage registered identities."
        )


# ==========================================
# REGISTER FACE
# ==========================================

elif page == "Register Face":

    st.title("Register a New Face")

    st.write(
        "Upload a clear photograph containing one "
        "consenting participant."
    )

    name = st.text_input(
        "Enter person's name"
    )

    uploaded_file = st.file_uploader(
        "Upload reference photograph",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = decode_image(uploaded_file)

        if image is not None:

            st.image(
                image,
                channels="BGR",
                caption="Reference Photograph",
                width=350
            )

            if st.button("Register Person"):

                if not name.strip():

                    st.error("Please enter a name.")

                else:

                    (
                        detect_faces,
                        extract_embedding,
                        load_database,
                        save_database,
                        recognize_face
                    ) = get_models()

                    with st.spinner("Processing face..."):

                        boxes = detect_faces(image)

                        if len(boxes) == 0:

                            st.error("No face detected.")

                        elif len(boxes) > 1:

                            st.error(
                                "Multiple faces detected. "
                                "Upload a photograph containing "
                                "only one person."
                            )

                        else:

                            embedding = extract_embedding(
                                image,
                                boxes[0]
                            )

                            if embedding is None:

                                st.error(
                                    "Could not extract face features."
                                )

                            else:

                                database = load_database()

                                clean_name = name.strip()

                                if clean_name not in database:
                                    database[clean_name] = []

                                database[clean_name].append(
                                    embedding.tolist()
                                )

                                save_database(database)

                                st.success(
                                    f"{clean_name} registered successfully!"
                                )

                                st.rerun()

        else:

            st.error("Could not read uploaded image.")


# ==========================================
# GROUP PHOTO RECOGNITION
# ==========================================

elif page == "Photo Recognition":

    st.title("Group Photo Recognition")

    st.write(
        "Upload a photograph to find registered identities."
    )

    uploaded_file = st.file_uploader(
        "Choose a group photograph",
        type=["jpg", "jpeg", "png"],
        key="group_photo"
    )

    if uploaded_file:

        image = decode_image(uploaded_file)

        if image is not None:

            st.image(
                image,
                channels="BGR",
                caption="Original Photograph",
                width=600
            )

            if st.button("Recognize Faces"):

                (
                    detect_faces,
                    extract_embedding,
                    load_database,
                    save_database,
                    recognize_face
                ) = get_models()

                with st.spinner("Recognizing faces..."):

                    database = load_database()

                    if not database:

                        st.error(
                            "No registered identities found."
                        )

                    else:

                        boxes = detect_faces(
                            image,
                            image_size=1280
                        )

                        result_image = image.copy()

                        results_list = []

                        for box in boxes:

                            x1, y1, x2, y2 = box

                            embedding = extract_embedding(
                                image,
                                box
                            )

                            if embedding is None:

                                person = "Unknown"
                                score = 0

                            else:

                                person, score = recognize_face(
                                    embedding,
                                    database
                                )

                            results_list.append(
                                (person, score)
                            )

                            color = (
                                (0, 255, 0)
                                if person != "Unknown"
                                else (0, 0, 255)
                            )

                            cv2.rectangle(
                                result_image,
                                (x1, y1),
                                (x2, y2),
                                color,
                                2
                            )

                            cv2.putText(
                                result_image,
                                person,
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2
                            )

                        st.subheader("Recognition Results")

                        st.image(
                            result_image,
                            channels="BGR",
                            caption="Processed Photograph",
                            width=800
                        )

                        recognized = sum(
                            1 for person, score in results_list
                            if person != "Unknown"
                        )

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Faces Detected",
                            len(boxes)
                        )

                        col2.metric(
                            "Recognized",
                            recognized
                        )

                        st.write("### Detected Identities")

                        for person, score in results_list:

                            st.write(
                                f"**{person}** — Similarity: {score:.3f}"
                            )

                        output_path = (
                            OUTPUT_DIR / "gui_recognized.jpg"
                        )

                        cv2.imwrite(
                            str(output_path),
                            result_image
                        )

                        st.success(
                            "Processed photograph saved successfully."
                        )

        else:

            st.error("Invalid image.")


# ==========================================
# LIVE WEBCAM
# ==========================================

elif page == "Live Webcam":

    st.title("Live Face Recognition")

    st.write(
        "Launch the webcam recognition engine."
    )

    st.warning(
        "The webcam opens in a separate desktop window. "
        "This is the initial GUI integration; live video "
        "will be embedded in a later version."
    )

    if st.button("Start Webcam Recognition"):

        script = BASE_DIR / "recognize_webcam.py"

        if not script.exists():

            st.error("Webcam script not found.")

        else:

            try:

                creation_flags = (
                    subprocess.CREATE_NEW_CONSOLE
                    if sys.platform == "win32"
                    else 0
                )

                subprocess.Popen(
                    [sys.executable, str(script)],
                    cwd=str(BASE_DIR),
                    creationflags=creation_flags
                )

                st.success(
                    "Webcam recognition launched! "
                    "Press Q in the webcam window to stop."
                )

            except Exception as error:

                st.error(str(error))


# ==========================================
# FACE DATABASE
# ==========================================

elif page == "Face Database":

    st.title("Registered Face Database")

    database = read_database()

    if not database:

        st.info("No registered identities found.")

    else:

        st.write(
            f"Total registered identities: {len(database)}"
        )

        for person, embeddings in database.items():

            with st.expander(person):

                st.write(
                    f"Stored face samples: {len(embeddings)}"
                )

                if st.button(
                    f"Delete {person}",
                    key=f"delete_{person}"
                ):

                    st.session_state["delete_person"] = person

        if "delete_person" in st.session_state:

            person = st.session_state["delete_person"]

            st.warning(
                f"Confirm deletion of {person}'s biometric embeddings."
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button("Confirm Delete"):

                    database.pop(person, None)

                    with open(DB_PATH, "w") as file:
                        json.dump(database, file, indent=4)

                    del st.session_state["delete_person"]

                    st.success("Identity deleted.")

                    st.rerun()

            with col2:

                if st.button("Cancel"):

                    del st.session_state["delete_person"]

                    st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "YOLO Face Recognition System | "
    "Local Development Version"
)
