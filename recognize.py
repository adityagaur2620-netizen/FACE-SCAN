import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

DATASET_PATH = "dataset"

known_encodings = []
known_names = []

print("Loading dataset...")

for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print("Dataset loaded successfully!")

def mark_attendance(name):
    with open("attendance.csv", "a+") as f:
        f.seek(0)
        data = f.readlines()
        name_list = [line.split(",")[0] for line in data]

        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d,%H:%M:%S")
            f.write(f"{name},{dt_string}\n")
            print(f"Attendance marked for {name}")

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Recognition started... Press Q to exit")

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):

        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)

        top, right, bottom, left = face_location

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
