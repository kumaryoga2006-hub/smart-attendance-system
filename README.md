# smart-attendance-system
mini project
import cv2
import os
import numpy as np
import sqlite3
from datetime import datetime

# ========== DATABASE SETUP ==========
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    name TEXT,
    time TEXT
)
""")
conn.commit()

# ========== FACE CASCADE ==========
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ========== STEP 1: CAPTURE DATASET ==========
def capture_faces():
    name = input("Enter student name: ")
    path = f"dataset/{name}"
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("Capturing images...")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{path}/{count}.jpg", face)
            count += 1

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        cv2.imshow("Capturing Faces", frame)

        if count >= 20:
            break

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Dataset captured!")


# ========== STEP 2: TRAIN MODEL ==========
def train_model():
    data_path = "dataset"
    faces = []
    labels = []
    label_dict = {}
    label_id = 0

    for person in os.listdir(data_path):
        label_dict[label_id] = person
        person_path = os.path.join(data_path, person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, 0)

            faces.append(img)
            labels.append(label_id)

        label_id += 1

    labels = np.array(labels)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, labels)
    model.save("model.yml")

    # Save labels
    np.save("labels.npy", label_dict)

    print("Training completed!")


# ========== STEP 3: RECOGNITION ==========
def recognize_faces():
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("model.yml")

    label_dict = np.load("labels.npy", allow_pickle=True).item()

    cap = cv2.VideoCapture(0)

    print("Starting attendance system...")

    marked = set()  # avoid duplicate entries

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            label, confidence = model.predict(face)

            if confidence < 70:
                name = label_dict[label]

                if name not in marked:
                    time = datetime.now().strftime("%H:%M:%S")
                    cursor.execute(
                        "INSERT INTO attendance VALUES (?, ?)", (name, time)
                    )
                    conn.commit()
                    marked.add(name)

                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ========== MAIN MENU ==========
while True:
    print("\n===== FACE ATTENDANCE SYSTEM =====")
    print("1. Capture Faces")
    print("2. Train Model")
    print("3. Start Attendance")
    print("4. Exit")

    choice = input("Enter choice: ")

    if choice == '1':
        capture_faces()
    elif choice == '2':
        train_model()
    elif choice == '3':
        recognize_faces()
    elif choice == '4':
        break
    else:
        print("Invalid choice!")

conn.close()
