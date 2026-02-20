import cv2
import os

name = input("Enter person name: ")

dataset_path = f"dataset/{name}"
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

print("Capturing images... Press Q to stop")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    
    img_path = f"{dataset_path}/{count}.jpg"
    cv2.imwrite(img_path, frame)
    count += 1

    if count >= 50:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset captured successfully!")
