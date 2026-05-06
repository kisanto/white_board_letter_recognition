import cv2
import numpy as np
import os

with np.load("processed_data.npz") as data:
    train_data = data["train_data"]
    train_labels = data["train_labels"]

KNN =cv2.ml.KNearest_create()
KNN.train(train_data,cv2.ml.ROW_SAMPLE,train_labels)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

letters =sorted(os.listdir("dataset"))

print("KNN μοντελο ετοιμο δειξε του πινακα")

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    x1, y1, x2, y2 = w//2-80, h//2-80, w//2+80, h//2+80
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    test_img = cv2.resize(thresh, (20, 20))
    test_flattened = test_img.reshape(-1, 400).astype(np.float32)
    ret, result, neighbors, dist = KNN.findNearest(test_flattened, k=7)
    prediction_index = int(result[0][0])
    predicted_letter = letters[prediction_index]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"I see: {predicted_letter}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Recognition", frame)
    cv2.imshow("What the AI sees", thresh)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()