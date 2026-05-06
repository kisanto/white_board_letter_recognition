import cv2
import cv2.aruco as aruco 
import numpy as np
import os

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

def get_wraped_board(frame, corners, ids):
    pts_src =np.zeros((4,2),dtype="float32")

    for i in range(len(ids)):
        marker_id = ids[i][0]
        if marker_id <4 :
            pts_src[marker_id]= np.mean(corners[i][0], axis=0)
    
    width = 800
    height =600
    pts_dts =np.array([[0,0],[width,0],[width,height],[0,height]],dtype="float32")

    matrix =cv2.getPerspectiveTransform(pts_src,pts_dts)

    wraped =cv2.warpPerspective(frame,matrix,(width,height))
    return wraped

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

hog = cv2.HOGDescriptor((20,20), (10,10), (5,5), (5,5), 9)

with np.load("processed_data_hog.npz") as data:
    train_data = data["train_data"]
    train_labels = data["train_labels"]

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(12.5)
svm.setGamma(0.5)
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
letters =sorted(os.listdir("dataset" )) 


while True:
    ret ,frame = cap.read()
    if not ret : break

    corners, ids ,_ =aruco.detectMarkers(frame,aruco_dict,parameters=parameters)

    if ids is not None:
        
        print(f"I see {len(ids)} markers: {ids.flatten()}") 

        if len(ids) == 4:
            board_view = get_wraped_board(frame, corners, ids)
            
            # 2. PHYSICALLY COVER THE MARKERS
            # We "paint" white rectangles over the corners of the color image
            m = 140  # Size of the patch in pixels
            h, w = board_view.shape[:2]
            bg_color = (255, 255, 255) # White (or 200 for light gray if board is off-white)
            gray_board = cv2.cvtColor(board_view, cv2.COLOR_BGR2GRAY)
            # Draw filled white rectangles over the 4 corners
            cv2.rectangle(board_view, (0, 0), (m, m), bg_color, -1)              # Top-Left
            cv2.rectangle(board_view, (w-m, 0), (w, m), bg_color, -1)            # Top-Right
            cv2.rectangle(board_view, (w-m, h-m), (w, h), bg_color, -1)          # Bottom-Right
            cv2.rectangle(board_view, (0, h-m), (m, h), bg_color, -1)            # Bottom-Left
            board_view = get_wraped_board(frame, corners, ids)
            cv2.imshow("Digital Whiteboard", board_view)
            gray_board = cv2.cvtColor(board_view, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_board, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
            margin = 70 
            h_board, w_board = thresh.shape

       
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)


            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if 20 < w < 150 and 20 < h < 150:
                    roi = thresh[y:y+h, x:x+w]

                    mask = np.zeros((max(w,h), max(w,h)), np.uint8)
                    mask[(max(w,h)-h)//2 : (max(w,h)-h)//2 + h, (max(w,h)-w)//2 : (max(w,h)-w)//2 + w] = roi

                    test_img = cv2.resize(mask, (20, 20))
                    test_img = deskew(test_img)
                    test_piggy = hog.compute(test_img).reshape(1,-1)

                    res = svm.predict(test_piggy, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                    raw_score = res[1][0][0]
                    _, result =svm.predict(test_piggy)
                    _, result = svm.predict(test_piggy)
                    idx = int(result[0][0])  

                    if 0 <= idx < len(letters):
                        label = letters[idx]
                        # Draw the box and the label on the board_view
                        cv2.rectangle(board_view, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(board_view, label, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Digital Whiteboard", board_view)

    aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("Raw Camera View", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break
cap.release()
cv2.destroyAllWindows()