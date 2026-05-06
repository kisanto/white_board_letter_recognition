import cv2
import cv2.aruco as aruco 
import numpy as np

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


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    ret ,frame = cap.read()
    if not ret : break

    corners, ids ,_ =aruco.detectMarkers(frame,aruco_dict,parameters=parameters)

    # Inside the while loop...
    if ids is not None:
        # Draw the markers it DOES see
        #aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"I see {len(ids)} markers: {ids.flatten()}") # This prints to your terminal

        if len(ids) == 4:
            board_view = get_wraped_board(frame, corners, ids)
            cv2.imshow("Digital Whiteboard", board_view)
        else:
            # Tell us which ones are missing
            cv2.putText(frame, f"Only found {len(ids)} markers. Need 4.", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            
    aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("Raw Camera View", frame)
    if cv2.waitKey(1) & 0xFF == 27: # Press Esc to exit
        break
cap.release()
cv2.destroyAllWindows()