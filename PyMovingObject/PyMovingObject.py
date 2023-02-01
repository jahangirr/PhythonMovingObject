
import cv2
import numpy as np

def get_background(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=8)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame

cap = cv2.VideoCapture(0);
ori_frame = get_background(0);
ori_gray_frame = cv2.cvtColor(ori_frame,cv2.COLOR_BGR2GRAY)


while (cap.isOpened()):
    
    ret , cur_frame  =  cap.read();
    cur_gray_frame =  cv2.cvtColor(cur_frame ,cv2.COLOR_BGR2GRAY)
    diff_frame = cv2.absdiff(cur_gray_frame,ori_gray_frame)
    ret , thres_frame = cv2.threshold(diff_frame,50,255,cv2.THRESH_BINARY)
    dialute_frame = cv2.dilate(thres_frame,None,iterations=2)

    contours, hierarchy = cv2.findContours(dialute_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
       cv2.drawContours(cur_gray_frame, contours, i, (0, 0, 255), 3)
       for contour in contours:

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(cur_gray_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow('Detected Objects', cur_gray_frame)
            
    if cv2.waitKey(100) & 0xFF == ord('q'):
     break  

cap.release()
cv2.destroyAllWindows()  

           
           

















