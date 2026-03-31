import cv2
import numpy as np
from collections import deque

# --- CONFIGURATION ---
width_img, height_img = 480, 640
# High buffer (15) ensures the green box and result are rock-solid
points_buffer = deque(maxlen=15) 
color_mode = False
count = 0

def pre_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    # Adaptive thresholding is the best for detecting paper edges
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5))
    dial = cv2.dilate(thresh, kernel, iterations=2)
    return cv2.erode(dial, kernel, iterations=1)

def get_contours(img):
    biggest, max_area = np.array([]), 0
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest, max_area = approx, area
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)] # Top-Left
    myPointsNew[3] = myPoints[np.argmax(add)] # Bottom-Right
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] # Top-Right
    myPointsNew[2] = myPoints[np.argmax(diff)] # Bottom-Left
    return myPointsNew

# --- MAIN ENGINE ---
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.resize(img, (width_img, height_img))
    img_display = img.copy()
    img_thresh = pre_processing(img)
    biggest = get_contours(img_thresh)
    
    img_final_scan = np.zeros((height_img, width_img, 3), np.uint8)

    if biggest.size != 0:
        # STABILITY: Average the last 15 detections
        points_buffer.append(biggest)
        avg_biggest = np.mean(points_buffer, axis=0).astype(np.int32)
        
        # Warp using smoothed coordinates
        pts1 = np.float32(reorder(avg_biggest))
        pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, matrix, (width_img, height_img))
        
        if color_mode:
            img_final_scan = cv2.detailEnhance(img_warp, sigma_s=10, sigma_r=0.15)
        else:
            # High-Contrast B&W Filter
            img_gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
            img_final_scan = cv2.adaptiveThreshold(img_gray, 255, 1, 1, 15, 15)
            img_final_scan = cv2.cvtColor(cv2.medianBlur(img_final_scan, 3), cv2.COLOR_GRAY2BGR)

        cv2.drawContours(img_display, [avg_biggest], -1, (0, 255, 0), 10)
        cv2.putText(img_display, "READY", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(img_display, "FINDING...", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

    # UI DASHBOARD
    dashboard = np.hstack((img_display, img_final_scan))
    
    # Header & Footer
    header = np.zeros((60, dashboard.shape[1], 3), np.uint8)
    cv2.putText(header, "PRO DOCUMENT SCANNER", (350, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    
    footer = np.zeros((50, dashboard.shape[1], 3), np.uint8)
    cv2.putText(footer, "[S] SAVE SCAN  [C] TOGGLE COLOR  [Q] QUIT", (300, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)
    
    final_view = np.vstack((header, dashboard, footer))
    cv2.imshow("AI Vision Project", final_view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'): color_mode = not color_mode
    if key == ord('s') and biggest.size != 0:
        cv2.imwrite(f"scan_{count}.jpg", img_final_scan)
        # Visual White Flash
        cv2.imshow("AI Vision Project", np.full(final_view.shape, 255, dtype=np.uint8))
        cv2.waitKey(100)
        print(f"Saved: scan_{count}.jpg")
        count += 1

cap.release()
cv2.destroyAllWindows()