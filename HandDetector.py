import cv2
import numpy as np
import time
import random
import os
import psutil

PATH = "benchmark/"

# ------------------- Funções auxiliares -------------------

def process_frame(frame):
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    ycrcb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77])
    upper = np.array([255, 173, 127])
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)
    return mask

def calc_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def calc_angle(a, b, c):
    ab = calc_distance(a, b)
    bc = calc_distance(b, c)
    ac = calc_distance(a, c)
    if ab*bc == 0:
        return 0
    angle = np.arccos((ab**2 + bc**2 - ac**2)/(2*ab*bc))
    return np.degrees(angle)

def count_fingers(defects, contour):
    if defects is None:
        return 0, 0
    count = 0
    sum_angle = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        depth = d / 256.0
        if depth > 25:
            angle = calc_angle(start, far, end)
            if angle < 85:
                count += 1
                sum_angle += angle
    avg_angle = sum_angle / count if count > 0 else 0
    return min(5, count+1), avg_angle

def classify_gesture(fingers, contour, defects):
    if fingers == 0:
        return "Fist"
    if fingers == 5:
        return "Palm"
    if fingers == 1:
        return "Thumbs Up"
    if fingers == 2:
        return "Peace"
    if fingers == 3:
        return "OK"
    return f"{fingers} Fingers"

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 5000
    index = -1
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            index = i
    return contours, index

def save_image(frame):
    filename = os.path.join(PATH, f"media/hand_snapshot_{random.randint(0,999)}.png")
    cv2.imwrite(filename, frame)
    print("📸 Imagem salva como", filename)

def get_cpu_memory():
    process = psutil.Process()
    used_mem = process.memory_info().rss / (1024*1024)
    cpu = psutil.cpu_percent() / 100
    return used_mem, cpu

# ------------------- Main Loop -------------------

def main():
    os.makedirs(PATH+"media", exist_ok=True)
    os.makedirs(PATH+"csvs", exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    frame_number = 0
    csv_file = open(PATH+"csvs/performance_"+str(random.randint(0,999))+".csv", "w")
    csv_file.write("frame,fingers,maxContourArea,centerX,centerY,convexDefects,avgAngle,fps,usedMemoryMB,cpuLoad,gesture\n")

    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()

        mask = process_frame(frame)
        contours, index = find_largest_contour(mask)
        
        fingers = 0
        avg_angle = 0
        gesture = ""
        max_area = 0
        cx = cy = 0
        convex_defects = 0

        if index != -1:
            cnt = contours[index]
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            fingers, avg_angle = count_fingers(defects, cnt)
            gesture = classify_gesture(fingers, cnt, defects)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"]/M["m00"]
                cy = M["m01"]/M["m00"]
            max_area = cv2.contourArea(cnt)
            convex_defects = 0 if defects is None else defects.shape[0]
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
            hull_pts = cv2.convexHull(cnt)
            cv2.drawContours(frame, [hull_pts], -1, (255,0,0), 2)

        fps = 1.0 / (time.time() - prev_time)
        prev_time = time.time()
        used_mem, cpu = get_cpu_memory()

        if index != -1:
            csv_file.write(f"{frame_number},{fingers},{max_area:.2f},{cx:.2f},{cy:.2f},{convex_defects},{avg_angle:.2f},{fps:.2f},{used_mem:.2f},{cpu:.4f},{gesture}\n")
            csv_file.flush()
        
        # Show info
        cv2.putText(frame, f"Dedos: {fingers} - {gesture}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f} CPU: {cpu*100:.2f}% MEM: {used_mem:.2f}MB", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,50,50), 2)

        cv2.imshow("Detecção de Mão", frame)
        key = cv2.waitKey(100) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("2"):  # tecla 2 para snapshot
            save_image(frame)

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == "__main__":
    main()
