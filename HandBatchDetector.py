import cv2
import numpy as np
import os
import random
import csv

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

def analyze_image(image_path, save_contours=True):
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Erro ao carregar imagem:", image_path)
        return None
    
    mask = process_frame(frame)
    contours, index = find_largest_contour(mask)
    
    if index == -1:
        print("Nenhuma mão detectada em:", image_path)
        return None
    
    cnt = contours[index]
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    fingers, avg_angle = count_fingers(defects, cnt)
    gesture = classify_gesture(fingers, cnt, defects)
    
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"]/M["m00"]
        cy = M["m01"]/M["m00"]
    else:
        cx = cy = 0
    max_area = cv2.contourArea(cnt)
    convex_defects = 0 if defects is None else defects.shape[0]

    # Desenha contornos se necessário
    if save_contours:
        cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
        hull_pts = cv2.convexHull(cnt)
        cv2.drawContours(frame, [hull_pts], -1, (255,0,0), 2)
        save_path = os.path.join(PATH, "images/processed", os.path.basename(image_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)

    return {
        "file": os.path.basename(image_path),
        "fingers": fingers,
        "maxContourArea": max_area,
        "centerX": cx,
        "centerY": cy,
        "convexDefects": convex_defects,
        "avgAngle": avg_angle,
        "gesture": gesture
    }

# ------------------- Batch processing -------------------

def main():
    input_dir = os.path.join(PATH, "images/allimgs")
    output_csv = os.path.join(PATH, "csvs/batch_results.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["file","fingers","maxContourArea","centerX","centerY","convexDefects","avgAngle","gesture"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for root, _, files in os.walk(input_dir):
            for f in files:
                image_path = os.path.join(root, f)
                info = analyze_image(image_path)
                if info:
                    writer.writerow(info)
                    print("✅ Processado:", f)

if __name__ == "__main__":
    main()
