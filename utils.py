import cv2
import numpy as np
import os
from pdf2image import convert_from_path

def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2) + 1e-10))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  
    rect[2] = pts[np.argmax(s)]  

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]  
    return rect

def warp_board(img, board):
    epsilon = 0.02 * cv2.arcLength(board, True)
    approx = cv2.approxPolyDP(board, epsilon, True)

    if len(approx) != 4:
        return None 

    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    pages_cv = []

    if ext == '.pdf':
        pages_pil = convert_from_path(file_path, dpi=150)
        for page in pages_pil:
            img_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            pages_cv.append(img_cv)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        pages_cv.append(img)
        
    return pages_cv