import cv2
import numpy as np
from utils import angle_cos

class ChessboardDetector:
    def __init__(self, min_area=1000):
        self.min_area = min_area

    def _check_chessboard_pattern(self, x, y, w, h, gray):
        img_h, img_w = gray.shape
        step_x = w / 8.0
        step_y = h / 8.0
        
        offset_x = 7  
        offset_y = 7
        cell_colors = np.zeros((8, 8))
        
        for r in range(8):
            for c in range(8):
                cell_x = x + c * step_x
                cell_y = y + r * step_y
                
                pts = [
                    (int(cell_y + offset_y), int(cell_x + offset_x)),                        
                    (int(cell_y + offset_y), int(cell_x + step_x - offset_x)),                
                    (int(cell_y + step_y - offset_y), int(cell_x + offset_x)),                
                    (int(cell_y + step_y - offset_y), int(cell_x + step_x - offset_x))          
                ]
                
                valid_vals = []
                for py, px in pts:
                    if 0 <= py < img_h and 0 <= px < img_w:
                        valid_vals.append(gray[py, px])
                
                if valid_vals:
                    cell_colors[r, c] = np.median(valid_vals)

        median_board_val = np.median(cell_colors)
        match_pattern_1 = 0
        match_pattern_2 = 0
        
        for r in range(8):
            for c in range(8):
                is_light_cell = cell_colors[r, c] > median_board_val
                expected_light = (r + c) % 2 == 0 
                
                if is_light_cell == expected_light:
                    match_pattern_1 += 1
                else:
                    match_pattern_2 += 1
                    
        return max(match_pattern_1, match_pattern_2) >= 51

    def detect_boards(self, img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for channel in cv2.split(blurred):
            channel_edges = cv2.Canny(channel, 30, 100)
            edges = cv2.bitwise_or(edges, channel_edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        squares = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area: 
                continue
                
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            
            rect = cv2.minAreaRect(cnt)
            (center, (w, h), angle) = rect
            if w == 0 or h == 0: 
                continue
                
            aspect_ratio = min(w, h) / max(w, h)
            fill_ratio = area / (w * h)
            is_square = False
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                approx_pts = approx.reshape(-1, 2)
                max_cos = np.max([angle_cos(approx_pts[i], approx_pts[(i+1) % 4], approx_pts[(i+2) % 4]) for i in range(4)])
                if max_cos < 0.03 and aspect_ratio > 0.8:
                    is_square = True
            elif 4 <= len(approx) <= 6 and cv2.isContourConvex(approx):
                if fill_ratio > 0.95 and aspect_ratio > 0.85:
                    is_square = True
                    
            if is_square:
                squares.append(approx)

        squares = sorted(squares, key=cv2.contourArea, reverse=True)
        boards = []
        unused_squares = []

        for sq in squares:
            M = cv2.moments(sq)
            if M["m00"] == 0: 
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            is_inside = False
            for board in boards:
                if cv2.pointPolygonTest(board, (cX, cY), measureDist=False) >= 0:
                    is_inside = True
                    break
                    
            if is_inside:
                continue  
            
            x, y, w, h = cv2.boundingRect(sq)
            
            if self._check_chessboard_pattern(x, y, w, h, gray):
                boards.append(sq)
            else:
                unused_squares.append(sq)

        sq_info = []
        for sq in unused_squares:
            x, y, w, h = cv2.boundingRect(sq)
            cx, cy = x + w / 2.0, y + h / 2.0
            sq_info.append((cx, cy, w, h, x, y, sq))

        visited = set()
        components = []

        for i in range(len(sq_info)):
            if i in visited: continue
            
            comp = []
            queue = [i]
            visited.add(i)
            
            while queue:
                curr = queue.pop(0)
                comp.append(sq_info[curr])
                cx1, cy1, w1, h1, _, _, _ = sq_info[curr]
                
                for j in range(len(sq_info)):
                    if j in visited: continue
                    cx2, cy2, w2, h2, _, _, _ = sq_info[j]
                    
                    if max(w1, w2) / min(w1, w2) > 1.3: continue
                    if max(h1, h2) / min(h1, h2) > 1.3: continue
                    
                    dist = np.hypot(cx1 - cx2, cy1 - cy2)
                    if dist < 1.5 * max(w1, w2):
                        visited.add(j)
                        queue.append(j)
                        
            components.append(comp)

        for comp in components:
            if 50 <= len(comp) <= 80:
                min_x = min([item[4] for item in comp])
                min_y = min([item[5] for item in comp])
                max_x = max([item[4] + item[2] for item in comp])
                max_y = max([item[5] + item[3] for item in comp])
                
                board_w = max_x - min_x
                board_h = max_y - min_y
                
                if self._check_chessboard_pattern(min_x, min_y, board_w, board_h, gray):
                    merged_board = np.array([
                        [[min_x, min_y]], 
                        [[max_x, min_y]], 
                        [[max_x, max_y]], 
                        [[min_x, max_y]]
                    ], dtype=np.int32)
                    boards.append(merged_board)
                    
        return boards

    @staticmethod
    def draw_boards(img, boards, color=(0, 255, 0), thickness=3):
        output = img.copy()
        
        cv2.drawContours(output, boards, -1, color, thickness)
        
        for i, board in enumerate(boards):
            x, y, w, h = cv2.boundingRect(board)
            
            text = f"Board {i + 1}"
            
            text_x = x
            text_y = y - 10 if y > 25 else y + 25
            
            cv2.putText(output, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            
            cv2.putText(output, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return output