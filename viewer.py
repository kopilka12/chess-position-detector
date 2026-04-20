import cv2
import numpy as np
from utils import warp_board

class BoardViewer:
    def __init__(self, max_w=1640, max_h=760):
        self.max_w = max_w
        self.max_h = max_h

    def _show_with_ratio(self, window_name, img):
        rect = cv2.getWindowImageRect(window_name)
        if rect[2] <= 0 or rect[3] <= 0:
            cv2.imshow(window_name, img)
            return

        win_w, win_h = rect[2], rect[3]
        img_h, img_w = img.shape[:2]
        
        scaling = min(win_w / img_w, win_h / img_h)
        new_w, new_h = int(img_w * scaling), int(img_h * scaling)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x_offset = (win_w - new_w) // 2
        y_offset = (win_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        cv2.imshow(window_name, canvas)

    def display_interactive(self, pages, detector, analyzer=None):
        current_page = 0
        total_pages = len(pages)
        window_name = "Chessboards Detector Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        last_page = -1
        processed_img = None

        while True:
            if current_page != last_page:
                img = pages[current_page].copy()
                boards = detector.detect_boards(img)
                processed_img = detector.draw_boards(img, boards)
                
                info_lines = [f"Page: {current_page + 1}/{total_pages} | Boards: {len(boards)}"]
                if analyzer and boards:
                    for idx, board in enumerate(boards):
                        warped = warp_board(img, board)
                        if warped is not None:
                            fen = analyzer.predict_fen(warped, strict=False)
                            if fen: info_lines.append(f"Board {idx+1} FEN: {fen}")

                y_offset = 40
                for line in info_lines:
                    (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(processed_img, (15, y_offset - h - 5), (25 + w, y_offset + 5), (0, 0, 0), -1)
                    cv2.putText(processed_img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y_offset += 35
                last_page = current_page

            self._show_with_ratio(window_name, processed_img)
            
            key = cv2.waitKeyEx(30)
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: 
                break
            elif key in (2555904, 65363, 63235, 100, 68):
                if current_page < total_pages - 1: current_page += 1
            elif key in (2424832, 65361, 63234, 97, 65):
                if current_page > 0: current_page -= 1

        cv2.destroyAllWindows()

    def display_video_frames(self, frames_with_info, detector):
        current_idx = 0
        total_frames = len(frames_with_info)
        window_name = "Chessboards Video Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        last_idx = -1
        processed_img = None

        while True:
            if current_idx != last_idx:
                frame, timestamp, fens = frames_with_info[current_idx]
                img = frame.copy()
                boards = detector.detect_boards(img)
                processed_img = detector.draw_boards(img, boards)
                
                info_lines = [f"Pos: {current_idx + 1}/{total_frames} | Time: [{timestamp}]"]
                if fens:
                    for idx, fen in enumerate(fens):
                        info_lines.append(f"Board {idx+1} FEN: {fen}")

                y_offset = 40
                for line in info_lines:
                    (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(processed_img, (15, y_offset - h - 5), (25 + w, y_offset + 5), (0, 0, 0), -1)
                    cv2.putText(processed_img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y_offset += 35
                last_idx = current_idx

            self._show_with_ratio(window_name, processed_img)
            
            key = cv2.waitKeyEx(30)
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: 
                break
            elif key in (2555904, 65363, 63235, 100, 68):
                if current_idx < total_frames - 1: current_idx += 1
            elif key in (2424832, 65361, 63234, 97, 65):
                if current_idx > 0: current_idx -= 1

        cv2.destroyAllWindows()
