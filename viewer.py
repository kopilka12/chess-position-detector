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

    def draw_info(self, img, info_lines):
        y_offset = 40
        for line in info_lines:
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (15, y_offset - h - 5), (25 + w, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 35
        return img

    def draw_heatmap(self, heatmap, class_names):
        # heatmap: (8, 8, 13)
        h, w = 900, 900
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        step_x = w // 8
        step_y = h // 8
        
        confidences = np.max(heatmap, axis=-1)
        min_conf = np.min(confidences)

        for r in range(8):
            for c in range(8):
                prob_vec = heatmap[r, c]
                best_idx = np.argmax(prob_vec)
                prob = prob_vec[best_idx]
                class_name = class_names[best_idx]

                # Draw cell border
                cv2.rectangle(canvas, (c * step_x, r * step_y), ((c + 1) * step_x, (r + 1) * step_y), (200, 200, 200), 1)

                # Background color: orange for min confidence, light blue for pieces
                if prob == min_conf:
                    cv2.rectangle(canvas, (c * step_x + 1, r * step_y + 1), ((c + 1) * step_x - 1, (r + 1) * step_y - 1), (0, 165, 255), -1)
                elif class_name != 'empty':
                    cv2.rectangle(canvas, (c * step_x + 1, r * step_y + 1), ((c + 1) * step_x - 1, (r + 1) * step_y - 1), (255, 230, 200), -1)


                # Text
                text1 = class_name.replace("white_", "W_").replace("black_", "B_")
                text2 = f"{prob:.2f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.8
                thick = 2
                
                (tw1, th1), _ = cv2.getTextSize(text1, font, scale, thick)
                (tw2, th2), _ = cv2.getTextSize(text2, font, scale, thick)
                
                tx = c * step_x + (step_x - tw1) // 2
                ty = r * step_y + (step_y // 2) - 10
                cv2.putText(canvas, text1, (tx, ty), font, scale, (0, 0, 0), thick)
                
                tx = c * step_x + (step_x - tw2) // 2
                ty = r * step_y + (step_y // 2) + 20
                cv2.putText(canvas, text2, (tx, ty), font, scale, (50, 50, 50), thick)

        return canvas

    def display_interactive(self, pages, detector, analyzer=None, show_heatmap=False):
        current_page = 0
        total_pages = len(pages)
        window_name = "Chessboards Detector Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        last_page = -1
        processed_img = None

        print("\nNavigation:")
        print("  <- / A   : Previous page")
        print("  -> / D   : Next page")
        print("    ESC    : Exit\n")

        while True:
            if current_page != last_page:
                img = pages[current_page].copy()
                boards = detector.detect_boards(img)
                board_img = detector.draw_boards(img, boards)
                
                info_lines = [f"Page: {current_page + 1}/{total_pages} | Boards: {len(boards)}"]
                heatmap_img = None

                if analyzer and boards:
                    for idx, board in enumerate(boards):
                        warped = warp_board(img, board)
                        if warped is not None:
                            if show_heatmap:
                                fen, heatmap = analyzer.predict_fen(warped, strict=False, return_heatmap=True)
                                if heatmap is not None:
                                    heatmap_img = self.draw_heatmap(heatmap, analyzer.class_names)
                            else:
                                fen = analyzer.predict_fen(warped, strict=False)
                            
                            if fen: info_lines.append(f"Board {idx+1}: {fen}")

                self.draw_info(board_img, info_lines)
                
                if show_heatmap and heatmap_img is not None:
                    # Combine board_img and heatmap_img side by side
                    h1, w1 = board_img.shape[:2]
                    h2, w2 = heatmap_img.shape[:2]
                    new_h = max(h1, h2)
                    new_w = w1 + w2
                    combined = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    combined[:h1, :w1] = board_img
                    combined[:h2, w1:w1+w2] = heatmap_img
                    processed_img = combined
                else:
                    processed_img = board_img
                    
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

    def display_video_frames(self, frames_with_info, detector, analyzer=None):
        current_idx = 0
        total_frames = len(frames_with_info)
        window_name = "Chessboards Video Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        last_idx = -1
        processed_img = None

        while True:
            if current_idx != last_idx:
                frame_data = frames_with_info[current_idx]
                frame = frame_data[0]
                timestamp = frame_data[1]
                fens = frame_data[2]
                heatmap = frame_data[3] if len(frame_data) > 3 else None
                
                img = frame.copy()
                boards = detector.detect_boards(img)
                board_img = detector.draw_boards(img, boards)
                
                info_lines = [f"Pos: {current_idx + 1}/{total_frames} | Time: [{timestamp}]"]
                if fens:
                    for idx, fen in enumerate(fens):
                        info_lines.append(f"Board {idx+1}: {fen}")

                self.draw_info(board_img, info_lines)
                
                if heatmap is not None and analyzer is not None:
                    heatmap_img = self.draw_heatmap(heatmap, analyzer.class_names)
                    # Combine board_img and heatmap_img side by side
                    h1, w1 = board_img.shape[:2]
                    h2, w2 = heatmap_img.shape[:2]
                    new_h = max(h1, h2)
                    new_w = w1 + w2
                    combined = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    combined[:h1, :w1] = board_img
                    combined[:h2, w1:w1+w2] = heatmap_img
                    processed_img = combined
                else:
                    processed_img = board_img
                    
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
