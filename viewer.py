import cv2
from utils import warp_board

class BoardViewer:
    def __init__(self, max_w=1640, max_h=760):
        self.max_w = max_w
        self.max_h = max_h

    def display_interactive(self, pages, detector, analyzer=None):
        current_page = 0
        total_pages = len(pages)
        
        window_name = "Chessboards Detector Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\nNavigation:")
        print("  <- / A   : Previous page")
        print("  -> / D   : Next page")
        print("    ESC    : Exit\n")

        while True:
            img = pages[current_page].copy()
            
            boards = detector.detect_boards(img)
            processed_img = detector.draw_boards(img, boards)
            boards_count = len(boards)
            
            text = f"Page: {current_page + 1}/{total_pages} | Boards: {boards_count}"
            
            if analyzer and boards:
                fens = []
                for board in boards:
                    warped = warp_board(img, board)
                    if warped is not None:
                        # For images/PDF, strict=False as in app.py
                        fen = analyzer.predict_fen(warped, strict=False)
                        if fen:
                            fens.append(fen)
                
                if fens:
                    if len(fens) == 1:
                        text += f" | FEN: {fens[0]}"
                    else:
                        text += f" | FEN[1]: {fens[0]} ..."

            cv2.putText(processed_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_name, processed_img)
            
            key = cv2.waitKeyEx(0)
            
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: 
                break
            elif key in (2555904, 65363, 63235, 100, 68):
                if current_page < total_pages - 1:
                    current_page += 1
            elif key in (2424832, 65361, 63234, 97, 65):
                if current_page > 0:
                    current_page -= 1

        cv2.destroyAllWindows()

    def display_video_frames(self, frames_with_info, detector):
        current_idx = 0
        total_frames = len(frames_with_info)
        
        window_name = "Chessboards Video Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\nVideo Navigation:")
        print("  <- / A   : Previous detected position")
        print("  -> / D   : Next detected position")
        print("    ESC    : Exit\n")

        while True:
            frame, timestamp, fens = frames_with_info[current_idx]
            img = frame.copy()
            
            boards = detector.detect_boards(img)
            processed_img = detector.draw_boards(img, boards)
            
            text = f"Pos: {current_idx + 1}/{total_frames} | Time: [{timestamp}]"
            
            if fens:
                if len(fens) == 1:
                    text += f" | FEN: {fens[0]}"
                else:
                    text += f" | FEN[1]: {fens[0]} ..."

            cv2.putText(processed_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_name, processed_img)
            
            key = cv2.waitKeyEx(0)
            
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: 
                break
            elif key in (2555904, 65363, 63235, 100, 68):
                if current_idx < total_frames - 1:
                    current_idx += 1
            elif key in (2424832, 65361, 63234, 97, 65):
                if current_idx > 0:
                    current_idx -= 1

        cv2.destroyAllWindows()
