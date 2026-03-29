import cv2

class BoardViewer:
    def __init__(self, max_w=1640, max_h=760):
        self.max_w = max_w
        self.max_h = max_h

    def display_interactive(self, pages, detector):
        current_page = 0
        total_pages = len(pages)
        
        print("\nNawigacja:")
        print("  <- / A   : Poprzednia strona")
        print("  -> / D   : Następna strona")
        print("    ESC    : Wyjście\n")

        while True:
            img = pages[current_page].copy()
            
            boards = detector.detect_boards(img)
            processed_img = detector.draw_boards(img, boards)
            boards_count = len(boards)
            
            text = f"Page: {current_page + 1}/{total_pages} | Boards: {boards_count}"
            cv2.putText(processed_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            h, w = processed_img.shape[:2]
            if w > self.max_w or h > self.max_h:
                scale = min(self.max_w / w, self.max_h / h)
                processed_img = cv2.resize(processed_img, None, fx=scale, fy=scale)

            cv2.imshow("Chessboards Detector Viewer", processed_img)
            
            key = cv2.waitKeyEx(0)
            
            if key == 27 or cv2.getWindowProperty('Chessboards Detector Viewer', cv2.WND_PROP_VISIBLE) < 1: 
                break
            elif key in (2555904, 65363, 63235, 100, 68):
                if current_page < total_pages - 1:
                    current_page += 1
            elif key in (2424832, 65361, 63234, 97, 65):
                if current_page > 0:
                    current_page -= 1

        cv2.destroyAllWindows()