import os
import cv2
from utils import load_document, warp_board
from detector import ChessboardDetector
from analyzer import ChessPositionAnalyzer
from viewer import BoardViewer

class ChessApp:
    def __init__(self, file_path, show=False, split=False, generate_txt=False):
        self.file_path = file_path
        self.show = show
        self.split = split
        self.generate_txt = generate_txt
        
        self.detector = ChessboardDetector()
        self.analyzer = ChessPositionAnalyzer() if generate_txt else None
        self.viewer = BoardViewer() if show else None

    def run(self):
        print(f"Loading document: {self.file_path}...")
        try:
            pages_cv = load_document(self.file_path)
        except Exception as e:
            print(f"Error: {e}")
            return

        if self.generate_txt and self.analyzer:
            if not self.analyzer.load_resources():
                return
            with open("boards_data.txt", "w", encoding="utf-8") as f:
                f.write(f"{'='*40}\n" f"File: {self.file_path}\n" f"{'='*40}\n\n")

        for i, img in enumerate(pages_cv):
            boards = self.detector.detect_boards(img)
            
            if not boards:
                print(f"Page {i+1}: No boards found.")
                continue
                
            print(f"Page {i+1}: {len(boards)} boards found.")
            
            if self.split:
                self._slice_and_save_boards(img, boards, page_num=i+1)
                
            if self.generate_txt and self.analyzer:
                self._analyze_and_save_data(img, boards, page_num=i+1)

        if self.split:
            print("File saving completed in the 'split' folder!")
        if self.generate_txt:
            print("Data saved in 'boards_data.txt'!")

        if self.show and self.viewer:
            self.viewer.display_interactive(pages_cv, self.detector)

    def _slice_and_save_boards(self, img, boards, page_num, output_dir="split"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for b_idx, board in enumerate(boards):
            warped = warp_board(img, board)
            if warped is None:
                continue 
                
            h, w = warped.shape[:2]
            step_x = w // 8
            step_y = h // 8

            for r in range(8):
                for c in range(8):
                    start_y = r * step_y
                    end_y = (r + 1) * step_y
                    start_x = c * step_x
                    end_x = (c + 1) * step_x

                    cell = warped[start_y:end_y, start_x:end_x]

                    if cell.shape[0] > 0 and cell.shape[1] > 0:
                        cell_resized = cv2.resize(cell, (64, 64))
                        filename = os.path.join(
                            output_dir,
                            f"page_{page_num}_board_{b_idx}_cell_{r}_{c}.png"
                        )
                        cv2.imwrite(filename, cell_resized)

    def _analyze_and_save_data(self, img, boards, output_filename="boards_data.txt", page_num=None):
        with open(output_filename, "a", encoding="utf-8") as f:
            if page_num is not None:
                f.write("=================================\n")
                f.write(f"Page {page_num} \n")
                f.write("=================================\n\n")
            for b_idx, board in enumerate(boards):
                warped = warp_board(img, board)
                if warped is None:
                    continue
                    
                fen_string = self.analyzer.predict_fen(warped)
                x, y, bw, bh = cv2.boundingRect(board)
                
                f.write(f"Board {b_idx + 1}.\n")
                f.write(f"x: {x}\n")
                f.write(f"y: {y}\n")
                f.write(f"w: {bw}\n")
                f.write(f"h: {bh}\n")
                f.write(f"Position (FEN) : {fen_string}\n\n")



                #[00:00:00] ~~~~~~~
                