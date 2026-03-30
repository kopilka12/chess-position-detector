import os
import cv2
import numpy as np
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
        
        self.last_fens = None

    def _is_video(self):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg']
        return os.path.splitext(self.file_path)[1].lower() in video_extensions

    def run(self):
        if self._is_video():
            if self.split:
                print("Error: --split is not supported for video files.")
                return
            self._process_video()
            return

        print(f"Loading document: {self.file_path}...")
        try:
            pages_cv = load_document(self.file_path)
        except Exception as e:
            print(f"Error: {e}")
            return

        if not pages_cv:
            print("Error: No images loaded from the document.")
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

    def _process_video(self):
        print(f"Processing video: {self.file_path}...")
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.file_path}")
            return

        interval_frames = 2

        if self.generate_txt and self.analyzer:
            if not self.analyzer.load_resources():
                cap.release()
                return
            with open("boards_data.txt", "w", encoding="utf-8") as f:
                f.write(f"{'='*40}\n" f"File: {self.file_path}\n" f"{'='*40}\n\n")

        frames_to_show = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval_frames == 0:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_str = self._format_time(timestamp_ms)
                
                boards = self.detector.detect_boards(frame)
                if boards:
                    is_changed = True
                    if self.generate_txt and self.analyzer:
                        is_changed = self._analyze_and_save_video_data(frame, boards, timestamp_str)
                        if is_changed:
                            print(f"[{timestamp_str}] Position changed. Saved.")
                    
                    if self.show and is_changed:
                        frames_to_show.append((frame.copy(), timestamp_str))
                else:
                    self.last_fens = None

            frame_count += 1

        cap.release()
        if self.generate_txt:
            print("Data saved in 'boards_data.txt'!")
            
        if self.show and self.viewer and frames_to_show:
            self.viewer.display_video_frames(frames_to_show, self.detector)
            frames_to_show.clear()
        elif self.show and not frames_to_show:
            print("No boards detected in the video to show.")

    def _format_time(self, milliseconds):
        seconds = int(milliseconds // 1000)
        ms = int(milliseconds % 1000)
        hh = seconds // 3600
        mm = (seconds % 3600) // 60
        ss = seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ms:03d}"

    def _analyze_and_save_video_data(self, img, boards, timestamp_str, output_filename="boards_data.txt"):
        current_fens = []
        for board in boards:
            warped = warp_board(img, board)
            if warped is not None:
                fen = self.analyzer.predict_fen(warped)
                if fen:
                    current_fens.append(fen)
        
        if not current_fens:
            return False

        if current_fens == self.last_fens:
            return False
            
        self.last_fens = current_fens
        with open(output_filename, "a", encoding="utf-8") as f:
            for fen in current_fens:
                f.write(f"[{timestamp_str}] - {fen}\n")
        return True

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
        current_fens = []
        board_data = []

        for b_idx, board in enumerate(boards):
            warped = warp_board(img, board)
            if warped is None:
                continue
            
            fen = self.analyzer.predict_fen(warped)
            if fen:
                current_fens.append(fen)
                x, y, bw, bh = cv2.boundingRect(board)
                board_data.append({
                    'idx': b_idx + 1,
                    'x': x, 'y': y, 'w': bw, 'h': bh,
                    'fen': fen
                })
        
        with open(output_filename, "a", encoding="utf-8") as f:
            if page_num is not None:
                f.write("=================================\n")
                f.write(f"Page {page_num} \n")
                f.write("=================================\n\n")
            
            for data in board_data:
                f.write(f"Board {data['idx']}.\n")
                f.write(f"x: {data['x']}\n")
                f.write(f"y: {data['y']}\n")
                f.write(f"w: {data['w']}\n")
                f.write(f"h: {data['h']}\n")
                f.write(f"Position (FEN) : {data['fen']}\n\n")
