import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import load_document, warp_board
from detector import ChessboardDetector
from analyzer import ChessPositionAnalyzer
from viewer import BoardViewer

class ChessApp:
    def __init__(self, file_path, show=False, save_video=False, split=False, generate_txt=None):
        self.file_path = os.path.abspath(file_path)
        self.show = show
        self.save_video = save_video
        self.split = split
        self.generate_txt = os.path.abspath(generate_txt) if generate_txt else None
        
        self.detector = ChessboardDetector()
        self.analyzer = ChessPositionAnalyzer() if (generate_txt or show or save_video) else None
        self.viewer = BoardViewer() if (show or save_video) else None
        
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

        if (self.generate_txt or self.show) and self.analyzer:
            if not self.analyzer.load_resources():
                return
            if self.generate_txt:
                with open(self.generate_txt, "w", encoding="utf-8") as f:
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
                self._analyze_and_save_data(img, boards, output_filename=self.generate_txt, page_num=i+1)

        if self.split:
            print(f"File saving completed in the '{os.path.abspath('split')}' folder!")
        if self.generate_txt:
            print(f"Data saved in '{self.generate_txt}'!")

        if self.show and self.viewer:
            self.viewer.display_interactive(pages_cv, self.detector, self.analyzer)

    def _process_video(self):
        print(f"Processing video: {self.file_path}...")
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.file_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, unit="frame", desc="Processing Video")

        out = None
        if self.save_video:
            base, ext = os.path.splitext(self.file_path)
            output_path = f"{base}_positions{ext}"
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        interval_frames = 1

        if (self.generate_txt or self.show or self.save_video) and self.analyzer:
            if not self.analyzer.load_resources():
                cap.release()
                if out: out.release()
                pbar.close()
                return
            if self.generate_txt:
                with open(self.generate_txt, "w", encoding="utf-8") as f:
                    f.write(f"{'='*40}\n" f"File: {self.file_path}\n" f"{'='*40}\n\n")

        frames_to_show = []
        frame_count = 0
        current_fens = []

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
                    if self.analyzer:
                        is_changed, current_fens = self._analyze_and_save_video_data(frame, boards, timestamp_str, output_filename=self.generate_txt)
                        if is_changed:
                            pbar.write(f"[{timestamp_str}] Position changed. Saved." if self.generate_txt else f"[{timestamp_str}] Position changed.")
                    
                    if self.show and is_changed:
                        frames_to_show.append((frame.copy(), timestamp_str, current_fens))
                else:
                    self.last_fens = None
                    current_fens = []

                if self.save_video:
                    processed_frame = self.detector.draw_boards(frame, boards)
                    info_lines = [f"Time: [{timestamp_str}] | Boards: {len(boards)}"]
                    for idx, fen in enumerate(current_fens):
                        info_lines.append(f"Board {idx+1}: {fen}")
                    
                    self.viewer.draw_info(processed_frame, info_lines)
                    out.write(processed_frame)

            frame_count += 1
            pbar.update(1)

        cap.release()
        if out: out.release()
        pbar.close()

        if self.save_video:
            video_clip = None
            original_clip = None
            final_clip = None
            temp_output = output_path.replace("_positions", "_positions_temp")
            
            try:
                try:
                    from moviepy.editor import VideoFileClip
                except ImportError:
                    from moviepy import VideoFileClip
                
                print("Adding audio to the processed video...")
                
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
                os.rename(output_path, temp_output)
                
                video_clip = VideoFileClip(temp_output)
                original_clip = VideoFileClip(self.file_path)
                
                if original_clip.audio is not None:
                    # MoviePy 2.0 uses with_audio instead of set_audio
                    if hasattr(video_clip, 'with_audio'):
                        final_clip = video_clip.with_audio(original_clip.audio)
                    else:
                        final_clip = video_clip.set_audio(original_clip.audio)
                        
                    final_clip.write_videofile(output_path, codec="libx264", logger=None)
                else:
                    print("Original video has no audio.")
                    video_clip.close()
                    video_clip = None # Set to None so finally doesn't close it again
                    os.rename(temp_output, output_path)

            except Exception as e:
                print(f"Warning: Could not add audio to the video. {e}")
            finally:
                if video_clip: video_clip.close()
                if original_clip: original_clip.close()
                if final_clip: final_clip.close()
                
                if os.path.exists(temp_output):
                    if not os.path.exists(output_path):
                        try:
                            os.rename(temp_output, output_path)
                        except Exception as e:
                            print(f"Error restoring video from temp: {e}")
                    else:
                        try:
                            os.remove(temp_output)
                        except Exception as e:
                            print(f"Note: Could not remove temp file {temp_output}: {e}")

            print(f"Processed video saved as: {output_path}")

        if self.generate_txt:
            print(f"Data saved in '{self.generate_txt}'!")
            
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

    def _analyze_and_save_video_data(self, img, boards, timestamp_str, output_filename=None):
        current_fens = []
        for board in boards:
            warped = warp_board(img, board)
            if warped is not None:
                # For video, keep strict check (strict=True)
                fen = self.analyzer.predict_fen(warped, timestamp=timestamp_str, strict=True)
                if fen:
                    current_fens.append(fen)
        
        if not current_fens:
            return False, []

        if current_fens == self.last_fens:
            return False, current_fens
            
        self.last_fens = current_fens
        if self.generate_txt and output_filename:
            with open(output_filename, "a", encoding="utf-8") as f:
                for fen in current_fens:
                    f.write(f"[{timestamp_str}] - {fen}\n")
        return True, current_fens

    def _slice_and_save_boards(self, img, boards, page_num, output_dir="split"):
        output_dir = os.path.abspath(output_dir)
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

    def _analyze_and_save_data(self, img, boards, output_filename=None, page_num=None):
        current_fens = []
        board_data = []

        for b_idx, board in enumerate(boards):
            warped = warp_board(img, board)
            if warped is None:
                continue
            
            timestamp = f"Page {page_num}" if page_num else None
            # For images and PDF, disable strict check (strict=False)
            fen = self.analyzer.predict_fen(warped, timestamp=timestamp, strict=False)
            if fen:
                current_fens.append(fen)
                x, y, bw, bh = cv2.boundingRect(board)
                board_data.append({
                    'idx': b_idx + 1,
                    'x': x, 'y': y, 'w': bw, 'h': bh,
                    'fen': fen
                })
        
        if output_filename:
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
