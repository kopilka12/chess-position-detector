import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import os

class ChessPositionAnalyzer:
    def __init__(self, model_path="chess_fcn_2x2.keras", classes_path="classes.txt"):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.class_names = None
        self.is_loaded = False
        
        self.fen_mapping = {
            'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B', 'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
            'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b', 'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k',
            'empty': 'empty'
        }

    def load_resources(self):
        if self.is_loaded: return True
        try:
            if not os.path.exists(self.model_path): return False
            self.model = tf.keras.models.load_model(self.model_path, safe_mode=False)
            with open(self.classes_path, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Model loading error: {e}")
            return False

    def predict_fen(self, warped_board, timestamp=None, strict=True):
        if not self.is_loaded and not self.load_resources(): return None

        # 1. Preprocess the whole board
        board_gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
        board_gray = cv2.resize(board_gray, (512, 512))
        img_normalized = board_gray.astype(np.float32) / 255.0

        # 2. FCN magic: Predict entire board at once (512x512 -> 8x8 heatmap)
        board_input = np.expand_dims(img_normalized, axis=[0, -1]) # (1, 512, 512, 1)
        prediction = self.model.predict(board_input, verbose=0) # (1, 8, 8, 13)
        final_heatmap = prediction[0] # (8, 8, 13)

        # 3. Logging and Confidence
        self._log_predictions_heatmap(final_heatmap, timestamp)

        if strict:
            threshold = 0.95 # Adjusted for FCN context on virtual boards
            confidences = np.max(final_heatmap, axis=-1)
            min_conf = np.min(confidences)
            
            if min_conf < threshold:
                low_conf_idx = np.unravel_index(np.argmin(confidences), confidences.shape)
                cell_name = f"{chr(ord('a')+low_conf_idx[1])}{8-low_conf_idx[0]}"
                # Only print if it's likely a moving piece or very low confidence
                if min_conf < 0.95:
                    print(f"Low confidence at {cell_name}: {min_conf:.4f}")
                return "intermediate"
            
        class_indices = np.argmax(final_heatmap, axis=-1)
        return self._get_fen_from_matrix(class_indices)

    def _log_predictions_heatmap(self, heatmap, timestamp=None):
        if timestamp is None: timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open("predictions.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- Sliding Window 2x2 Prediction at [{timestamp}] ---\n")
                for r in range(8):
                    for c in range(8):
                        prob_vec = heatmap[r, c]
                        best_idx = np.argmax(prob_vec)
                        if self.class_names[best_idx] != 'empty' or prob_vec[best_idx] < 0.9:
                            cell_name = f"{chr(ord('a')+c)}{8-r}"
                            probs_str = ", ".join([f"{self.class_names[j]}: {prob_vec[j]:.2f}" for j in range(len(self.class_names)) if prob_vec[j] > 0.1])
                            f.write(f"Cell {cell_name}: {probs_str}\n")
        except: pass

    def _get_fen_from_matrix(self, matrix):
        fen_rows = []
        for r in range(8):
            empty_count, row_str = 0, ""
            for c in range(8):
                predicted_class = self.class_names[matrix[r, c]]
                piece = self.fen_mapping.get(predicted_class, 'empty') 
                if piece == 'empty':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += piece
            if empty_count > 0: row_str += str(empty_count)
            fen_rows.append(row_str)
        return "/".join(fen_rows)
