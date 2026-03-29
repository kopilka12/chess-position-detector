import numpy as np
import cv2
import tensorflow as tf

class ChessPositionAnalyzer:
    def __init__(self, model_path="chess_piece_model.keras", classes_path="classes.txt"):
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
        if self.is_loaded:
            return True
            
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.classes_path, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Model loading error: {e}")
            return False

    def predict_fen(self, warped_board):
        if not self.is_loaded and not self.load_resources():
            return None

        h, w = warped_board.shape[:2]
        step_x = w / 8.0
        step_y = h / 8.0
        
        cells = []
        for r in range(8):
            for c in range(8):
                start_y, end_y = int(r * step_y), int((r + 1) * step_y)
                start_x, end_x = int(c * step_x), int((c + 1) * step_x)
                
                cell = warped_board[start_y:end_y, start_x:end_x]
                
                if cell.shape[0] > 0 and cell.shape[1] > 0:
                    cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    cell_resized = cv2.resize(cell_gray, (64, 64))
                    cell_expanded = np.expand_dims(cell_resized, axis=-1)
                    cells.append(cell_expanded)
                else:
                    cells.append(np.zeros((64, 64, 1), dtype=np.uint8))
        
        batch = np.array(cells)
        predictions = self.model.predict(batch, verbose=0)
        class_indices = np.argmax(predictions, axis=1)
        
        return self._get_fen_from_predictions(class_indices)

    def _get_fen_from_predictions(self, class_indices):
        fen_rows = []
        for r in range(8):
            empty_count = 0
            row_str = ""
            for c in range(8):
                idx = r * 8 + c
                predicted_class = self.class_names[class_indices[idx]]
                piece = self.fen_mapping.get(predicted_class, 'empty') 
                
                if piece == 'empty':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += piece
                    
            if empty_count > 0:
                row_str += str(empty_count)
            fen_rows.append(row_str)
            
        return "/".join(fen_rows)