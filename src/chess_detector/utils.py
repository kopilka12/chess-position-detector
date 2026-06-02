import cv2
import numpy as np
import os
from pdf2image import convert_from_path

def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2) + 1e-10))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  
    rect[2] = pts[np.argmax(s)]  

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]  
    return rect

def check_display():
    """Checks if a graphical display is available."""
    if os.name == 'nt':
        return True
    return os.environ.get('DISPLAY') is not None or os.environ.get('WAYLAND_DISPLAY') is not None

def warp_board(img, board):
    epsilon = 0.02 * cv2.arcLength(board, True)
    approx = cv2.approxPolyDP(board, epsilon, True)

    if len(approx) != 4:
        return None 

    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    pages_cv = []

    if ext == '.pdf':
        pages_pil = convert_from_path(file_path, dpi=150)
        for page in pages_pil:
            img_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            pages_cv.append(img_cv)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        pages_cv.append(img)
        
    return pages_cv

def fen_to_matrix(fen):
    """Converts the board part of a FEN string to an 8x8 matrix."""
    if not fen or fen == "intermediate":
        return None
        
    # Standard FEN can have multiple parts (board, turn, castling, etc.)
    # Our analyzer seems to return just the board part: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    board_part = fen.split(' ')[0]
    rows = board_part.split('/')
    matrix = []
    
    for row in rows:
        matrix_row = []
        for char in row:
            if char.isdigit():
                matrix_row.extend(['.'] * int(char))
            else:
                matrix_row.append(char)
        matrix.append(matrix_row)
    
    return matrix if len(matrix) == 8 else None

def get_move_from_fens(fen1, fen2):
    """Compares two FENs and returns ((r1, c1), (r2, c2)) for the most likely move."""
    m1 = fen_to_matrix(fen1)
    m2 = fen_to_matrix(fen2)
    
    if m1 is None or m2 is None:
        return None
        
    diffs = []
    for r in range(8):
        for c in range(8):
            if m1[r][c] != m2[r][c]:
                diffs.append((r, c))
                
    if not diffs:
        return None
        
    # Heuristic for a move: 
    # start_square: was piece, now empty ('.')
    # end_square: was something else, now a piece (or different piece)
    start_candidates = [d for d in diffs if m2[d[0]][d[1]] == '.']
    end_candidates = [d for d in diffs if m2[d[0]][d[1]] != '.']
    
    if start_candidates and end_candidates:
        # If there are multiple, try to find the one that matches piece type (simplified)
        # For now, just take the first ones
        return (start_candidates[0], end_candidates[0])
        
    # If it's a capture and only piece change is detected (shouldn't happen with FEN)
    # or other edge cases, just return the first two differences
    if len(diffs) >= 2:
        return (diffs[0], diffs[1])
        
    return None