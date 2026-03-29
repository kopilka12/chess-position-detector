import argparse
from app import ChessApp
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main():
    parser = argparse.ArgumentParser(description='Chessboard Detection and Analysis Tool (OOP Version)')
    parser.add_argument('path', type=str, help='Path to the file (PDF, image, or video)')
    parser.add_argument('--show', action='store_true', help='Show visualization')
    parser.add_argument('--split', action='store_true', help='Cut the detected boards into squares and save them in the split folder')
    parser.add_argument('--txt', action='store_true', help='Save detected board data')
    args = parser.parse_args()

    app = ChessApp(
        file_path=args.path, 
        show=args.show, 
        split=args.split, 
        generate_txt=args.txt
    )
    
    app.run()

if __name__ == "__main__":
    main()