# Chess Position Detector

A computer vision tool designed to detect chessboards and analyze piece positions from images, PDFs, and video files. It uses deep learning to generate FEN (Forsyth-Edwards Notation) strings representing the board state.

## Features

- **Multi-format Support**: Process single images, multi-page PDF documents, and video files.
- **Board Detection**: Automatically locates one or more chessboards within a frame.
- **Position Analysis**: Uses a custom Convolutional Neural Network (FCN) to recognize pieces and empty squares.
- **Visualization**: 
  - Interactive viewer for images and PDFs.
  - Heatmap generation for piece detection confidence.
  - Real-time video overlay with move animations (shiny sparks!).
- **Data Export**: Save detected positions to text files or split boards into individual square images for dataset creation.
- **Headless Support**: Gracefully handles execution on servers without a graphical display.

## Technologies & Libraries

- **Python**: Core programming language.
- **OpenCV (`opencv-python`)**: Primary library for image processing, board detection (contour analysis), and video manipulation.
- **TensorFlow**: Powers the Deep Learning model used for piece classification.
- **NumPy**: Efficient numerical operations for image matrices and coordinate transformations.
- **pdf2image & Pillow**: Used to convert PDF pages into processable image formats.
- **tqdm**: Provides progress bars for long-running video processing tasks.
- **MoviePy**: Handles video post-processing, such as adding audio back to analyzed videos.

## Requirements

- **Python 3.10 or higher**
- **System Dependencies**:
  - **Poppler**: Required by `pdf2image` for PDF processing.
    - *Ubuntu/Debian*: `sudo apt install poppler-utils`
    - *Windows*: Download from [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) and add `bin` to PATH.

## Installation

You can install the project directly from GitHub using the following command:

```bash
pip install git+https://github.com/kopilka12/chess-position-detector.git
```

## Usage

After installation, the tool is available via the `chess-position-detector` command.

### Basic Analysis
```bash
# Analyze an image and save results to a text file
chess-position-detector path/to/image.png --txt
```

### Visualization
```bash
# Show interactive board detection on a PDF
chess-position-detector document.pdf --show

# Show piece detection confidence heatmap
chess-position-detector board.jpg --showheatmap
```

### Video Processing
```bash
# Process video, save with overlays and animations
chess-position-detector game.mp4 --video --effects --show
```

### Dataset Preparation
```bash
# Cut detected boards into 64 individual square images
chess-position-detector page.png --split
```

## License

This project is licensed under the MIT License.
