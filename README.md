# Chess Position Detector

A powerful computer vision tool designed to detect chessboards and analyze piece positions from images, PDFs, and video files. It uses deep learning to generate FEN (Forsyth-Edwards Notation) strings representing the board state.

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
  - **OpenGL**: Required by OpenCV GUI.
    - *Ubuntu*: `sudo apt install libgl1-mesa-glx`

## Installation

You can install the project directly from GitHub using the following command (recommended to use the `2x2` branch for latest updates):

```bash
pip install git+https://github.com/kopilka12/chess-position-detector.git@2x2
```

## Usage

After installation, the tool is available via the `chess-position-detector` command.

### Command Line Arguments

| Argument | Description |
| :--- | :--- |
| `path` | **Required.** Path to the input file (Image, PDF, or Video). |
| `--show` | Launch an interactive viewer to see detected boards and positions. |
| `--showheatmap` | Show a confidence heatmap alongside the board detection. |
| `--video` | Process a video file and save a new version with detection overlays. |
| `--effects` | Enable shiny sparks animation on moves (use with `--video` and `--show`). |
| `--split` | Crop detected boards into 64 square images and save them in the `split/` folder. |
| `--txt [FILE]` | Export detected FEN positions to a text file (default: `boards_data.txt`). |

### Examples

**1. Analyze a PDF and save data to a custom file:**
```bash
chess-position-detector manual.pdf --txt results.txt
```

**2. Process a video with effects and visual feedback:**
```bash
chess-position-detector game.mp4 --video --effects --show
```

## License

This project is licensed under the MIT License.
