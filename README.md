# Chessboard Detection and Analysis Tool

Сomputer vision tool designed to detect chessboards in images, PDFs, and videos, analyze the piece positions, and generate FEN.

## Tech Stack

- **Language:** Python 3.10+
- **Computer Vision:** OpenCV (`opencv-python`)
- **Deep Learning:** TensorFlow / Keras
- **PDF Processing:** `pdf2image` (requires Poppler)
- **Data Handling:** NumPy

## Prerequisites

### Poppler (for PDF support)
This tool uses `pdf2image`, which requires **Poppler** to be installed on your system.

- **Windows:** Download the latest binary from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and add the `bin` folder to your System PATH.
- **Linux:** `sudo apt-get install poppler-utils`
- **macOS:** `brew install poppler`

## Installation (Recommended)

This project is now a standardized Python package. To install it:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kopilka12/chess-position-detector.git
   cd chess-position-detector
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install the project in editable mode:**
   ```bash
   pip install -e .
   ```
   *Note: This will automatically install all dependencies like `tensorflow`, `opencv-python`, etc.*

## Usage

After installation, you can run the tool from anywhere in your terminal using the `chess-position-detector` command:

### Basic Command
```bash
chess-position-detector path/to/your/file.jpg
```

### Flags
| Flag | Description |
|------|-------------|
| `--show` | Show interactive visualization of detected boards. |
| `--showheatmap` | Show heatmap of piece detections. |
| `--txt` | Analyze positions and save FEN data to a file. Example: `--txt custom_output.txt` |
| `--split` | Cut detected boards into 64 squares and save them in the `/split` folder. |
| `--video` | Save processed video with overlays. |
| `--effects` | Show shiny sparks animation on move (requires `--show` or `--video`). |

### Examples

**1. Analyze a PDF and save FEN strings:**
```bash
chess-position-detector documents/chess_book.pdf --txt
```

**2. Visualize detections in a video with effects:**
```bash
chess-position-detector videos/gameplay.mp4 --show --effects
```
