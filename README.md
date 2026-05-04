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

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kopilka12/chess-position-detector.git
   cd chess-position-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `tensorflow`, `opencv-python`, `numpy`, and `pdf2image` installed.*

## Usage

Run the tool via `main.py` with various flags depending on your needs:

### Basic Command
```bash
python main.py path/to/your/file.jpg
```

### Flags
| Flag | Description |
|------|-------------|
| `--show` | Show interactive visualization of detected boards. |
| `--txt` | Analyze positions and save FEN data to a file. Defaults to `boards_data.txt` if no filename is provided. Example: `--txt custom_output.txt` |
| `--split` | Cut detected boards into 64 squares and save them in the `/split` folder. |

### Examples

**1. Analyze a PDF and save FEN strings:**
```bash
python main.py documents/chess_book.pdf --txt
```

**2. Visualize detections in a video:**
```bash
python main.py videos/gameplay.mp4 --show
```

**3. Generate training data from an image:**
```bash
python main.py photo.png --split
```
