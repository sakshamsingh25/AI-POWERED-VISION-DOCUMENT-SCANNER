# 📄 AI-Powered Vision Document Scanner

An advanced Computer Vision application that transforms your webcam into a high-performance document scanner. This project uses real-time image processing, perspective transformation, and temporal smoothing to create professional-grade digital scans from physical paper.

## ✨ Key Features

  * **🎯 Real-Time Detection:** Automatically identifies rectangular documents using contour approximation.
  * **⚖️ Temporal Smoothing:** Uses a 15-frame buffer to eliminate flickering and ensure a rock-solid UI.
  * **📐 Perspective Warp:** Corrects camera angles to provide a perfectly flat, top-down view.
  * **🪄 Professional Filters:** Includes an adaptive high-contrast B\&W mode and an enhanced color mode.
  * **📸 Instant Capture:** Visual "Flash" feedback when saving scans to local storage.
  * **🖥️ Aesthetic Dashboard:** A unified side-by-side view showing the live AI "brain" and the final result.

-----

## 🚀 How It Works

The scanner follows a sophisticated 4-stage pipeline to ensure accuracy:

1.  **Pre-processing:** Grayscale conversion and Adaptive Thresholding to find edges in any lighting.
2.  **Contour Logic:** Filtering for the largest 4-sided polygon in the frame.
3.  **Point Reordering:** Sorting coordinates (Top-Left, Top-Right, etc.) to prevent inverted warps.
4.  **Warping:** Applying a Perspective Transform matrix to map the paper to an A4-ratio canvas.

-----

## 🛠️ Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/AI-Doc-Scanner.git
    cd AI-Doc-Scanner
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install opencv-python numpy
    ```

4.  **Run the application:**

    ```bash
    python scanner.py
    ```
