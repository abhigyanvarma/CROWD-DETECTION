
# YOLOv8 Video Detection with Clusters

## Overview

This project implements a video detection system using YOLOv8, which detects people in video frames and applies clustering techniques to assess crowd density and risk levels. The system is built using OpenCV, Tkinter, and the Ultralytics YOLO model, with DBSCAN clustering for analyzing detected individuals.

## Features

- **Real-time video processing**: Loads and processes video files for object detection.
- **Person detection**: Identifies people using YOLOv8.
- **Crowd clustering**: Groups detected people using DBSCAN clustering.
- **Risk assessment**: Categorizes the crowd density into low, medium, and high risk.
- **GUI Interface**: User-friendly interface built with Tkinter for video loading and visualization.

## Dependencies

To run this project, install the following dependencies:

```bash
pip install ultralytics opencv-python numpy pillow scikit-learn
```

## Installation & Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/YOLOv8-Video-Detection.git
   cd YOLOv8-Video-Detection
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python main.py
   ```

4. Click on "Load Video" to select a video file and start processing.

## How It Works

1. The application loads a selected video file.
2. YOLOv8 detects people in each frame.
3. DBSCAN clustering groups detected people based on proximity.
4. A risk level (low, medium, high) is assigned based on the number of clusters.
5. The processed video frames, along with detection and risk assessment, are displayed in the Tkinter interface.

## File Structure

```
Crowd Detection/
│── C Analysis/
│   ├── crowddetect.py   # Script for crowd detection
│   ├── output_video.mp4 # Processed output video
│   ├── yolov3.cfg       # YOLOv3 configuration file
│   ├── yolov8n.pt       # Pre-trained YOLOv8 model
│── License              # License information
│── README.md            # Project documentation
```

## Future Enhancements

- Implement real-time webcam support.
- Optimize clustering parameters for better accuracy.
- Enhance GUI with additional controls and visualization.

## License

This project is licensed under the MIT License.

## Team Members

- **Chiluvuri Siva Abhigyan Varma**
- **Uchuru Rushindra Kumar Reddy**
- **Jambugolam Eshwar Chand**
