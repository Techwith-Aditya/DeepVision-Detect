# DeepVision-Detect

## Project Overview  
This project processes an input video and detects human faces using **MTCNN (Multi-Task Cascaded Convolutional Networks)**. The detected faces are highlighted, and the processed video is saved for further analysis.  
This project has been successfully deployed on an NVIDIA DGX A100 installed at my college, leveraging its powerful computational capabilities for efficient face detection.

## Applications  
This project can be used in various real-world scenarios, such as:  
- **Surveillance Systems** – Detect and track faces in security footage.  
- **Automated Attendance** – Identify students or employees from video recordings.  
- **Smart Video Editing** – Enhance or blur faces automatically in post-production.  
- **Behavior Analysis** – Analyze crowd movement and facial presence in public areas.  
- **Retail Analytics** – Understand customer engagement by detecting faces in store footage.  

## Features  
- Detects human faces in video footage.  
- Draws bounding boxes around detected faces.  
- Processes and saves the output video for later review.  
- Displays real-time face detection while processing.  

## Technologies Used  
- **Python** – Used for scripting and automation.
- **OpenCV** – Handles video processing and frame manipulation.  
- **MTCNN** – Deep learning model that detects and highlights faces with high accuracy.
- **NVIDIA DGX A100** – Provides high-performance AI computing for real-time processing.


## Installation and Setup  
1. Clone the repository:  
```sh
git clone https://github.com/your-username/Face-Detection.git
cd Face-Detection
```
2. Install required dependencies:
```sh
pip install opencv-python mtcnn
```
3. Run the face detection script:
```sh
python video_face_detection.py
```
4. The output video will be saved as output_classroom.mp4.
