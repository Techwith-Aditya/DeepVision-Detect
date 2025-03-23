# Original-Previous:
import cv2
from mtcnn import MTCNN

detector = MTCNN()
video_path = "classroom.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video not found.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_filename = "output_classroom.mp4"  
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved as: {output_filename}")
# _________________________________________________________________________________________________________________________________
# 1st: Fixing Issue #1: Lack of Error Handling for MTCNN Initialization).

import cv2
from mtcnn import MTCNN

# Here's what I changed...
try:
    detector = MTCNN()
except Exception as e:
    print(f"Error: Failed to initialize MTCNN detector. Details: {str(e)}")
    print("Please ensure all dependencies (e.g., TensorFlow) are installed correctly.")
    exit(1)

video_path = "classroom.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video not found.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_filename = "output_classroom.mp4"  
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved as: {output_filename}")
# _________________________________________________________________________________________________________________________________
# 2nd:  Fixed the Issue #2: Hardcoded Video File Path

import cv2
from mtcnn import MTCNN
import argparse

try:
    detector = MTCNN()
except Exception as e:
    print(f"Error: Failed to initialize MTCNN detector. Details: {str(e)}")
    print("Please ensure all dependencies (e.g., TensorFlow) are installed correctly.")
    exit(1)

# Set up argument parser for video file path
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
args = parser.parse_args()

# Use the provided video path from command-line argument
cap = cv2.VideoCapture(args.video_path)

if not cap.isOpened():
    print(f"Error: Video not found at {args.video_path}.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_filename = "output_classroom.mp4"  
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved as: {output_filename}")
# _________________________________________________________________________________________________________________________________
# 3rd: Fixed Issue #3: Issue of the output file being overwritten without warning if it already exists

import cv2
from mtcnn import MTCNN
import argparse
import os
from datetime import datetime

try:
    detector = MTCNN()
except Exception as e:
    print(f"Error: Failed to initialize MTCNN detector. Details: {str(e)}")
    print("Please ensure all dependencies (e.g., TensorFlow) are installed correctly.")
    exit(1)

# Set up argument parser for video file path
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
args = parser.parse_args()

# Use the provided video path from command-line argument
cap = cv2.VideoCapture(args.video_path)

if not cap.isOpened():
    print(f"Error: Video not found at {args.video_path}.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
base_output_filename = "output_classroom.mp4"

# Check if the output file exists and generate a unique name if necessary
if os.path.exists(base_output_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_classroom_{timestamp}.mp4"
else:
    output_filename = base_output_filename

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved as: {output_filename}")
# _________________________________________________________________________________________________________________________________

