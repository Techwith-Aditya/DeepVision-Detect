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
# 4th: Fixed Issue #4: Issue of having no Frame Skipping Option because the script processes every frame, which can be slow for long videos or high FPS

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

# Set up argument parser for video file path and frame skipping
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between processing (default: 0, process every frame)")
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

frame_count = 0  # Track the current frame number

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process only if frame_count meets the skip-frames condition
    if frame_count % (args.skip_frames + 1) == 0:
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
# 5th: Fixed Issue #5: Issue of having no Face Detection Confidence Threshold

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

# Set up argument parser for video file path, frame skipping, and confidence threshold
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between processing (default: 0, process every frame)")
parser.add_argument("--confidence", type=float, default=0.9, help="Minimum confidence threshold for face detection (default: 0.9)")
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

frame_count = 0  # Track the current frame number

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process only if frame_count meets the skip-frames condition
    if frame_count % (args.skip_frames + 1) == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)

        for face in faces:
            # Filter faces based on confidence threshold
            if face['confidence'] >= args.confidence:
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
#7th: Fixed Issue #7: No updates or progress while detecting faces in videos

import cv2
from mtcnn import MTCNN
import argparse
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    detector = MTCNN()
    logger.info("MTCNN detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MTCNN detector: {str(e)}")
    logger.error("Please ensure all dependencies (e.g., TensorFlow) are installed correctly")
    exit(1)

# Set up argument parser
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between processing (default: 0)")
parser.add_argument("--confidence", type=float, default=0.9, help="Minimum confidence threshold for face detection (default: 0.9)")
args = parser.parse_args()

# Video setup
cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    logger.error(f"Video not found at {args.video_path}")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
base_output_filename = "output_classroom.mp4"

# Generate unique output filename
if os.path.exists(base_output_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_classroom_{timestamp}.mp4"
else:
    output_filename = base_output_filename

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
logger.info(f"Processing video: {args.video_path}")
logger.info(f"Output will be saved as: {output_filename}")
logger.info(f"Total frames to process: {total_frames}")

frame_count = 0
faces_detected_total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Show progress every 100 frames or at the start
    if frame_count % 100 == 0 or frame_count == 1:
        progress = (frame_count / total_frames) * 100
        logger.info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

    # Process frames based on skip-frames argument
    if frame_count % (args.skip_frames + 1) == 0:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(frame_rgb)
            
            faces_detected = 0
            for face in faces:
                if face['confidence'] >= args.confidence:
                    faces_detected += 1
                    faces_detected_total += 1
                    x, y, w, h = face['box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if faces_detected > 0:
                logger.debug(f"Frame {frame_count}: Detected {faces_detected} faces")

        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {str(e)}")

    out.write(frame)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Processing interrupted by user")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

logger.info(f"Processing completed. Total faces detected: {faces_detected_total}")
logger.info(f"Output saved as: {output_filename}")
# _________________________________________________________________________________________________________________________________
# 8th: Fixed issue #8: Hardcoded Output Codec may not be supported on all systems

import cv2
from mtcnn import MTCNN
import argparse
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    detector = MTCNN()
    logger.info("MTCNN detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MTCNN detector: {str(e)}")
    logger.error("Please ensure all dependencies (e.g., TensorFlow) are installed correctly")
    exit(1)

# Set up argument parser with codec option
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between processing (default: 0)")
parser.add_argument("--confidence", type=float, default=0.9, help="Minimum confidence threshold for face detection (default: 0.9)")
parser.add_argument("--codec", type=str, default='mp4v', help="FourCC codec for output video (e.g., mp4v, xvid, h264) (default: mp4v)")
args = parser.parse_args()

# Video setup
cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    logger.error(f"Video not found at {args.video_path}")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Use the user-specified codec
fourcc = cv2.VideoWriter_fourcc(*args.codec.upper())  # Convert codec to uppercase for consistency
base_output_filename = "output_classroom.mp4"

# Generate unique output filename
if os.path.exists(base_output_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_classroom_{timestamp}.mp4"
else:
    output_filename = base_output_filename

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    logger.error(f"Failed to initialize video writer with codec '{args.codec}'. Try a different codec (e.g., xvid, h264).")
    cap.release()
    exit()

logger.info(f"Processing video: {args.video_path}")
logger.info(f"Using codec: {args.codec}")
logger.info(f"Output will be saved as: {output_filename}")
logger.info(f"Total frames to process: {total_frames}")

frame_count = 0
faces_detected_total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Show progress every 100 frames or at the start
    if frame_count % 100 == 0 or frame_count == 1:
        progress = (frame_count / total_frames) * 100
        logger.info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

    # Process frames based on skip-frames argument
    if frame_count % (args.skip_frames + 1) == 0:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(frame_rgb)
            
            faces_detected = 0
            for face in faces:
                if face['confidence'] >= args.confidence:
                    faces_detected += 1
                    faces_detected_total += 1
                    x, y, w, h = face['box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if faces_detected > 0:
                logger.debug(f"Frame {frame_count}: Detected {faces_detected} faces")

        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {str(e)}")

    out.write(frame)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Processing interrupted by user")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

logger.info(f"Processing completed. Total faces detected: {faces_detected_total}")
logger.info(f"Output saved as: {output_filename}")
# _________________________________________________________________________________________________________________________________
# 9th: Fixed issue #9: Now the code cleans up properly when you stop it with Ctrl+C

import cv2
from mtcnn import MTCNN
import argparse
import os
from datetime import datetime
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def cleanup(cap, out, message="Processing completed"):
    """Clean up resources and exit gracefully."""
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"{message}. Total faces detected: {faces_detected_total}")
    logger.info(f"Output saved as: {output_filename}")
    sys.exit(0)

try:
    detector = MTCNN()
    logger.info("MTCNN detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MTCNN detector: {str(e)}")
    logger.error("Please ensure all dependencies (e.g., TensorFlow) are installed correctly")
    exit(1)

# Set up argument parser with codec option
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between processing (default: 0)")
parser.add_argument("--confidence", type=float, default=0.9, help="Minimum confidence threshold for face detection (default: 0.9)")
parser.add_argument("--codec", type=str, default='mp4v', help="FourCC codec for output video (e.g., mp4v, xvid, h264) (default: mp4v)")
args = parser.parse_args()

# Video setup
cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    logger.error(f"Video not found at {args.video_path}")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Use the user-specified codec
fourcc = cv2.VideoWriter_fourcc(*args.codec.upper())
base_output_filename = "output_classroom.mp4"

# Generate unique output filename
if os.path.exists(base_output_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_classroom_{timestamp}.mp4"
else:
    output_filename = base_output_filename

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    logger.error(f"Failed to initialize video writer with codec '{args.codec}'. Try a different codec (e.g., xvid, h264).")
    cap.release()
    exit()

logger.info(f"Processing video: {args.video_path}")
logger.info(f"Using codec: {args.codec}")
logger.info(f"Output will be saved as: {output_filename}")
logger.info(f"Total frames to process: {total_frames}")

frame_count = 0
faces_detected_total = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cleanup(cap, out)
            break

        frame_count += 1
        
        # Show progress every 100 frames or at the start
        if frame_count % 100 == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            logger.info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

        # Process frames based on skip-frames argument
        if frame_count % (args.skip_frames + 1) == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(frame_rgb)
                
                faces_detected = 0
                for face in faces:
                    if face['confidence'] >= args.confidence:
                        faces_detected += 1
                        faces_detected_total += 1
                        x, y, w, h = face['box']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if faces_detected > 0:
                    logger.debug(f"Frame {frame_count}: Detected {faces_detected} faces")

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")

        out.write(frame)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cleanup(cap, out, "Processing interrupted by user (q pressed)")
            break

except KeyboardInterrupt:
    logger.info("Processing stopped by keyboard interrupt (Ctrl+C)")
    cleanup(cap, out, "Processing stopped by keyboard interrupt")
# _________________________________________________________________________________________________________________________________
# 10TH: Fixed issue #10: Now the code checks if the video’s size and speed are valid before starting

import cv2
from mtcnn import MTCNN
import argparse
import os
from datetime import datetime
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def cleanup(cap, out, message="Processing completed"):
    """Clean up resources and exit gracefully."""
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"{message}. Total faces detected: {faces_detected_total}")
    logger.info(f"Output saved as: {output_filename}")
    sys.exit(0)

try:
    detector = MTCNN()
    logger.info("MTCNN detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MTCNN detector: {str(e)}")
    logger.error("Please ensure all dependencies (e.g., TensorFlow) are installed correctly")
    exit(1)

# Set up argument parser with codec option
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between processing (default: 0)")
parser.add_argument("--confidence", type=float, default=0.9, help="Minimum confidence threshold for face detection (default: 0.9)")
parser.add_argument("--codec", type=str, default='mp4v', help="FourCC codec for output video (e.g., mp4v, xvid, h264) (default: mp4v)")
args = parser.parse_args()

# Video setup
cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    logger.error(f"Video not found at {args.video_path}")
    exit()

# Validate video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if frame_width <= 0 or frame_height <= 0:
    logger.error(f"Invalid video dimensions: width={frame_width}, height={frame_height}")
    cap.release()
    exit()
if fps <= 0:
    logger.error(f"Invalid frame rate: fps={fps}")
    cap.release()
    exit()
if total_frames <= 0:
    logger.error(f"Invalid frame count: total_frames={total_frames}")
    cap.release()
    exit()

# Use the user-specified codec
fourcc = cv2.VideoWriter_fourcc(*args.codec.upper())
base_output_filename = "output_classroom.mp4"

# Generate unique output filename
if os.path.exists(base_output_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_classroom_{timestamp}.mp4"
else:
    output_filename = base_output_filename

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    logger.error(f"Failed to initialize video writer with codec '{args.codec}'. Try a different codec (e.g., xvid, h264).")
    cap.release()
    exit()

logger.info(f"Processing video: {args.video_path}")
logger.info(f"Using codec: {args.codec}")
logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
logger.info(f"Output will be saved as: {output_filename}")

frame_count = 0
faces_detected_total = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cleanup(cap, out)
            break

        frame_count += 1
        
        # Show progress every 100 frames or at the start
        if frame_count % 100 == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            logger.info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

        # Process frames based on skip-frames argument
        if frame_count % (args.skip_frames + 1) == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(frame_rgb)
                
                faces_detected = 0
                for face in faces:
                    if face['confidence'] >= args.confidence:
                        faces_detected += 1
                        faces_detected_total += 1
                        x, y, w, h = face['box']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if faces_detected > 0:
                    logger.debug(f"Frame {frame_count}: Detected {faces_detected} faces")

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")

        out.write(frame)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cleanup(cap, out, "Processing interrupted by user (q pressed)")
            break

except KeyboardInterrupt:
    logger.info("Processing stopped by keyboard interrupt (Ctrl+C)")
    cleanup(cap, out, "Processing stopped by keyboard interrupt")
# _________________________________________________________________________________________________________________________________
# Fixed issue #11: Now you can turn off the video display window while processing

import cv2
from mtcnn import MTCNN
import argparse
import os
from datetime import datetime
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def cleanup(cap, out, message="Processing completed"):
    """Clean up resources and exit gracefully."""
    cap.release()
    out.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    logger.info(f"{message}. Total faces detected: {faces_detected_total}")
    logger.info(f"Output saved as: {output_filename}")
    sys.exit(0)

try:
    detector = MTCNN()
    logger.info("MTCNN detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MTCNN detector: {str(e)}")
    logger.error("Please ensure all dependencies (e.g., TensorFlow) are installed correctly")
    exit(1)

# Set up argument parser with display option
parser = argparse.ArgumentParser(description="Detect faces in a video and save the output.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between processing (default: 0)")
parser.add_argument("--confidence", type=float, default=0.9, help="Minimum confidence threshold for face detection (default: 0.9)")
parser.add_argument("--codec", type=str, default='mp4v', help="FourCC codec for output video (e.g., mp4v, xvid, h264) (default: mp4v)")
parser.add_argument("--no-display", action='store_true', help="Disable the display window during processing")
args = parser.parse_args()

# Video setup
cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    logger.error(f"Video not found at {args.video_path}")
    exit()

# Validate video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if frame_width <= 0 or frame_height <= 0:
    logger.error(f"Invalid video dimensions: width={frame_width}, height={frame_height}")
    cap.release()
    exit()
if fps <= 0:
    logger.error(f"Invalid frame rate: fps={fps}")
    cap.release()
    exit()
if total_frames <= 0:
    logger.error(f"Invalid frame count: total_frames={total_frames}")
    cap.release()
    exit()

# Use the user-specified codec
fourcc = cv2.VideoWriter_fourcc(*args.codec.upper())
base_output_filename = "output_classroom.mp4"

# Generate unique output filename
if os.path.exists(base_output_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_classroom_{timestamp}.mp4"
else:
    output_filename = base_output_filename

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    logger.error(f"Failed to initialize video writer with codec '{args.codec}'. Try a different codec (e.g., xvid, h264).")
    cap.release()
    exit()

logger.info(f"Processing video: {args.video_path}")
logger.info(f"Using codec: {args.codec}")
logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
logger.info(f"Output will be saved as: {output_filename}")
if args.no_display:
    logger.info("Display window disabled")
else:
    logger.info("Display window enabled")

frame_count = 0
faces_detected_total = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cleanup(cap, out)
            break

        frame_count += 1
        
        # Show progress every 100 frames or at the start
        if frame_count % 100 == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            logger.info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

        # Process frames based on skip-frames argument
        if frame_count % (args.skip_frames + 1) == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(frame_rgb)
                
                faces_detected = 0
                for face in faces:
                    if face['confidence'] >= args.confidence:
                        faces_detected += 1
                        faces_detected_total += 1
                        x, y, w, h = face['box']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if faces_detected > 0:
                    logger.debug(f"Frame {frame_count}: Detected {faces_detected} faces")

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")

        out.write(frame)
        if not args.no_display:
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cleanup(cap, out, "Processing interrupted by user (q pressed)")
                break

except KeyboardInterrupt:
    logger.info("Processing stopped by keyboard interrupt (Ctrl+C)")
    cleanup(cap, out, "Processing stopped by keyboard interrupt")
