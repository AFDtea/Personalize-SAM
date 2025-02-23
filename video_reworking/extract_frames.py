import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, sequence_name, first_frame_path=None):
    """
    Extract frames from a video and save them in DAVIS dataset format.
    
    Args:
        video_path: Path to input video file
        output_dir: Base DAVIS directory path 
        sequence_name: Name for this video sequence
        first_frame_path: Path to already extracted first frame (if exists)
    """
    # Create output directory structure
    frames_dir = os.path.join(output_dir, '2017', 'JPEGImages', '480p', sequence_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create ImageSets directory and val.txt
    imagesets_dir = os.path.join(output_dir, '2017', 'ImageSets', '2017')
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # Write sequence name to val.txt
    with open(os.path.join(imagesets_dir, 'val.txt'), 'w') as f:
        f.write(sequence_name)

    # Open video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    if first_frame_path:
        # If we have a first frame extracted at 1 second, calculate its frame number
        first_frame_number = int(fps)  # frame number at 1 second
        print(f"Your FFMPEG frame from 1 second would be approximately frame {first_frame_number}")
        
        # Copy the existing first frame to the correct location
        first_frame = cv2.imread(first_frame_path)
        first_frame_out_path = os.path.join(frames_dir, f"{first_frame_number:05d}.jpg")
        cv2.imwrite(first_frame_out_path, first_frame)
        print(f"Saved your first frame as: {first_frame_out_path}")
    
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
            
        # Save frame as JPEG with DAVIS naming convention (00000.jpg, 00001.jpg, etc.)
        output_path = os.path.join(frames_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(output_path, frame)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    video.release()
    print(f"\nDone! Extracted {frame_count} frames to {frames_dir}")
    print(f"\nNote: Place your mask for the first frame (00000.png) in:")
    print(f"{os.path.join(output_dir, '2017', 'Annotations', '480p', sequence_name)}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Extract video frames in DAVIS format')
#     parser.add_argument('video_path', type=str, help='Path to input video file')
#     parser.add_argument('--davis_dir', type=str, default='./DAVIS',
#                         help='Path to DAVIS dataset directory (default: ./DAVIS)')
#     parser.add_argument('--sequence_name', type=str, required=True,
#                         help='Name for this video sequence')
#     parser.add_argument('--first_frame', type=str, help='Path to your FFMPEG-extracted first frame')
    
#     args = parser.parse_args()
    
#     extract_frames(args.video_path, args.davis_dir, args.sequence_name)

if __name__ == "__main__":
    video_path = input("Enter the path to the input video file: ")
    davis_dir = input("Enter the path to the DAVIS dataset directory (default: ./DAVIS): ") or './DAVIS'
    sequence_name = input("Enter the name for this video sequence: ")
    first_frame = input("Enter the path to your FFMPEG-extracted first frame (optional): ") or None
    
    extract_frames(video_path, davis_dir, sequence_name, first_frame)