import cv2

# Replace with your video path
video_path = "AI_Orchard.mp4"

try:
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        
        print(f"FPS: {fps}")
        print(f"Total frames: {frame_count}")
        print(f"Video duration: {duration:.2f} seconds")
        
    video.release()
except Exception as e:
    print(f"An error occurred: {str(e)}")