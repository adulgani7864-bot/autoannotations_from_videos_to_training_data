import cv2
import os
import numpy as np

VIDEOS_FOLDER = "downloads_9_10" 
OUTPUT_FOLDER = "DW_PICKITEM_NORMAL_TRIME"  
TARGET_SIZE = 448
INTERPOLATION = cv2.INTER_LANCZOS4  
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
video_files = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi', '.mkv'))]
video_files.sort()

if not video_files:
    print("No videos found in the 'videos' folder.")
    exit()

def high_quality_resize(frame, target_size=448):
    """
    Resize frame to target_size x target_size with aspect ratio preservation
    Uses padding to avoid distortion
    """
    h, w = frame.shape[:2]
    
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=INTERPOLATION)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

for video_file in video_files:
    video_path = os.path.join(VIDEOS_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Skipping corrupt video: {video_file}")
        continue  
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        print(f"Skipping empty video: {video_file}")
        cap.release()
        continue

    delay = int(1000 / fps) if fps > 0 else 30
    print(f"Processing: {video_file} | Original: {original_width}x{original_height} | Output: 448x448 | FPS: {fps}")

    recording = False  
    paused = False    
    video_writer = None
    trim_count = 0    

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break 
            # High-quality resize to 448x448
            frame = high_quality_resize(frame, TARGET_SIZE)
            cv2.imshow("Video Trimmer (448x448 - High Quality)", frame)
        
        # Wait for keypress - normal delay if playing, infinite if paused
        key = cv2.waitKey(delay if not paused else 0) & 0xFF  
        
        # Start recording segment
        if key == ord('s'):
            if not recording:
                trim_count += 1
                output_file = os.path.join(
                    OUTPUT_FOLDER, 
                    f"{os.path.splitext(video_file)[0]}_normal_14_april_{trim_count}.mp4"
                )
                
                # Use H.264 codec for best quality
                # Try different codec options for compatibility
                fourcc_options = [
                    cv2.VideoWriter_fourcc(*'avc1'),  # H.264 (best quality)
                    cv2.VideoWriter_fourcc(*'H264'),  # H.264 alternative
                    cv2.VideoWriter_fourcc(*'X264'),  # H.264 alternative
                    cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 (fallback)
                ]
                
                video_writer = None
                for fourcc in fourcc_options:
                    video_writer = cv2.VideoWriter(
                        output_file, 
                        fourcc, 
                        fps, 
                        (TARGET_SIZE, TARGET_SIZE)
                    )
                    if video_writer.isOpened():
                        break
                
                if video_writer and video_writer.isOpened():
                    recording = True
                    print(f"Recording started: {output_file}")
                else:
                    print(f"ERROR: Could not initialize video writer for {output_file}")
                    video_writer = None

        # End recording segment
        elif key == ord('e'):
            if recording and video_writer:
                video_writer.release()
                recording = False
                print(f"Saved: {output_file}")

        # Fast-forward 5 seconds
        elif key == ord('f'):
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = min(current_frame + (fps * 5), total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"Fast-forwarded to frame: {new_frame}")

        # Rewind 5 seconds
        elif key == ord('p'):
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(current_frame - (fps * 5), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"Rewound to frame: {new_frame}")
        
        # Pause/Resume playback
        elif key == 32:  # Space bar
            paused = not paused
            print("Paused" if paused else "Resumed")

        # Quit the program
        elif key == ord('q'):
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            exit("User exited.")

        # Write frame to output if recording
        if recording and video_writer:
            video_writer.write(frame)

    # Clean up after each video
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

# Final message after processing all videos
print("Processing complete.")