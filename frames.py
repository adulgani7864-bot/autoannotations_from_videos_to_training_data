import cv2
import os
import glob

def extract_frames_from_videos(source_folders, output_root):
    """
    Extract frames from MP4 videos while maintaining original resolution and quality
    
    Args:
        source_folders (list): List of source folder paths
        output_root (str): Root directory for output frames
    """
    
    # Create output root directory if it doesn't exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory: {output_root}")
    
    # Process each source folder
    for source_folder in source_folders:
        if not os.path.exists(source_folder):
            print(f"Source folder not found: {source_folder}")
            continue
            
        print(f"\nProcessing folder: {source_folder}")
        
        # Find all MP4 files in the source folder
        mp4_pattern = os.path.join(source_folder, "*.mp4")
        video_files = glob.glob(mp4_pattern)
        
        if not video_files:
            print(f"No MP4 files found in {source_folder}")
            continue
            
        print(f"Found {len(video_files)} MP4 files")
        
        # Process each video file
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Create video-specific output folder
            output_folder = os.path.join(output_root, video_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            print(f"Extracting frames from: {video_name}")
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error: Cannot open video file {video_path}")
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
            
            frame_count = 0
            success = True
            
            # Read and save frames
            while success:
                success, frame = cap.read()
                
                if success:
                    frame_count += 1
                    
                    # Generate frame filename with 5-digit numbering
                    frame_filename = f"{frame_count:05d}.jpg"
                    frame_path = os.path.join(output_folder, frame_filename)
                    
                    # Save frame with high quality (95% JPEG quality)
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Progress indicator
                    if frame_count % 100 == 0:
                        print(f"Extracted {frame_count}/{total_frames} frames")
            
            # Release video capture
            cap.release()
            
            print(f"Completed: {video_name} - {frame_count} frames extracted to {output_folder}")
    
    print("\nFrame extraction completed!")

def main():
    # Define source folders and output directory
    source_folders = [
        "trimmed_videos"
    ]
    
    output_root = "DW_Frame"
    
    # Extract frames from all videos
    extract_frames_from_videos(source_folders, output_root)

if __name__ == "__main__":
    main()