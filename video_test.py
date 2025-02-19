import cv2
import numpy as np
import scipy.io
import os
from rtmlib import Wholebody, draw_skeleton
from rtmlib import BodyWithFeet

# Load .mat file
mat_data = scipy.io.loadmat('YouTube_Pose_dataset.mat')
dataset = mat_data['data'].squeeze()

# Path to extracted frames
GT_FRAMES_DIR = './GT_frames/'

def process_youtube_pose():
    device = 'cpu'  
    backend = 'onnxruntime'
    openpose_skeleton = False

    # Initialize model
    wholebody = Wholebody(to_openpose=openpose_skeleton, mode='balanced', backend=backend, device=device)
    bodywithfeet = BodyWithFeet(backend=backend, device=device)


    output_dir = './vids/'
    os.makedirs(output_dir, exist_ok=True)

    for video_idx, video_data in enumerate(dataset):
        video_name = video_data['videoname'][0]  # Extract video name
        frame_ids = video_data['frameids'][0]  # List of frame indices
        gt_locs = video_data['locs']  # Ground-truth keypoints (2,7,100)
        video_url = video_data['url'][0]
        print(f"Video URL: {video_url}")

        for frame_idx, frame_id in enumerate(frame_ids):
            # Construct image filename
            img_path = os.path.join(GT_FRAMES_DIR, f"{video_name}/frame_00{frame_id}.jpg")

            if not os.path.exists(img_path):
                #print(f"Frame missing: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                #print(f"Failed to load {img_path}")
                continue

            # Run keypoint detection
            keypoints, scores = bodywithfeet(img)

            # Extract ground-truth keypoints for this frame
            gt_keypoints = gt_locs[:, :, frame_idx].T  # Convert (2,7) → (7,2)
            gt_keypoints = gt_keypoints.astype(int)

            # Draw skeleton
            img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)

            # Overlay ground-truth keypoints (Green circles)
            for x, y in gt_keypoints:
                cv2.circle(img_show, (x, y), 4, (0, 255, 0), -1)  # Green for GT

            # Save the image
            output_path = os.path.join(output_dir, f"{video_name}_frame_00{frame_id}.png")
            cv2.imwrite(output_path, img_show)
            #print(f"Saved: {output_path}")
        
        # only process one video for testing right now!
        break

if __name__ == '__main__':
    process_youtube_pose()
