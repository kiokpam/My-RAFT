import cv2
import os
import gc
import glob
import subprocess
import torch
def extract_frames(video_path, output_folder, num_frames=15):
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_interval = max(total_frames // num_frames, 1)
    frame_count = 0
    saved_frames = 0

    while cap.isOpened() and saved_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{saved_frames:04d}.jpg"), frame)
            saved_frames += 1
        frame_count += 1

    cap.release()
def extract_flow(dataset, output_rgb_folder, output_flow_folder, classname, start_index, num_frames=15):
    video_folder = dataset.format(classname)
    rgb_folder = output_rgb_folder.format(classname)
    flow_folder = output_flow_folder.format(classname)
    video_files = glob.glob(os.path.join(video_folder, '*.mp4'))  # Hoặc bất kỳ định dạng video nào bạn sử dụng
    for video_file in video_files[start_index:]:
        print(f"Processing {video_file}")
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_rgb_folder = os.path.join(rgb_folder, video_name)
        output_flow_folder = os.path.join(flow_folder, video_name)
        extract_frames(video_file, output_rgb_folder, num_frames)
        if not os.path.exists(output_flow_folder):
            os.makedirs(output_flow_folder)
        subprocess.run(["python", "demo.py", "--model", "models/raft-small.pth", "--path", output_rgb_folder, "--output_path", output_flow_folder, "--alternate_corr", "--small"])
        torch.cuda.empty_cache()
        gc.collect()
if __name__ == '__main__':
    dataset = "E:\FPTUni\SUMMER24\DPL\Real_Life_Violence_Dataset\{}"
    output_rgb_folder = r"E:\FPTUni\SUMMER24\DPL\Frames\train\{}\rgb"
    output_flow_folder = r"E:\FPTUni\SUMMER24\DPL\Frames\train\{}\flow"
    extract_flow(dataset, output_rgb_folder, output_flow_folder, "Violence", 1050)
    extract_flow(dataset, output_rgb_folder, output_flow_folder, "NonViolence", 0)
