import os
import json


image_dir = ''
caption_dir = ''


image_folders = sorted(os.listdir(image_dir))
caption_files = sorted(os.listdir(caption_dir))


for folder_name, caption_file in zip(image_folders, caption_files):
    image_folder_path = os.path.join(image_dir, folder_name)
    caption_path = os.path.join(caption_dir, caption_file)

    
    frame_files = sorted(os.listdir(image_folder_path))
    frame_indices = [os.path.splitext(f)[0].lstrip('0') for f in frame_files]  
    frame_indices = set(frame_indices)

    
    with open(caption_path, 'r') as f:
        captions_dict = json.load(f)
    caption_indices = set(captions_dict.keys())

   
    missing_captions = frame_indices - caption_indices
    
    missing_frames = caption_indices - frame_indices

   
    if missing_captions or missing_frames:
        print(f"⚠️  {folder_name}:")
        if missing_captions:
            print(f"  ⛔  {len(missing_captions)} : {sorted(missing_captions)}")
        if missing_frames:
            print(f"  ⛔  {len(missing_frames)} : {sorted(missing_frames)}")
    else:
        print(f"✅  {folder_name}")

print("\n！")
