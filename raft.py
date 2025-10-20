import os
import cv2
import scipy.io
import numpy as np
from tqdm import tqdm


base_img_dir = ""
base_mat_dir = ""
save_base_dir = ""
os.makedirs(save_base_dir, exist_ok=True)

for video_idx in range(1, 22):  
    folder_name = f"{video_idx:02d}"
    img_dir = os.path.join(base_img_dir, folder_name)
    mat_path = os.path.join(base_mat_dir, f"{video_idx}_label.mat")
    save_dir = os.path.join(save_base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    
    mat = scipy.io.loadmat(mat_path)
    volLabel = mat["volLabel"][0]  

    
    frame_names = sorted(f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png")))

    for idx, fname in tqdm(enumerate(frame_names), total=len(frame_names), desc=f"Video {video_idx:02d}"):
        frame_path = os.path.join(img_dir, fname)
        frame = cv2.imread(frame_path)

        mask_idx = idx + 1
        if mask_idx >= len(volLabel):
            break

        mask = volLabel[mask_idx]
        if isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = mask.astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 5 and h > 5:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        save_path = os.path.join(save_dir, fname)
        cv2.imwrite(save_path, frame)

print("✅ :", save_base_dir)
