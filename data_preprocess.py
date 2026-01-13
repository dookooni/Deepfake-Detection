import os
import cv2
import numpy as np
import torch
import multiprocessing as mp
from insightface.app import FaceAnalysis
import decord
from tqdm import tqdm
import json

DATA_ROOT = './train_data/Dataset/DFDC'  
SAVE_ROOT = './train_data/Metadata/DFDC'
FRAMES_PER_VIDEO = 10
NUM_GPUS = 3

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
# =================================================================

def get_face_info(detector, frame):
    faces = detector.get(frame)
    if not faces:
        return None, None, 0.0

    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return face.bbox.tolist(), face.kps.tolist(), float(face.det_score)

def process_chunk(gpu_id, file_list):
    """
    file_list 요소: (full_path, relative_dir, file_name, file_type)
    file_type: 'video' or 'image'
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    detector = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    detector.prepare(ctx_id=gpu_id, det_size=(640, 640))
    
    for full_path, rel_dir, fname, ftype in tqdm(file_list, desc=f"GPU {gpu_id}"):
        try:
            save_dir_path = os.path.join(SAVE_ROOT, rel_dir)
            os.makedirs(save_dir_path, exist_ok=True)
            json_path = os.path.join(save_dir_path, os.path.splitext(fname)[0] + ".json")
            
            if os.path.exists(json_path): continue

            metadata = {
                "file_path": full_path,
                "type": ftype,
                "data": [] # 프레임별 혹은 단일 이미지 정보가 담김
            }

            if ftype == 'video':
                vr = decord.VideoReader(full_path)
                total_frames = len(vr)
                metadata['meta'] = {
                    "total_frames": total_frames, 
                    "width": int(vr[0].shape[1]), 
                    "height": int(vr[0].shape[0])
                }
                
                step = max(1, total_frames // FRAMES_PER_VIDEO)
                
                for i in range(FRAMES_PER_VIDEO):
                    idx = min(i * step, total_frames - 1)
                    frame = vr[idx].asnumpy() # RGB
                    
                    bbox, landmarks, score = get_face_info(detector, frame)
                    
                    metadata["data"].append({
                        "frame_idx": int(idx),
                        "bbox": bbox,
                        "landmarks": landmarks,
                        "score": score
                    })

            elif ftype == 'image':
                frame = cv2.imread(full_path)
                if frame is None: continue
                
                h, w, _ = frame.shape
                metadata['meta'] = {"width": w, "height": h}
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                bbox, landmarks, score = get_face_info(detector, frame_rgb)
                
                metadata["data"].append({
                    "frame_idx": 0,
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "score": score
                })

            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            print(f"Error processing {full_path}: {e}")


def main():
    print("Checking Directory Structure...")
    
    all_files = []
    
    for root, dirs, files in os.walk(DATA_ROOT):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            full_path = os.path.join(root, fname)
            rel_dir = os.path.relpath(root, DATA_ROOT)
            
            if ext in VIDEO_EXTS:
                all_files.append((full_path, rel_dir, fname, 'video'))
            elif ext in IMAGE_EXTS:
                all_files.append((full_path, rel_dir, fname, 'image'))
    
    print(f"Total files found: {len(all_files)}")
    print("Initializing Models & Starting Processes...")

    FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=-1)
    
    chunks = np.array_split(all_files, NUM_GPUS)
    processes = []
    
    for i in range(NUM_GPUS):
        p = mp.Process(target=process_chunk, args=(i, chunks[i].tolist()))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("Done!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()