import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import cv2
from collections import Counter
try:
    import decord
    from decord import cpu
    decord.bridge.set_bridge('torch')
except ImportError:
    decord = None
    def cpu(id): return None
    print("Warning: decord not found. Video-based datasets might fail.")

class Celeb_DF_(Dataset):
    def __init__(self, root, transform=None, frames_per_video=10):
        self.root = root
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        self.folder_map = {
            'Celeb-real': 0,
            'YouTube-real': 0,
            'Celeb-synthesis': 1
        }
        
        self.samples = []
        print(f"Scanning for videos in {root}...")
        
        for folder_name, label in self.folder_map.items(): 
            folder_path = os.path.join(root, folder_name)
            if not os.path.exists(folder_path): continue
            
            fnames = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
            for fname in fnames:
                video_path = os.path.join(folder_path, fname)
                for i in range(self.frames_per_video):
                    self.samples.append({
                        'path': video_path,
                        'frame_order': i, 
                        'label': label
                    })
        
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['path']
        frame_order = sample['frame_order']
        label = sample['label']

        try:
            if decord is None:
                raise ImportError("decord not found")
            
            vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)
            
            step = max(1, total_frames // self.frames_per_video)
            target_frame_idx = min(frame_order * step, total_frames - 1)
            
            frame = vr[target_frame_idx]
            
            image = Image.fromarray(frame.numpy())
            
        except Exception as e:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        return image, label


class Celeb_DF(Dataset):
    def __init__(self, video_list, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        self.label_map = {
            'Celeb-real': 0,
            'YouTube-real': 0,
            'Celeb-synthesis': 1
        }

        for rel_v_path, folder_name in video_list:
            label = self.label_map.get(folder_name, 0)
            v_full_path = os.path.join(self.root_dir, rel_v_path)

            # Meta data
            fol = rel_v_path.split('/')[1]
            fol = os.path.splitext(fol)[0]
            meta_path = os.path.join(self.root_dir, rel_v_path.split('/')[0], fol, fol + '.json')
            meta_path = meta_path.replace("Dataset", "Metadata")
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            frames = meta.get("frames")
            for frame in frames:
                frame_idx = frame.get("frame_idx")
                landmarks = frame.get("landmarks")
                bbox = frame.get("bbox")

                self.samples.append({
                    'path': v_full_path,
                    'frame_idx': frame_idx,
                    'bbox': bbox,
                    'landmarks': landmarks,
                    'label': label,
                    'type': 'video'
                })

    def __len__(self):
        return len(self.samples)

    def _count_labels(self):
        label_count = Counter([sample['label'] for sample in self.samples])
        return label_count

    def _get_rotation_matrix(self, landmarks):
        """ 랜드마크(눈)를 기준으로 회전 매트릭스를 계산하는 함수 """
        if landmarks is None or len(landmarks) < 2:
            return None
        
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        return M

    def _rotate_bbox(self, bbox, M, img_w, img_h):
        if bbox is None: return None
        
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ])
        
        transformed_corners = corners @ M.T 
        
        tx1 = np.min(transformed_corners[:, 0])
        ty1 = np.min(transformed_corners[:, 1])
        tx2 = np.max(transformed_corners[:, 0])
        ty2 = np.max(transformed_corners[:, 1])
        
        tx1 = max(0, tx1)
        ty1 = max(0, ty1)
        tx2 = min(img_w, tx2)
        ty2 = min(img_h, ty2)
        
        return [tx1, ty1, tx2, ty2]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample['path']
        frame_idx = sample['frame_idx']
        bbox = sample['bbox']
        landmarks = sample['landmarks']
        label = sample['label']
        
        try:
            if decord is None:
                raise ImportError("decord not found")
                
            vr = decord.VideoReader(file_path, ctx=cpu(0))
            frame = vr[frame_idx].detach().cpu().numpy() # [H, W, 3] numpy array
            h, w, _ = frame.shape
            
            if landmarks is not None:
                M = self._get_rotation_matrix(landmarks)
                if M is not None:
                    frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
                    
                    if bbox is not None:
                        bbox = self._rotate_bbox(bbox, M, w, h)

            scales = [1.2, 1.5, 2.0]
            probs = [0.2, 0.6, 0.2]
            scale = np.random.choice(scales, p=probs)

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                
                center_x = (x2 - x1) // 2 + x1
                center_y = (y2 - y1) // 2 + y1
                new_center_w = (x2 - x1) * scale
                new_center_h = (y2 - y1) * scale
                
                x1 = int(center_x - new_center_w // 2)
                y1 = int(center_y - new_center_h // 2)
                x2 = int(center_x + new_center_w // 2)
                y2 = int(center_y + new_center_h // 2)

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    frame = frame[y1:y2, x1:x2]
            
            image = Image.fromarray(frame)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class FaceForensics(Dataset):
    def __init__(self, root_dir, video_ids, transform=None):
        self.transform = transform
        self.samples = []

        self.data_root = os.path.join(root_dir, "Dataset", "FaceForensics++_C23")
        self.meta_root = os.path.join(root_dir, "Metadata", "FaceForensics++_C23")

        for root, dirs, files in os.walk(self.meta_root):
            for file in files:
                if not file.endswith('.json'): continue
                
                meta_path = os.path.join(root, file) # json 파일의 전체 경로
                label = 0 if 'original' in meta_path else 1

                video_id = os.path.splitext(file)[0]
                if video_ids is not None and video_id not in video_ids:
                    continue

                parent_dir_of_json = os.path.dirname(root)
                rel_path = os.path.relpath(parent_dir_of_json, self.meta_root)
                video_path = os.path.join(self.data_root, rel_path, f"{video_id}.mp4")
                
                if not os.path.exists(video_path): 
                    continue

                with open(meta_path, "r") as f:
                    meta_data = json.load(f)

                frames = meta_data.get('frames')
                for frame in frames:
                    frame_idx = frame.get('frame_idx')
                    bbox = frame.get('bbox')
                    landmarks = frame.get('landmarks')
                
                    self.samples.append({
                        'path': video_path,
                        'frame_idx': frame_idx,
                        'bbox': bbox,
                        'landmarks': landmarks,
                        'label': label,
                        'type': 'video'
                    })

    def __len__(self):
        return len(self.samples)

    def _count_labels(self):
        label_count = Counter([sample['label'] for sample in self.samples])
        return label_count

    def _get_rotation_matrix(self, landmarks):
        """ 랜드마크(눈)를 기준으로 회전 매트릭스를 계산하는 함수 """
        if landmarks is None or len(landmarks) < 2:
            return None
        
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        return M

    def _rotate_bbox(self, bbox, M, img_w, img_h):
        if bbox is None: return None
        
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ])
        
        transformed_corners = corners @ M.T 
        
        tx1 = np.min(transformed_corners[:, 0])
        ty1 = np.min(transformed_corners[:, 1])
        tx2 = np.max(transformed_corners[:, 0])
        ty2 = np.max(transformed_corners[:, 1])
        
        tx1 = max(0, tx1)
        ty1 = max(0, ty1)
        tx2 = min(img_w, tx2)
        ty2 = min(img_h, ty2)
        
        return [tx1, ty1, tx2, ty2]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample['path']
        frame_idx = sample['frame_idx']
        bbox = sample['bbox']
        landmarks = sample['landmarks']
        label = sample['label']
        
        try:
            if decord is None:
                raise ImportError("decord not found")
                
            vr = decord.VideoReader(file_path, ctx=cpu(0))
            frame = vr[frame_idx].detach().cpu().numpy() # [H, W, 3] numpy array
            h, w, _ = frame.shape
            
            if landmarks is not None:
                M = self._get_rotation_matrix(landmarks)
                if M is not None:
                    frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
                    
                    if bbox is not None:
                        bbox = self._rotate_bbox(bbox, M, w, h)

            scales = [1.2, 1.5, 2.0]
            probs = [0.2, 0.6, 0.2]
            scale = np.random.choice(scales, p=probs)

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                
                center_x = (x2 - x1) // 2 + x1
                center_y = (y2 - y1) // 2 + y1
                new_center_w = (x2 - x1) * scale
                new_center_h = (y2 - y1) * scale
                
                x1 = int(center_x - new_center_w // 2)
                y1 = int(center_y - new_center_h // 2)
                x2 = int(center_x + new_center_w // 2)
                y2 = int(center_y + new_center_h // 2)

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    frame = frame[y1:y2, x1:x2]
            
            image = Image.fromarray(frame)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DFDC(Dataset):
    def __init__(self, root_dir, video_ids, transform=None):
        self.transform = transform
        self.video_ids = video_ids
        self.samples = []
        
        self.meta_root = os.path.join(root_dir, "Metadata", "DFDC")
        if video_ids is not None:
            self.allowed_ids = {os.path.splitext(item['path'].split('/')[-1])[0] for item in video_ids}
        else:
            self.allowed_ids = None
        
        print(f"Loading DFDC from {self.meta_root}...")
        
        if not os.path.exists(self.meta_root):
             print(f"Warning: {self.meta_root} not found.")

        for root, dirs, files in os.walk(self.meta_root):
            for file in files:
                if not file.endswith('.json'): continue
                
                meta_path = os.path.join(root, file)
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                except:
                    continue

                file_path_rel = meta.get('file_path')
                if not file_path_rel: continue
                ids = os.path.splitext(file_path_rel.split('/')[-1])[0]
                if self.allowed_ids is not None:
                    if ids not in self.allowed_ids:
                        continue

                label = 0
                if 'fake' in root.lower() or ('fake' in meta.get('file_path', '').lower()):
                    label = 1
                
                file_type = meta.get('type', 'image')
                file_path_rel = meta.get('file_path')
                
                if not file_path_rel:
                    continue

                for item in meta.get('data', []):
                    self.samples.append({
                        'path': file_path_rel,
                        'type': file_type,
                        'bbox': item.get('bbox'),
                        'landmarks': item.get('landmarks'),
                        'label': label,
                        'frame_idx': item.get('frame_idx', 0)
                    })

        print(f"DFDC: Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def _count_labels(self):
        label_count = Counter([sample['label'] for sample in self.samples])
        return label_count

    def _get_rotation_matrix(self, landmarks):
        """ 랜드마크(눈)를 기준으로 회전 매트릭스를 계산하는 함수 """
        if landmarks is None or len(landmarks) < 2:
            return None
        
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        return M

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample['path']
        file_type = sample['type']
        label = sample['label']
        bbox = sample['bbox']
        landmarks = sample['landmarks']
        
        image = None
        
        try:
            frame = None
            if file_type == 'image':
                frame = cv2.imread(file_path)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            elif file_type == 'video':
                if decord is None:
                    raise ImportError("decord not found")
                vr = decord.VideoReader(file_path, num_threads=1)
                frame_idx = sample['frame_idx']
                if frame_idx >= len(vr): frame_idx = len(vr) - 1
                frame = vr[frame_idx].asnumpy() # RGB

            if frame is not None:
                h, w, _ = frame.shape
                
                # Alignment (Rotation only, No Cropping as requested)
                if landmarks is not None:
                    M = self._get_rotation_matrix(landmarks)
                    if M is not None:
                        frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

                image = Image.fromarray(frame)
            else:
                 image = Image.new('RGB', (224, 224), (0, 0, 0))

        except Exception as e:
            # print(f"Error: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


class WildDeepfake(Dataset):
    def __init__(self, root_dir, video_ids, transform=None):
        self.transform = transform
        self.samples = []
        
        base_dataset_path = os.path.join(root_dir, "Dataset", "WildDeepfake", "deepfake_in_the_wild")
        base_metadata_path = os.path.join(root_dir, "Metadata", "WildDeepfake", "deepfake_in_the_wild")

        print(f"Loading WildDeepfake from {base_dataset_path}...")
        
        if video_ids is not None:
            self.allowed_paths = {os.path.abspath(item['path']) for item in video_ids}
        else:
            self.allowed_paths = None
        
        for root, dirs, files in os.walk(base_dataset_path):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                
                img_path = os.path.join(root, file)
                
                if self.allowed_paths is not None:
                    if os.path.abspath(img_path) not in self.allowed_paths:
                        continue

                label = 1
                if 'real' in root.lower():
                    label = 0
                
                rel_path = os.path.relpath(root, base_dataset_path)
                meta_dir = os.path.join(base_metadata_path, rel_path)
                json_name = os.path.splitext(file)[0] + ".json"
                meta_path = os.path.join(meta_dir, json_name)
                
                bbox = None
                landmarks = None
                
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            bbox = meta.get('bbox')
                            landmarks = meta.get('landmarks')
                    except:
                        pass
                
                self.samples.append({
                    'path': img_path,
                    'bbox': bbox,
                    'landmarks': landmarks,
                    'label': label
                })

        print(f"WildDeepfake: Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def _count_labels(self):
        label_count = Counter([sample['label'] for sample in self.samples])
        return label_count

    def _get_rotation_matrix(self, landmarks):
        """ 랜드마크(눈)를 기준으로 회전 매트릭스를 계산하는 함수 """
        if landmarks is None or len(landmarks) < 2:
            return None
        
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        return M

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        label = sample['label']
        bbox = sample['bbox']
        landmarks = sample['landmarks']
        
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                 raise FileNotFoundError(f"Image not found: {img_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            if landmarks is not None:
                M = self._get_rotation_matrix(landmarks)
                if M is not None:
                    frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

            image = Image.fromarray(frame)
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label