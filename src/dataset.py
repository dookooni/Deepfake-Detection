import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import decord

decord.bridge.set_bridge('torch')

class Celeb_DF(Dataset):
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


class Celeb_DF_FaceCrop(Dataset):
    def __init__(self, video_list, root_dir, transform=None, frames_per_video=10):
        """
        Args:
            video_list: [(relative_v_path, folder_name), ...] 형태의 리스트
            root_dir: 'preprocessed_crops' 폴더의 절대 경로
            transform: 이미지 변환 (Compose)
            frames_per_video: 비디오당 추출한 프레임 수
        """
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
            
            for i in range(frames_per_video):
                img_path = os.path.join(v_full_path, f"{i}.jpg")
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
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
                    print(f"Video not found: {video_path}")
                    continue

                with open(meta_path, "r") as f:
                    meta_data = json.load(f)

                frames = meta_data.get('frames')
                for frame in frames:
                    frame_idx = frame.get('frame_idx')
                    bbox = frame.get('bbox')
                    landmarks = frame.get('landmarks')
                
                    # (str, int, list[float], list[list[float]], int)
                    self.samples.append((video_path, frame_idx, bbox, landmarks, label))

    def __getitem__(self, idx):
        video_path, frame_idx, bbox, landmarks, label = self.samples[idx]
        vr = decord.VideoReader(video_path, ctx=cpu(0))
        frame = vr[frame_idx]
        image = frame.asnumpy()

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class WildDeepfake(Dataset):
    def __init__(self, root_dir, split='train', transform=None, max_frames=10):
        """
        Args:
            root_dir: 데이터셋 루트 (예: ./WildDeepfake)
            split: 'train' 또는 'test' (real_train/fake_train 폴더 결정)
            transform: 전처리
            max_frames: 시퀀스(폴더) 당 사용할 최대 프레임 수 (None이면 전체 다 사용)
                        * FaceForensics와 균형을 맞추려면 10~20 정도 추천
        """
        self.transform = transform
        self.samples = []
        
        target_folders = {
            f'real_{split}': 0,
            f'fake_{split}': 1
        }

        print(f"Loading WildDeepfake ({split}) with max_frames={max_frames}...")

        for folder_name, label in target_folders.items():
            base_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(base_path):
                print(f"Warning: {base_path} not found.")
                continue
            
            for root, dirs, files in os.walk(base_path):
                image_files = [f for f in files if f.lower().endswith('.png')]
                
                if len(image_files) == 0:
                    continue
                
                image_files.sort()
                
                if max_frames is not None and len(image_files) > max_frames:
                    indices = np.linspace(0, len(image_files) - 1, max_frames, dtype=int)
                    selected_files = [image_files[i] for i in indices]
                else:
                    selected_files = image_files
                
                # metadata 처리 부분
                meta_root = root.replace("Dataset", "Metadata")
                for file in selected_files:
                    img_path = os.path.join(root, file)
                    json_filename = os.path.splitext(file)[0] + ".json"
                    meta_path = os.path.join(meta_root, json_filename)

                    bbox = None
                    landmarks = None

                    if os.path.exists(json_path):
                        try:
                            with open(meta_path, "r") as f:
                                meta_data = json.load(f)

                            bbox = meta_data.get("bbox")    
                            landmarks = meta_data.get("landmarks")
                            

                for file in selected_files:
                    self.samples.append((os.path.join(root, file), label))

        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드 (OpenCV)
        # WildDeepfake 이미지는 간혹 깨진 파일이 있을 수 있어 예외처리 권장
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise IOError("Image load failed")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # 학습 중 멈추지 않도록 에러 출력 후 대체 데이터(0번 인덱스 등) 리턴하거나 
            # 여기서는 에러를 띄우고 DataLoader에서 collate_fn으로 처리하는 방법 등이 있음.
            print(f"Error loading {img_path}: {e}")
            # 임시 방편: 에러 시 0번 데이터 리턴 (혹은 랜덤)
            return self.__getitem__(0)

        # Transform 적용
        if self.transform:
            # Albumentations:
            image = self.transform(image=image)['image']
            # Torchvision:
            # image = self.transform(image)

        return image, label