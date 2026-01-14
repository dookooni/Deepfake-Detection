import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split
import random
from PIL import Image
from insightface.app import FaceAnalysis
import decord

class RandomJPEGCompression:
    def __init__(self, quality_range=(40, 90), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            import io
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=quality)
            img = Image.open(buffer)
        return img

class UniversalDetector:
    def __init__(self, gpu_id=0):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=gpu_id, det_size=(640, 640))

    def detect(self, input_path, num_frames=1):
        """
        input_path: .jpg, .png, .mp4 등
        num_frames: 영상일 경우 추출할 프레임 수
        """
        ext = os.path.splitext(input_path)[-1].lower()
        results = []

        if ext in ['.mp4', '.avi', '.mov']:
            vr = decord.VideoReader(input_path) # Video 읽기
            total_frames = len(vr) # Video의 총 프레임 수 
            step = max(1, total_frames // num_frames) 
            for i in range(num_frames):
                idx = min(i * step, total_frames - 1)
                frame = vr[idx].asnumpy() # vr[idx] = [H, W, C] (FaceAnalysis를 활용하기 위해 Numpy 배열로 변환)
                results.append(self._get_face_info(frame, idx))

        elif ext in ['.jpg', '.jpeg', '.png']:
            frame = cv2.imread(input_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results.append(self._get_face_info(frame, 0))

        return results

    def _get_face_info(self, frame, idx):
        faces = self.app.get(frame)
        if not faces:
            return {"frame_idx": idx, "bbox": None}
        
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        return {
            "frame_idx": idx,
            "bbox": face.bbox.tolist(),
            "landmarks": face.kps.tolist(),
            "score": float(face.det_score)
        }

def get_llrd_params(model, base_lr, decay_rate=0.75):
    n_blocks = len(model.backbone.blocks)
    param_groups = []
    
    seen_names = set()

    head_params = []
    for name, param in model.named_parameters():
        if ("head" in name or "backbone.norm" in name) and name not in seen_names:
            head_params.append(param)
            seen_names.add(name)
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    for i in range(n_blocks - 1, -1, -1):
        lr = base_lr * (decay_rate ** (n_blocks - i))
        block_params = []
        for name, param in model.named_parameters():
            if f"blocks.{i}." in name and name not in seen_names:
                block_params.append(param)
                seen_names.add(name)
        if block_params:
            param_groups.append({"params": block_params, "lr": lr})
    
    rest_params = []
    for name, param in model.named_parameters():
        if name not in seen_names:
            rest_params.append(param)
            seen_names.add(name)
    if rest_params:
        param_groups.append({
            "params": rest_params, 
            "lr": base_lr * (decay_rate ** (n_blocks + 1))
        })

    return param_groups

def split_faceforensics(root_dir, test_ratio=0.2):
    video_ids = set()

    data_root = os.path.join(root_dir, 'Dataset', 'FaceForensics++_C23')
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mp4'):
                video_id = os.path.splitext(file)[0]
                video_ids.add(video_id)

    train_ids, val_ids = train_test_split(sorted(list(video_ids)), test_size=test_ratio, random_state=42)

    return train_ids, val_ids

def split_celeb_df(root_dir, test_ratio=0.2):
    crop_root = os.path.join(root_dir, "Dataset", "celeb_df")

    all_videos = []
    all_groups = []

    for folder_name in os.listdir(crop_root):
        folder_path = os.path.join(crop_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        video_names = sorted(os.listdir(folder_path))
        for vid in video_names:
            if vid.startswith("."): continue
        
            group_id = None
            if folder_name == "YouTube-real":
                group_id = f"yt_{vid}"
            else:
                group_id = re.split(r'[_-]', vid)[0]

            video_rel_path = os.path.join(folder_name, vid)
            all_videos.append((video_rel_path, folder_name))
            all_groups.append(group_id)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, eval_idx = next(gss.split(all_videos, groups=all_groups))

    crop_train = [all_videos[i] for i in train_idx]
    crop_eval = [all_videos[i] for i in eval_idx]

    train_groups = set([all_groups[i] for i in train_idx])
    eval_groups = set([all_groups[i] for i in eval_idx])
    intersection = train_groups.intersection(eval_groups)

    print(f"Total Videos: {len(all_videos)}")
    print(f"Train Videos: {len(crop_train)} | Eval Videos: {len(crop_eval)}")
    
    if len(intersection) > 0:
        print(f"[CRITICAL WARNING] ID Leakage Detected! {list(intersection)[:5]}...")
    else:
        print("[SUCCESS] Identity-Disjoint Split Completed (인물 기준 완벽 분리됨).")

    return train_list, eval_list