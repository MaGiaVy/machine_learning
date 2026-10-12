# Kế hoạch Hoàn chỉnh: Hệ thống Phân loại 37 giống Chó Mèo & Phân tích Lỗi (Error Analysis)

> **Tài liệu Kế hoạch Kỹ thuật Chuẩn mực (Specification & Implementation Plan)**  
> **Phiên bản:** 2.1 (Tập trung chuyên sâu vào quy trình Phân tích Lỗi - Error Analysis với chú thích tiếng Việt chi tiết)  
> **Nguyên tắc thực thi:** Toàn bộ quy trình và mã nguồn được tích hợp trong **01 file Notebook duy nhất (`pet_error_analysis.ipynb`)**, lưu trữ file phẳng tại thư mục làm việc, tương thích linh hoạt giữa Google Colab và Local (Dual-mode).

---

## 1. Tổng quan Dự án (Project Overview)

### 1.1. Mục tiêu (Goals)
1. Xây dựng hệ thống phân loại 37 giống chó mèo trên tập dữ liệu Oxford-IIIT Pet Dataset.
2. Áp dụng kiến trúc Pipeline 2 bước:
   - **Bước A (Detector):** Sử dụng YOLOv8-nano (pretrained COCO) phát hiện vị trí con vật (cat/dog), cắt vùng bounding box (crop) có lề mở rộng (margin).
   - **Bước B (Classifier):** Sử dụng ResNet18 (pretrained ImageNet1k) fine-tune để phân loại chính xác 37 giống loài.
3. Thực hiện đầy đủ quy trình **Phân tích lỗi (Error Analysis)** theo chuẩn phương pháp luận Machine Learning của Andrew Ng:
   - Phân chia 4 tập dữ liệu: Train, Eyeball Dev, Blackbox Dev, Test.
   - Phân tích phân bổ lỗi theo Pipeline (Bước A gây lỗi hay Bước B gây lỗi).
   - Phân tích trần tiềm năng cải thiện (Ceiling Analysis) cho từng nhóm lỗi.
   - Cải tiến có mục tiêu (Targeted Improvement) cho Bước A hoặc Bước B và đo lường độ suy giảm khoảng cách Overfitting.
   - Đánh giá độc lập đúng 1 lần trên tập Test.

### 1.2. Tech Stack & Môi trường Dual-Mode
- **Ngôn ngữ & Framework:** Python 3, PyTorch (AMP float16, Cosine Annealing, AdamW), torchvision.
- **Phát hiện đối tượng:** `ultralytics` (YOLOv8n).
- **Học máy & Đo lường:** `scikit-learn` (Stratified split, Accuracy, Macro F1, Confusion Matrix).
- **Xử lý ảnh & Trực quan hóa:** OpenCV (headless), NumPy, Matplotlib, PIL, `ipywidgets` (giao diện gán nhãn tương tác).
- **Cơ chế Dual-mode:** Mã nguồn tự động phát hiện môi trường:
  - Nếu chạy trên **Google Colab:** tự động mount Google Drive và lưu trữ tại `/content/drive/MyDrive/PetErrorAnalysis` (hoặc `/content` nếu chưa mount).
  - Nếu chạy trên **Local (Windows/Linux):** lưu trữ trực tiếp tại thư mục hiện hành của dự án (`./`).
- **Cấu trúc lưu trữ phẳng (Flat Structure):** Tất cả các checkpoint (`.pt`), kết quả đo (`.json`, `.csv`), và hình ảnh (`.png`) được lưu trực tiếp tại thư mục gốc của dự án để tránh phụ thuộc đường dẫn thư mục lồng nhau.

---

## 2. Bảng Đối chiếu & Sửa lỗi Kỹ thuật (Bug Audit Table)

Dưới đây là các lỗi nghiêm trọng trong bản nháp cũ đã được sửa đổi triệt để trong bản kế hoạch này:

| STT | Vị trí (Task cũ) | Lỗi logic / Thiếu sót trong bản cũ | Giải pháp sửa đổi trong Bản v2.1 |
|:---:|:---|:---|:---|
| 1 | **Task 1** (Tách split) | `black_idx, train_idx = train_test_split(..., test_size=0.15)` khiến `black_idx` nhận 85% (~2.873 mẫu) còn `train_idx` chỉ nhận 15% (~507 mẫu). Đảo ngược hoàn toàn dữ liệu train! | Sửa thành: `train_idx, black_idx = train_test_split(rest, test_size=0.15, ...)` đảm bảo Train = 2.873 mẫu và Blackbox Dev = 507 mẫu. |
| 2 | **Task 2** (Oracle Bbox) | Nếu trimap bị lỗi/rỗng, hàm trả về `None`, lệnh `crop_with_margin` sẽ văng ngoại lệ `TypeError: cannot unpack NoneType`. | Thêm cơ chế Fallback: nếu mask trimap rỗng, tự động trả về `(0, 0, W, H)` (toàn bộ khung hình) kèm cờ cảnh báo. |
| 3 | **Task 4** (Vòng lặp Train) | `train_model` nhận `val_ds` nhưng trong code không bao giờ đánh giá trên tập val; chỉ chạy hết 15 epoch rồi trả về model cuối (dễ bị overfit, không có best checkpoint). | Xây dựng vòng lặp validation đầy đủ sau mỗi epoch, ghi nhận lịch sử train_loss, val_loss, val_f1 và **tự động lưu `best_model.pt`** có Macro F1 cao nhất trên Blackbox Dev. |
| 4 | **Task 5, 6, 7** (Biến toàn cục) | Dùng các biến `yolo_model`, `oracle_model` trong Task 6, 7 nhưng trong Task 5 chỉ gán đè vào biến tạm `m`. Chạy nối tiếp sẽ gặp lỗi `NameError`. | Đặt tên biến tường minh (`model_oracle`, `model_yolo`), có hàm `save_checkpoint` và `load_checkpoint` an toàn từ đĩa. |
| 5 | **Task 6** (Gán nhãn thủ công) | Yêu cầu mở CSV bằng Excel rồi tự điền tay 100 dòng khiến quy trình bị đứt đoạn, tốn hàng giờ đồng hồ. | Cung cấp **Interactive Widget (`ipywidgets`)** hiển thị ảnh + checkbox nhãn ngay trong notebook (bấm 5-10 phút là xong), đồng thời có cờ **Quick Auto-mock** để chạy test luồng tự động khi cần. |
| 6 | **Task 7 & 9** (Hướng can thiệp) | Task 7 phát hiện nguyên nhân lỗi nhưng Task 9 chỉ fix cứng Data Augmentation cho Bước B mà bỏ qua Bước A. | Thêm nhánh điều kiện logic: Nếu Bước A chiếm tỷ lệ lỗi cao (>30-40%), ưu tiên tinh chỉnh Bước A (mở rộng crop margin, hạ ngưỡng conf). Nếu Bước B chiếm ưu thế, áp dụng Targeted Augmentation. |
| 7 | **Task 10** (Tải Test set) | Vòng lặp trích xuất nhãn `test_labels = [ds[i]...]` đọc từng file ảnh từ ổ cứng gây nghẽn I/O. | Sử dụng trực tiếp thuộc tính có sẵn `test_ds._labels` của torchvision dataset mà không cần tải lại toàn bộ ảnh. |

---

## 3. Chi tiết 12 Tasks Thực hiện (Tasks Breakdown)

Toàn bộ 12 Tasks này sẽ được biên soạn thành các cell tuần tự trong notebook `pet_error_analysis.ipynb`.

```
pet_error_analysis.ipynb
├── Task 0: Thiết lập Môi trường Dual-Mode & Kiểm tra GPU
├── Task 1: Tải Oxford-IIIT Pet & Phân chia 4 Tập Chuẩn hóa
├── Task 2: Trích xuất Bounding Box Chuẩn (Oracle) từ Trimap
├── Task 3: Bước A - Xây dựng Detector YOLOv8-nano & Đo IoU
├── Task 4: Bước B - Xây dựng Classifier ResNet18 & Vòng lặp Train Chuẩn
├── Task 5: Huấn luyện Baseline (Model O vs Model Y) & So sánh Đầu vào [Bắt đầu nền tảng Phân tích Lỗi]
├── Task 6: Bảng Phân tích Lỗi trên Eyeball Dev (Gắn nhãn Tương tác & Giải thích Chuyên sâu)
├── Task 7: Phân tích Phân bổ Lỗi theo Pipeline (Pipeline Error Attribution - Bước A vs Bước B)
├── Task 8: Phân tích Trần Cải thiện (Ceiling Analysis - Đo lường Giá trị Kinh tế của việc Sửa Lỗi)
├── Task 9: Cải tiến có Mục tiêu & Huấn luyện Model v2 (Đặc trị từng Nhóm Bệnh)
├── Task 10: Kiểm tra Overfitting (Eyeball vs Blackbox Dev Gap) & Đánh giá Cuối cùng trên Test
└── Task 11: Tổng hợp Báo cáo Khoa học & Phân tích Ma trận Nhầm lẫn (Confusion Matrix)
```

---

### Task 0: Thiết lập Môi trường Dual-Mode & Kiểm tra Thư viện

**Mục tiêu:** Nhận diện môi trường thực thi (Colab hay Local), cấu hình thư mục làm việc, seed ngẫu nhiên và kiểm tra tăng tốc phần cứng (GPU CUDA).

**Mã nguồn chuẩn cho Notebook:**
```python
import os, sys, random, json
import numpy as np
import torch

# 1. Cố định Seed tái lập kết quả thực nghiệm
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 2. Cơ chế Dual-Mode: Tự động phát hiện Google Colab hoặc Local máy tính
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    WORKSPACE_DIR = "/content/drive/MyDrive/PetErrorAnalysis"
    DATA_ROOT = "/content/data" # Lưu data trên đĩa ảo Colab để tránh nghẽn I/O Google Drive
else:
    WORKSPACE_DIR = os.path.abspath("./")
    DATA_ROOT = os.path.join(WORKSPACE_DIR, "data")

os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)
os.chdir(WORKSPACE_DIR)

# 3. Cài đặt thư viện phụ thuộc
!pip install -q ultralytics scikit-learn opencv-python-headless matplotlib ipywidgets pillow seaborn

# 4. Kiểm tra Thiết bị tính toán (GPU / CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Môi trường: {'Google Colab' if IN_COLAB else 'Local Máy tính'}")
print(f"[*] Thư mục làm việc: {WORKSPACE_DIR}")
print(f"[*] Thiết bị tính toán: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] Chạy không báo lỗi import.
- [ ] In ra đúng `DEVICE` (Ưu tiên CUDA nếu có GPU).
- [ ] Thư mục làm việc và thư mục data được khởi tạo thành công.

---

### Task 1: Tải Oxford-IIIT Pet & Phân chia 4 Tập Dữ liệu Chuẩn hóa

**Mục tiêu:** Tải dataset chính thức, phân chia dữ liệu thành 4 tập hoàn toàn độc lập, đảm bảo phân tầng (Stratified) đủ 37 giống loài, không rò rỉ dữ liệu.

**Định lượng tập dữ liệu:**
- `trainval` (3.680 ảnh) được chia thành:
  - **Eyeball Dev:** 300 ảnh (phân tầng, ~8 ảnh/giống) - dùng để soi lỗi bằng mắt ở Task 6.
  - **Blackbox Dev:** 15% phần còn lại (~507 ảnh) - dùng để chọn model/early stopping và đo tự động.
  - **Train:** 85% phần còn lại (~2.873 ảnh) - dùng để huấn luyện mô hình.
- `test` (3.669 ảnh): Giữ nguyên vẹn, niêm phong tuyệt đối cho tới Task 10.

**Mã nguồn chuẩn cho Notebook:**
```python
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import train_test_split
from collections import Counter

# 1. Tải Dataset chính thức từ torchvision
ds_trainval = OxfordIIITPet(
    root=DATA_ROOT, split="trainval",
    target_types=("category", "segmentation"), download=True
)
ds_test = OxfordIIITPet(
    root=DATA_ROOT, split="test",
    target_types=("category", "segmentation"), download=True
)

CLASS_NAMES = ds_trainval.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"[*] Tổng số lớp: {NUM_CLASSES} (25 giống chó + 12 giống mèo)")

# 2. Trích xuất nhãn category trực tiếp từ danh sách mẫu
labels_trainval = [int(ds_trainval[i][1][0]) for i in range(len(ds_trainval))]

# 3. Phân chia 3 tập có phân tầng (Stratified Split - Đã sửa lỗi đảo tập ở bản cũ)
idx_all = np.arange(len(labels_trainval))

# Bước a: Tách chính xác 300 mẫu cho Eyeball Dev (dùng soi lỗi bằng mắt)
idx_eyeball, idx_rest = train_test_split(
    idx_all, train_size=300, stratify=labels_trainval, random_state=SEED
)

# Bước b: Tách 3.380 mẫu còn lại thành Train (85% ~ 2.873 mẫu) và Blackbox Dev (15% ~ 507 mẫu)
idx_train, idx_blackbox = train_test_split(
    idx_rest, test_size=0.15, stratify=np.array(labels_trainval)[idx_rest], random_state=SEED
)

splits = {
    "train": sorted([int(x) for x in idx_train]),
    "eyeball": sorted([int(x) for x in idx_eyeball]),
    "blackbox": sorted([int(x) for x in idx_blackbox])
}

# 4. Lưu lại danh sách index để bảo đảm tính tái lập vĩnh viễn
with open("splits.json", "w") as f:
    json.dump(splits, f, indent=2)

print(f"[*] Phân chia hoàn tất: Train={len(splits['train'])}, Blackbox={len(splits['blackbox'])}, Eyeball={len(splits['eyeball'])}")
assert len(splits["train"]) + len(splits["blackbox"]) + len(splits["eyeball"]) == 3680, "Lỗi tổng số mẫu trainval!"
assert len(splits["train"]) > len(splits["blackbox"]), "Lỗi đảo tập Train và Blackbox!"
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] `splits["train"]` có đúng ~2.873 mẫu.
- [ ] `splits["blackbox"]` có đúng ~507 mẫu.
- [ ] `splits["eyeball"]` có đúng 300 mẫu.
- [ ] File `splits.json` được ghi thành công tại thư mục gốc.

---

### Task 2: Trích xuất Bounding Box Chuẩn (Oracle) từ Trimap

**Mục tiêu:** Trích xuất khung bao con vật chuẩn xác từ mặt nạ phân đoạn Trimap làm "đầu vào hoàn hảo" (Ground Truth / Oracle Bounding Box). Xử lý đầy đủ trường hợp ngoại lệ (mask rỗng).

**Mã nguồn chuẩn cho Notebook:**
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def trimap_to_bbox(trimap_img):
    """
    Quy ước giá trị pixel trong mặt nạ Trimap của Oxford-IIIT Pet:
      - 1: Foreground (Điểm ảnh thuộc thân thể con vật)
      - 2: Background (Điểm ảnh thuộc hậu cảnh/môi trường xung quanh)
      - 3: Boundary (Đường biên/viền ranh giới giữa con vật và nền)
    Mục tiêu: Lấy toàn bộ vùng điểm ảnh là 1 hoặc 3 để bao trọn con vật (kể cả viền lông/tai/đuôi).
    """
    arr = np.asarray(trimap_img)
    mask = (arr == 1) | (arr == 3)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        # Cơ chế Fallback an toàn: nếu trimap bị lỗi/trắng trơn, trả về toàn bộ ảnh kèm cờ False
        W, H = trimap_img.size
        return (0, 0, W, H), False
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), True

def crop_with_margin(pil_img, bbox, margin=0.1):
    """
    Cắt ảnh quanh bbox kèm theo một khoảng lề an toàn (margin = 10% theo mặc định).
    Lý do kỹ thuật: Bbox quá sát mép thường cắt đứt chóp tai hoặc râu ria của con vật,
    gây khó khăn cho mô hình phân loại. Thêm 10% margin giúp giữ trọn ngữ cảnh cục bộ.
    """
    W, H = pil_img.size
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    crop_x1 = max(0, int(x1 - mx))
    crop_y1 = max(0, int(y1 - my))
    crop_x2 = min(W, int(x2 + mx))
    crop_y2 = min(H, int(y2 + my))
    return pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

# Kiểm tra trực quan trên 3 mẫu đầu tiên của Eyeball Dev để đảm bảo tính chính xác
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, idx in zip(axes, splits["eyeball"][:3]):
    img, (lbl, trimap) = ds_trainval[idx]
    bbox, valid = trimap_to_bbox(trimap)
    ax.imshow(img)
    x1, y1, x2, y2 = bbox
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"{CLASS_NAMES[lbl]} (Trimap Oracle Bbox)")
    ax.axis("off")
plt.tight_layout()
plt.savefig("check_trimap_bbox.png")
plt.show()
print("[*] Đã lưu hình kiểm tra: check_trimap_bbox.png")
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] Hàm xử lý an toàn không quăng lỗi `NoneType` khi mask rỗng.
- [ ] Khung màu đỏ trên ảnh `check_trimap_bbox.png` bao trọn con vật.

---

### Task 3: Bước A - Xây dựng Bộ phát hiện vật thể YOLOv8-nano & Đo lường IoU Benchmark

**Mục tiêu:** Tích hợp mô hình YOLOv8-nano pretrained COCO để tự động phát hiện con vật (Cat=15, Dog=16). Đánh giá chất lượng của Bước A bằng chỉ số giao cắt (IoU) so với Bbox chuẩn của Trimap trên Eyeball Dev.

**Mã nguồn chuẩn cho Notebook:**
```python
from ultralytics import YOLO

# Khởi tạo mô hình phát hiện YOLOv8-nano (tự tải trọng số yolov8n.pt lần đầu tiên)
detector_yolo = YOLO("yolov8n.pt")

# Trong tập dữ liệu chuẩn COCO: Class 15 là 'cat' (mèo), Class 16 là 'dog' (chó)
PET_COCO_CLASSES = {15, 16}

def detect_pet_yolo(pil_img, conf_min=0.15, device=DEVICE):
    """
    Dự đoán vị trí con vật bằng YOLOv8n:
    - Nếu có nhiều bounding box hợp lệ, chọn box có độ tin cậy (confidence) cao nhất.
    - Nếu không phát hiện thấy con vật nào (confidence < conf_min), trả về toàn ảnh (fallback).
    """
    W, H = pil_img.size
    results = detector_yolo.predict(pil_img, conf=conf_min, verbose=False, device=device)[0]
    
    best_box, best_conf = None, -1.0
    for box, cls_id, conf in zip(results.boxes.xyxy.tolist(), results.boxes.cls.tolist(), results.boxes.conf.tolist()):
        if int(cls_id) in PET_COCO_CLASSES and conf > best_conf:
            best_conf = conf
            best_box = [int(round(v)) for v in box]
            
    if best_box is None:
        # Fallback: Không tìm thấy thú cưng, giữ nguyên toàn bộ ảnh và trả cờ False
        return (0, 0, W, H), False
    return tuple(best_box), True

def compute_iou(boxA, boxB):
    """
    Tính chỉ số Intersection over Union (IoU) giữa 2 Bounding Box A và B:
    IoU = Diện tích phần giao nhau / Diện tích phần hợp nhau.
    Chỉ số này nằm trong đoạn [0, 1], đo lường mức độ khớp giữa YOLO và Oracle Bbox.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# Đánh giá IoU trung bình trên tập Eyeball Dev (300 mẫu)
ious = []
found_count = 0
for idx in splits["eyeball"]:
    img, (_, trimap) = ds_trainval[idx]
    box_oracle, _ = trimap_to_bbox(trimap)
    box_yolo, found = detect_pet_yolo(img)
    if found:
        found_count += 1
    ious.append(compute_iou(box_oracle, box_yolo))

mean_iou = float(np.mean(ious))
detection_rate = found_count / len(splits["eyeball"])
print(f"[*] Tỷ lệ phát hiện thú cưng của YOLO trên Eyeball Dev: {detection_rate*100:.1f}%")
print(f"[*] IoU trung bình giữa YOLO và Oracle Bbox: {mean_iou:.3f} (Chất lượng phát hiện Bước A)")
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] Tỷ lệ phát hiện đạt > 90%.
- [ ] IoU trung bình ghi nhận được nằm trong khoảng 0.70 - 0.88.

---

### Task 4: Bước B - Xây dựng Bộ phân loại ResNet18 & Vòng lặp Train Chuẩn hóa

**Mục tiêu:** Xây dựng Dataset class `PetCropsDataset`, cấu hình data augmentation, xây dựng mạng ResNet18 fine-tuning, và hoàn thiện vòng lặp huấn luyện hỗ trợ AMP (Automatic Mixed Precision), scheduler, validation monitor và **lưu Best Checkpoint** dựa trên validation Macro F1.

**Mã nguồn chuẩn cho Notebook:**
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score

# 1. Dataset Wrapper cho Bbox Crops
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class PetCropsDataset(Dataset):
    """
    Dataset tùy biến:
    Nhận vào ảnh gốc + toạ độ Bounding Box tương ứng -> thực hiện cắt (crop) với lề margin -> biến đổi Transform.
    """
    def __init__(self, base_dataset, indices, bbox_dict, transform, margin=0.1):
        self.base_dataset = base_dataset
        self.indices = indices
        self.bbox_dict = bbox_dict
        self.transform = transform
        self.margin = margin
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, (lbl, _) = self.base_dataset[real_idx]
        bbox = self.bbox_dict[real_idx]
        cropped = crop_with_margin(img, bbox, margin=self.margin)
        return self.transform(cropped), lbl

# 2. Định nghĩa Transforms Chuẩn cho Giai đoạn Huấn luyện và Đánh giá
tf_train_baseline = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

tf_eval_standard = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# 3. Khởi tạo Mô hình ResNet18 Pretrained trên ImageNet-1K
def build_classifier(num_classes=37):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    # Thay thế Fully Connected Layer cuối cùng bằng Linear Layer 37 lớp (37 giống chó mèo)
    model.fc = nn.Linear(in_features, num_classes)
    return model

# 4. Hàm Dự đoán và Đánh giá Metric (Accuracy và Macro F1)
@torch.no_grad()
def evaluate_model(model, dataloader, device=DEVICE):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    acc = float(accuracy_score(all_labels, all_preds))
    # Sử dụng Macro F1: tính trung bình F1 của từng lớp riêng biệt để đối xử bình đẳng giữa 37 lớp
    f1 = float(f1_score(all_labels, all_preds, average="macro"))
    return acc, f1, all_preds, all_labels

# 5. Vòng lặp Huấn luyện Chuẩn hóa (Theo dõi Validation trên Blackbox Dev & Tự động lưu Best Checkpoint)
def train_classifier(model, train_loader, val_loader, epochs=15, lr=3e-4, device=DEVICE, checkpoint_path="best_model.pt"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # Label smoothing 0.05 giúp giảm overconfidence
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    
    best_val_f1 = -1.0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if device == "cuda":
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        
        # Đánh giá độc lập trên tập Blackbox Dev ở cuối mỗi epoch
        val_acc, val_f1, _, _ = evaluate_model(model, val_loader, device=device)
        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        # Tiêu chí chọn Checkpoint: Lấy mô hình có Macro F1 cao nhất trên Blackbox Dev
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            
    print(f"[*] Huấn luyện hoàn tất. Best Val Macro-F1: {best_val_f1:.4f}. Đã lưu checkpoint tại {checkpoint_path}")
    # Nạp lại trọng số tốt nhất trước khi trả về
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] Code chạy mượt mà với AMP `torch.amp.autocast`.
- [ ] Checkpoint tốt nhất được lưu lại và nạp lại thành công.
- [ ] Lưu lại mảng `history` để phục vụ vẽ đồ thị ở Task 11.

---

### Task 5: Huấn luyện Baseline (Model O vs Model Y) & So sánh Đầu vào

> [!NOTE]
> **Ý NGHĨA PHÂN TÍCH LỖI (ERROR ANALYSIS FOUNDATION):**
> Ở bước này, ta áp dụng kỹ thuật **Component Ablation / Oracle Experiment**:
> Ta huấn luyện và so sánh 2 mô hình có kiến trúc và tham số giống hệt nhau, nhưng khác nhau duy nhất ở nguồn cấp Bounding Box:
> - **Model O (Oracle Baseline):** Nhận Bounding Box "hoàn hảo" từ mặt nạ phân đoạn Trimap (Ground Truth).
> - **Model Y (Pipeline Thực tế):** Nhận Bounding Box từ bộ dò vật thể YOLOv8-nano.
> 
> Khoảng cách hiệu năng giữa Model O và Model Y chính là căn cứ khoa học trực tiếp đo lường tổn thất gây ra bởi Bước A (Detector) trong Pipeline.

**Mã nguồn chuẩn cho Notebook (Kèm Chú thích Tiếng Việt Chi tiết):**
```python
# ------------------------------------------------------------------------------------------------
# BƯỚC 1: TRÍCH XUẤT TRƯỚC TOÀN BỘ BOUNDING BOX CHO TẬP TRAINVAL
# ------------------------------------------------------------------------------------------------
# Để tránh việc gọi đi gọi lại YOLO và Trimap trong mỗi epoch (làm chậm tốc độ huấn luyện),
# ta trích xuất trước toàn bộ toạ độ hộp bao cho 3.680 ảnh và lưu vào bộ nhớ cache.

oracle_bboxes = {}  # Lưu toạ độ Bbox hoàn hảo từ Trimap
yolo_bboxes = {}    # Lưu toạ độ Bbox do YOLOv8n dự đoán
yolo_flags = {}     # Cờ ghi nhận YOLO có tìm thấy con vật hay không (True/False)

print("[*] Đang trích xuất Bounding Boxes cho toàn bộ tập trainval...")
for i in range(len(ds_trainval)):
    img, (_, trimap) = ds_trainval[i]
    box_o, _ = trimap_to_bbox(trimap)     # Lấy Ground Truth Bbox
    box_y, found = detect_pet_yolo(img)    # Lấy YOLO Bbox
    oracle_bboxes[i] = box_o
    yolo_bboxes[i] = box_y
    yolo_flags[i] = found

# ------------------------------------------------------------------------------------------------
# BƯỚC 2: HUẤN LUYỆN MODEL O (ORACLE BASELINE - ĐẦU VÀO HOÀN HẢO TỪ TRIMAP)
# ------------------------------------------------------------------------------------------------
# Mục đích: Thiết lập mức trần tiềm năng (Upper Bound) của bộ phân loại nếu Bước A không mắc bất kỳ sai sót nào.
print("\n" + "="*70)
print("[*] BẮT ĐẦU HUẤN LUYỆN MODEL O (ORACLE BASELINE - DÙNG BBOX CHUẨN TỪ TRIMAP)")
print("="*70)
loader_train_O = DataLoader(PetCropsDataset(ds_trainval, splits["train"], oracle_bboxes, tf_train_baseline), batch_size=32, shuffle=True, num_workers=2)
loader_black_O = DataLoader(PetCropsDataset(ds_trainval, splits["blackbox"], oracle_bboxes, tf_eval_standard), batch_size=64, shuffle=False, num_workers=2)
loader_eye_O   = DataLoader(PetCropsDataset(ds_trainval, splits["eyeball"], oracle_bboxes, tf_eval_standard), batch_size=64, shuffle=False, num_workers=2)

model_oracle = build_classifier()
model_oracle, history_oracle = train_classifier(
    model_oracle, loader_train_O, loader_black_O, epochs=15, lr=3e-4, checkpoint_path="oracle_v1.pt"
)

# ------------------------------------------------------------------------------------------------
# BƯỚC 3: HUẤN LUYỆN MODEL Y (YOLO BASELINE - PIPELINE THỰC TẾ TRONG SẢN PHẨM)
# ------------------------------------------------------------------------------------------------
# Mục đích: Đo lường hiệu năng của toàn bộ chuỗi Pipeline nối tiếp (Detector -> Crop -> Classifier).
print("\n" + "="*70)
print("[*] BẮT ĐẦU HUẤN LUYỆN MODEL Y (YOLO BASELINE - PIPELINE GHÉP NỐI THỰC TẾ)")
print("="*70)
loader_train_Y = DataLoader(PetCropsDataset(ds_trainval, splits["train"], yolo_bboxes, tf_train_baseline), batch_size=32, shuffle=True, num_workers=2)
loader_black_Y = DataLoader(PetCropsDataset(ds_trainval, splits["blackbox"], yolo_bboxes, tf_eval_standard), batch_size=64, shuffle=False, num_workers=2)
loader_eye_Y   = DataLoader(PetCropsDataset(ds_trainval, splits["eyeball"], yolo_bboxes, tf_eval_standard), batch_size=64, shuffle=False, num_workers=2)

model_yolo = build_classifier()
model_yolo, history_yolo = train_classifier(
    model_yolo, loader_train_Y, loader_black_Y, epochs=15, lr=3e-4, checkpoint_path="yolo_v1.pt"
)

# ------------------------------------------------------------------------------------------------
# BƯỚC 4: ĐÁNH GIÁ ĐỐI SÁNH TRÊN CẢ 2 TẬP DEV (EYEBALL VÀ BLACKBOX)
# ------------------------------------------------------------------------------------------------
# Đánh giá Model O
acc_o_black, f1_o_black, _, _ = evaluate_model(model_oracle, loader_black_O)
acc_o_eye,   f1_o_eye,   _, _ = evaluate_model(model_oracle, loader_eye_O)

# Đánh giá Model Y (Lưu lại nhãn dự đoán preds_y_eye trên Eyeball Dev để phục vụ Task 6 phân tích lỗi)
acc_y_black, f1_y_black, _, _ = evaluate_model(model_yolo, loader_black_Y)
acc_y_eye,   f1_y_eye, preds_y_eye, labels_eye = evaluate_model(model_yolo, loader_eye_Y)

# Lưu kết quả tổng hợp vào baseline.json
baseline_results = {
    "oracle_blackbox": {"accuracy": acc_o_black, "macro_f1": f1_o_black},
    "oracle_eyeball":  {"accuracy": acc_o_eye,   "macro_f1": f1_o_eye},
    "yolo_blackbox":   {"accuracy": acc_y_black, "macro_f1": f1_y_black},
    "yolo_eyeball":    {"accuracy": acc_y_eye,   "macro_f1": f1_y_eye},
}

with open("baseline.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print("\n[*] KẾT QUẢ ĐỐI SÁNH BASELINE ĐÃ LƯU VÀO baseline.json:")
print(json.dumps(baseline_results, indent=2))
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] Cả 2 checkpoint `oracle_v1.pt` và `yolo_v1.pt` được lưu thành công.
- [ ] File `baseline.json` chứa đủ 4 bộ chỉ số.
- [ ] Điểm số của Model O thường cao hơn Model Y (phản ánh tác động của việc cắt bbox chuẩn).

---

### Task 6: Bảng Phân tích Lỗi trên Eyeball Dev (Gắn nhãn Tương tác & Giải thích Chuyên sâu)

> [!IMPORTANT]
> **TRIẾT LÝ PHÂN TÍCH LỖI EYEBALL DEV CỦA ANDREW NG:**
> 1. **Tại sao lấy mẫu tối đa 100 ảnh sai?**  
>    Con số 100 mẫu sai là con số chuẩn mực trong thực hành ML. Nó đủ lớn để đại diện thống kê cho phân phối các loại lỗi (sai số thống kê khoảng $\pm 5\%$), nhưng đủ nhỏ để con người có thể trực tiếp quan sát bằng mắt trong vòng 10–15 phút mà không bị quá tải nhận thức.
> 2. **Hệ thống nhãn lỗi kép (Dual-Tagging System):**
>    - **Nhãn tự động (tags_auto):** Đo lường bằng các thuật toán xử lý ảnh khách quan:
>      - `mờ`: Tính phương sai Laplacian ($\sigma^2_{\Delta}$). Phép toán này phát hiện cường độ cạnh (edge gradient). Ảnh sắc nét có nhiều cạnh dốc $\rightarrow$ phương sai cao; ảnh bị nhòe/mờ chuyển động có biên cạnh phẳng $\rightarrow$ phương sai thấp (< 100.0).
>      - `tối`: Giá trị xám trung bình $< 60.0$ (thang 0–255), nhận diện ảnh chụp ban đêm hoặc ngược sáng.
>      - `vật thể nhỏ`: Tỷ lệ diện tích Bbox so với diện tích toàn ảnh $< 20\%$. Khi ảnh con vật quá nhỏ, khi resize về $224 \times 224$, đặc trưng khuôn mặt bị biến dạng nặng.
>    - **Nhãn thủ công (tags_manual):** Đánh giá ngữ nghĩa bằng mắt người:
>      - `giống tương tự`: Model nhầm giữa 2 giống họ hàng gần (ví dụ: Bengal vs Egyptian Mau, Pug vs French Bulldog).
>      - `che khuất`: Bị chăn, rọ mõm, cành cây hoặc người bế che mất đặc trưng nhận diện.
>      - `góc lạ`: Chụp từ trên đỉnh đầu xuống, chụp từ phía sau mông, hoặc chỉ thấy nửa thân.
>      - `nhiều con`: Trong ảnh có 2–3 con vật cùng xuất hiện, gây nhiễu crop.
>      - `nhãn sai (Label Noise)`: Nhãn trong dataset gốc bị sai hoặc gây tranh cãi.

**Mã nguồn chuẩn cho Notebook (Kèm Chú thích Tiếng Việt Chi tiết):**
```python
import cv2
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

# ================================================================================================
# BƯỚC 1: THUẬT TOÁN TỰ ĐỘNG GẮN NHÃN CHẤT LƯỢNG ẢNH (AUTO QUALITY TAGS)
# ================================================================================================
BLUR_THRESHOLD = 100.0   # Ngưỡng phương sai toán tử Laplacian (Dưới 100 là ảnh mờ/out-of-focus)
DARK_THRESHOLD = 60.0    # Ngưỡng độ sáng trung bình kênh grayscale (Dưới 60 là ảnh thiếu sáng)
SMALL_THRESHOLD = 0.20   # Ngưỡng tỷ lệ diện tích (Bbox chiếm dưới 20% khung hình là vật thể nhỏ)

def get_auto_quality_tags(pil_img, bbox):
    """
    Phân tích thuộc tính vật lý của ảnh và bounding box để gắn nhãn tự động.
    Trả về danh sách các nhãn: ['mờ', 'tối', 'vật thể nhỏ']
    """
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    W, H = pil_img.size
    tags = []
    
    # 1. Đo độ mờ (Blur detection via Laplacian variance)
    # Toán tử Laplacian tính vi phân bậc 2: ảnh nét có bước nhảy gradient lớn -> phương sai cao
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        tags.append("mờ")
        
    # 2. Đo độ sáng (Mean pixel intensity)
    if gray.mean() < DARK_THRESHOLD:
        tags.append("tối")
        
    # 3. Đo kích thước tương đối của vật thể trong khung hình
    area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (W * H)
    if area_ratio < SMALL_THRESHOLD:
        tags.append("vật thể nhỏ")
        
    return tags

# ================================================================================================
# BƯỚC 2: TRÍCH XUẤT CÁC CA ĐOÁN SAI TRÊN EYEBALL DEV VÀ CHỌN RA 100 MẪU
# ================================================================================================
# Lọc ra danh sách index của các ảnh mà Model Y dự đoán sai trên tập Eyeball Dev
wrong_indices = [splits["eyeball"][k] for k, (p, t) in enumerate(zip(preds_y_eye, labels_eye)) if p != t]

random.seed(SEED) # Giữ seed cố định để đảm bảo việc lấy mẫu có tính tái lập
sample_wrong_indices = random.sample(wrong_indices, min(100, len(wrong_indices)))
print(f"[*] Tổng số mẫu đoán sai trên Eyeball Dev: {len(wrong_indices)}/300 ({len(wrong_indices)/300*100:.1f}%)")
print(f"[*] Chọn ngẫu nhiên {len(sample_wrong_indices)} mẫu sai để tiến hành phân tích lỗi chuyên sâu.")

# Khởi tạo bảng dữ liệu ghi nhận lỗi
error_records = []
for idx in sample_wrong_indices:
    img, (lbl, _) = ds_trainval[idx]
    pred = preds_y_eye[splits["eyeball"].index(idx)]
    auto_tags = get_auto_quality_tags(img, yolo_bboxes[idx])
    error_records.append({
        "index": idx,
        "true_label": CLASS_NAMES[lbl],
        "pred_label": CLASS_NAMES[pred],
        "tags_auto": "|".join(auto_tags),
        "tags_manual": "",
        "ghi_chu": ""
    })
df_errors = pd.DataFrame(error_records)

# ================================================================================================
# BƯỚC 3: QUY TRÌNH GÁN NHÃN THỦ CÔNG (TƯƠNG TÁC QUA WIDGET HOẶC TỰ ĐỘNG QUICK-MOCK)
# ================================================================================================
# Cờ QUICK_MOCK_TAGS:
# - Đặt False: Mở giao diện tương tác trực quan để bạn tự tay tick chọn nhãn cho 100 ảnh (Khuyến nghị).
# - Đặt True:  Tự động sinh nhãn mẫu để chạy kiểm thử nhanh toàn bộ luồng mà không phải chờ đợi.
QUICK_MOCK_TAGS = False 

if QUICK_MOCK_TAGS:
    print("[!] KÍCH HOẠT CHẾ ĐỘ QUICK AUTO-MOCK: Tự động gán nhãn giả định để test nhanh luồng code...")
    mock_candidates = ["giống tương tự", "che khuất", "góc lạ", "nhiều con", "nhãn sai"]
    for i in range(len(df_errors)):
        assigned = []
        if df_errors.loc[i, "tags_auto"]:
            assigned.append(df_errors.loc[i, "tags_auto"].split("|")[0])
        assigned.append(random.choice(mock_candidates))
        df_errors.loc[i, "tags_manual"] = "|".join(list(set(assigned)))
    df_errors.to_csv("error_table.csv", index=False)
    print("[*] Đã lưu bảng phân tích lỗi tự động vào file: error_table.csv")
else:
    print("[*] KHỞI TẠO GIAO DIỆN GÁN NHÃN TƯƠNG TÁC (INTERACTIVE WIDGET)...")
    print("[*] Bạn chỉ cần quan sát ảnh con vật được crop, tick chọn các lý do gây sai rồi bấm 'Lưu & Ảnh kế tiếp'.")
    current_step = 0
    
    out_img = widgets.Output()
    lbl_info = widgets.Label()
    chk_similar = widgets.Checkbox(description="giống tương tự (Nhầm giống họ hàng gần)")
    chk_occluded = widgets.Checkbox(description="che khuất (Bị vật cản/người che)")
    chk_angle = widgets.Checkbox(description="góc lạ (Chụp góc quái chiêu/mất đầu)")
    chk_mult = widgets.Checkbox(description="nhiều con (Trong ảnh có nhiều con vật)")
    chk_wrong_label = widgets.Checkbox(description="nhãn sai (Dataset gốc gán sai nhãn)")
    btn_next = widgets.Button(description="Lưu & Ảnh kế tiếp", button_style='success', icon='check')
    
    def render_current():
        with out_img:
            clear_output(wait=True)
            row = df_errors.iloc[current_step]
            img, _ = ds_trainval[row["index"]]
            lbl_info.value = f"[{current_step+1}/{len(df_errors)}] Nhãn Thật: '{row['true_label']}' | Mô hình đoán: '{row['pred_label']}' | Auto: [{row['tags_auto']}]"
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.imshow(crop_with_margin(img, yolo_bboxes[row["index"]]))
            ax.set_title(f"Thực tế: {row['true_label']} -> Đoán nhầm: {row['pred_label']}", color="red")
            ax.axis("off")
            plt.show()
            
    def on_next_clicked(b):
        global current_step
        selected = []
        if chk_similar.value: selected.append("giống tương tự")
        if chk_occluded.value: selected.append("che khuất")
        if chk_angle.value: selected.append("góc lạ")
        if chk_mult.value: selected.append("nhiều con")
        if chk_wrong_label.value: selected.append("nhãn sai")
        
        df_errors.loc[current_step, "tags_manual"] = "|".join(selected)
        
        # Reset lại các ô checkbox cho ảnh kế tiếp
        for chk in [chk_similar, chk_occluded, chk_angle, chk_mult, chk_wrong_label]:
            chk.value = False
            
        current_step += 1
        if current_step < len(df_errors):
            render_current()
        else:
            df_errors.to_csv("error_table.csv", index=False)
            print("\n" + "="*70)
            print("[*] CHÚC MỪNG! ĐÃ HOÀN TẤT GÁN NHÃN 100 MẪU LỖI!")
            print("[*] Kết quả đã được lưu tự động vào file: error_table.csv")
            print("="*70)
            
    btn_next.on_click(on_next_clicked)
    ui = widgets.VBox([
        lbl_info, 
        out_img, 
        widgets.VBox([chk_similar, chk_occluded, chk_angle, chk_mult, chk_wrong_label]), 
        btn_next
    ])
    display(ui)
    render_current()
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] File `error_table.csv` được tạo với đầy đủ các cột: index, true_label, pred_label, tags_auto, tags_manual.
- [ ] Không có mẫu nào bị trống cả 2 nhãn tags_auto và tags_manual.

---

### Task 7: Phân tích Phân bổ Lỗi theo Pipeline (Pipeline Error Attribution)

> [!NOTE]
> **NGUYÊN LÝ PHÂN TÍCH QUY THUỘC LỖI (ERROR ATTRIBUTION LOGIC):**
> Trong một Pipeline gồm nhiều thành phần nối tiếp nhau (Bước A $\rightarrow$ Bước B), một lỗi ở đầu ra cuối cùng có thể xuất phát từ:
> 1. **Do Bước A (Detector):** Detector cắt lệch, cắt cụt mất tai/đầu con vật, hoặc hoàn toàn không tìm thấy con vật $\rightarrow$ Cung cấp đầu vào hỏng cho Bước B (Garbage In $\rightarrow$ Garbage Out).
> 2. **Do Bước B (Classifier):** Dù được cung cấp vùng cắt hoàn hảo (Oracle Bbox), Classifier vẫn không thể phân biệt được đặc trưng của giống loài.
>
> **Thuật toán quy thuộc toán học:**
> Xét tập các mẫu bị Pipeline (Model Y) đoán sai trên Eyeball Dev: $E_Y = \{i \mid \hat{y}_{Y, i} \neq y_i\}$.
> - **Lỗi thuộc Bước A:** Số mẫu mà Model Y sai nhưng Model O (Oracle) lại đoán ĐÚNG:  
>   $$E_A = \{i \in E_Y \mid \hat{y}_{O, i} = y_i\}$$  
>   *(Chứng minh: Khi cấp Bbox hoàn hảo, hệ thống đoán đúng $\rightarrow$ Suy ra lỗi ban đầu là do Bbox của Bước A bị sai).*
> - **Lỗi thuộc Bước B:** Số mẫu mà cả Model Y lẫn Model O đều đoán SAI:  
>   $$E_B = \{i \in E_Y \mid \hat{y}_{O, i} \neq y_i\}$$  
>   *(Chứng minh: Ngay cả khi cấp Bbox hoàn hảo, mô hình vẫn đoán sai $\rightarrow$ Lỗi thuần túy thuộc năng lực phân loại của Bước B).*
>
> **Giá trị định hướng hành động (Actionable Insight):**
> - Nếu $\text{Tỷ lệ } E_A > 35\%$: Nút thắt cổ chai (bottleneck) nằm ở Detector! Việc bỏ công sức tuning ResNet18 sẽ ít mang lại hiệu quả; ta bắt buộc phải tối ưu hóa Bước A trước.
> - Nếu $\text{Tỷ lệ } E_B > 65\%$: Detector đã làm rất tốt; nút thắt nằm ở việc phân biệt đặc trưng giống loài của Bước B.

**Mã nguồn chuẩn cho Notebook (Kèm Chú thích Tiếng Việt Chi tiết):**
```python
# ================================================================================================
# ĐO LƯỜNG VÀ PHÂN BỔ NGUYÊN NHÂN LỖI TRONG PIPELINE
# ================================================================================================
# Lấy kết quả dự đoán của Model O (Oracle) trên tập Eyeball Dev
_, _, preds_o_eye, _ = evaluate_model(model_oracle, loader_eye_O)

# 1. Tìm tập các mẫu đoán sai của từng mô hình
err_y_indices = [k for k in range(len(splits["eyeball"])) if preds_y_eye[k] != labels_eye[k]]
err_o_indices = [k for k in range(len(splits["eyeball"])) if preds_o_eye[k] != labels_eye[k]]

# 2. Quy thuộc nguyên nhân:
# - fixed_by_oracle: Model Y đoán sai, nhưng khi dùng Bbox chuẩn của Oracle thì Model O đoán ĐÚNG -> Lỗi do Bước A (Detector)
fixed_by_oracle = [k for k in err_y_indices if preds_o_eye[k] == labels_eye[k]]

# - still_wrong: Cả Model Y và Model O đều đoán SAI -> Lỗi do Bước B (Classifier)
still_wrong = [k for k in err_y_indices if k in err_o_indices]

total_pipeline_errors = len(err_y_indices)
step_a_errors = len(fixed_by_oracle)
step_b_errors = len(still_wrong)

attribution = {
    "total_pipeline_errors": total_pipeline_errors,
    "step_a_detector_errors": step_a_errors,
    "step_b_classifier_errors": step_b_errors,
    "step_a_percentage": float(step_a_errors / max(1, total_pipeline_errors)),
    "step_b_percentage": float(step_b_errors / max(1, total_pipeline_errors)),
}

# Lưu kết quả định lượng vào pipeline_attribution.json
with open("pipeline_attribution.json", "w") as f:
    json.dump(attribution, f, indent=2)

print("\n" + "="*70)
print("[*] KẾT QUẢ ĐỊNH LƯỢNG PHÂN BỔ LỖI THEO PIPELINE (PIPELINE ERROR ATTRIBUTION):")
print(f"- Tổng số mẫu bị Pipeline dự đoán sai: {total_pipeline_errors} / 300 mẫu Eyeball Dev")
print(f"- Lỗi bắt nguồn từ Bước A (Bộ phát hiện YOLO):  {step_a_errors} ca ({attribution['step_a_percentage']*100:.2f}%)")
print(f"- Lỗi bắt nguồn từ Bước B (Bộ phân loại ResNet): {step_b_errors} ca ({attribution['step_b_percentage']*100:.2f}%)")
print("="*70)

if attribution['step_a_percentage'] > 0.35:
    print("[!] KẾT LUẬN CHIẾN LƯỢC: Bước A (Detector) chiếm tỷ lệ lỗi đáng kể! Cần ưu tiên cải thiện Bbox margin hoặc conf_min.")
else:
    print("[!] KẾT LUẬN CHIẾN LƯỢC: Bước B (Classifier) là nguyên nhân chính (>65%)! Cần tập trung Targeted Data Augmentation.")
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] File `pipeline_attribution.json` được tạo thành công.
- [ ] Tổng số lỗi Bước A + Bước B bằng đúng tổng số lỗi của Model Y.
- [ ] Có kết luận rõ ràng về tỷ lệ phần trăm để định hướng Task 9.

---

### Task 8: Phân tích Trần Cải thiện (Ceiling Analysis)

> [!IMPORTANT]
> **Ý NGHĨA KINH TẾ CỦA PHÂN TÍCH TRẦN CẢI THIỆN (CEILING ANALYSIS):**
> Trong một dự án kỹ thuật phần mềm hoặc Machine Learning, nguồn lực kỹ sư và thời gian luôn có hạn. Trước khi quyết định dành 2 tuần để code một module xử lý ảnh mờ hoặc ảnh tối, ta cần trả lời câu hỏi:
> *"Nếu kỹ sư giải quyết triệt để 100% bài toán này mà không gặp lỗi phụ nào, độ chính xác của toàn hệ thống sẽ tăng tối đa thêm bao nhiêu điểm phần trăm?"*
> 
> **Công thức toán học:**
> 1. Tính tỷ lệ của nhóm lỗi $T$ trong tập 100 mẫu phân tích:
>    $$\text{Tỷ lệ trong mẫu}(T) = \frac{\text{Số lượng mẫu có nhãn } T}{100}$$
> 2. Ước lượng tổng số ca mắc lỗi $T$ trên toàn bộ tập Eyeball Dev ($E_Y$ lỗi):
>    $$\text{Estimated Errors}(T) = \text{Tỷ lệ trong mẫu}(T) \times E_Y$$
> 3. Tính trần cải thiện độ chính xác (Accuracy Ceiling Gain %):
>    $$\text{Ceiling Gain}(T) = \frac{\text{Estimated Errors}(T)}{N_{\text{Eyeball}} (300)} \times 100\%$$
>
> **Lưu ý quan trọng về tính Đa nhãn (Multi-label):**
> Một ảnh có thể đồng thời vừa bị "mờ" vừa bị "che khuất". Do đó, các nhóm lỗi không loại trừ lẫn nhau (non-mutually exclusive), và tổng các Ceiling Gain có thể lớn hơn tổng tỷ lệ lỗi của toàn bộ mô hình.

**Mã nguồn chuẩn cho Notebook (Kèm Chú thích Tiếng Việt Chi tiết):**
```python
# ================================================================================================
# TÍNH TOÁN BẢNG TRẦN CẢI THIỆN (CEILING ANALYSIS TABLE)
# ================================================================================================
df_err_loaded = pd.read_csv("error_table.csv").fillna("")

# 1. Trích xuất toàn bộ các nhãn (gồm cả nhãn tự động và nhãn thủ công) của từng ảnh
# Sử dụng set() cho mỗi ảnh để tránh việc 1 ảnh bị đếm trùng lặp cùng 1 nhãn nhiều lần
all_tags_list = []
for _, r in df_err_loaded.iterrows():
    combined_tags = []
    if r["tags_auto"]:   combined_tags.extend(r["tags_auto"].split("|"))
    if r["tags_manual"]: combined_tags.extend(r["tags_manual"].split("|"))
    all_tags_list.extend(list(set(combined_tags)))

tag_counts = Counter(all_tags_list)
ceiling_rows = []
n_eyeball = len(splits["eyeball"]) # 300 mẫu

# 2. Tính toán Ceiling Gain cho từng loại lỗi
for tag, count in tag_counts.most_common():
    # Tỷ lệ xuất hiện của nhóm lỗi này trong 100 mẫu phân tích
    share_in_sample = count / len(df_err_loaded)
    # Ước lượng số lượng mẫu thực tế mắc lỗi này trên toàn tập Eyeball Dev
    estimated_total_errors = share_in_sample * total_pipeline_errors
    # Mức trần độ chính xác tăng thêm nếu loại bỏ hoàn toàn nhóm lỗi này
    accuracy_ceiling_gain = (estimated_total_errors / n_eyeball) * 100.0
    
    ceiling_rows.append({
        "loai_loi": tag,
        "so_luong_trong_100_mau": count,
        "ty_le_trong_loi (%)": round(share_in_sample * 100, 1),
        "so_loi_uoc_tinh_tren_eyeball": round(estimated_total_errors, 1),
        "can_tren_tang_accuracy (%)": round(accuracy_ceiling_gain, 2)
    })

df_ceiling = pd.DataFrame(ceiling_rows)
# Lưu bảng kết quả trần cải thiện vào ceiling.csv
df_ceiling.to_csv("ceiling.csv", index=False)

print("\n" + "="*85)
print("[*] BẢNG PHÂN TÍCH TRẦN CẢI THIỆN (CEILING ANALYSIS RANKING):")
print("="*85)
print(df_ceiling.to_string(index=False))
print("="*85)
print("[*] Gợi ý: Hãy chọn 2-3 nhóm lỗi đứng đầu bảng có 'can_tren_tang_accuracy' cao nhất để cải tiến ở Task 9!")
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] File `ceiling.csv` được lưu với danh sách nhóm lỗi được xếp hạng từ cao xuống thấp.
- [ ] Định hình được 2-3 nhóm lỗi lớn nhất để tiến hành can thiệp ở Task 9.

---

### Task 9: Cải tiến có Mục tiêu & Huấn luyện Model v2

> [!NOTE]
> **TRIẾT LÝ DATA-CENTRIC AI: CẢI TIẾN CÓ ĐỊNH HƯỚNG THAY VÌ THỬ NGHIỆM MÙ QUÁNG:**
> Thay vì áp dụng bừa bãi hàng chục kỹ thuật tăng cường dữ liệu phức tạp (gây nhiễu và làm mô hình khó hội tụ), phương pháp luận phân tích lỗi yêu cầu: **Mỗi biện pháp can thiệp kỹ thuật phải là một "liều thuốc đặc trị" cho một căn bệnh cụ thể đã được chẩn đoán ở Task 6, 7 và 8:**
> 1. **Nếu lỗi Bước A (Detector) lớn:** Tăng lề cắt `IMPROVED_MARGIN = 0.15` (hoặc 0.20) để con vật không bị cắt mất chóp tai, đuôi hoặc cằm.
> 2. **Nếu lỗi "mờ" (Blur) đứng đầu bảng:** Bổ sung `GaussianBlur(kernel_size=5)` vào tập huấn luyện.
> 3. **Nếu lỗi "tối" (Dark) đứng đầu bảng:** Tăng cường độ `ColorJitter(brightness=0.4, contrast=0.3)`.
> 4. **Nếu lỗi "che khuất" (Occlusion) đứng đầu bảng:** Bổ sung `RandomErasing(p=0.3)` buộc mạng nơ-ron phải học cách nhận diện giống loài từ các bộ phận phụ (như hoa văn lông, hình dáng tai) khi khuôn mặt bị che.

**Mã nguồn chuẩn cho Notebook (Kèm Chú thích Tiếng Việt Chi tiết):**
```python
# ================================================================================================
# BƯỚC 1: XÂY DỰNG DATA AUGMENTATION CÓ MỤC TIÊU (TARGETED TRANSFORMS) CHO MODEL V2
# ================================================================================================
# Dựa trên kết quả bảng Ceiling ở Task 8, ta phối hợp các phép biến đổi đặc trị:
tf_train_targeted_v2 = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    # Đặc trị lỗi "tối / ngược sáng": Tăng biên độ biến thiên độ sáng và độ tương phản
    T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3),
    # Đặc trị lỗi "mờ / out-of-focus": Áp dụng ngẫu nhiên bộ lọc làm nhòe Gaussian Blur với xác suất 30%
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=0.3),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    # Đặc trị lỗi "che khuất": Xoá ngẫu nhiên một vùng pixel hình chữ nhật để mô hình quen với vật cản
    T.RandomErasing(p=0.3, scale=(0.02, 0.20), value='random')
])

# BƯỚC 2: TINH CHỈNH BƯỚC A (MỞ RỘNG MARGIN CẮT BBOX NẾU BƯỚC A GÂY LỖI)
# Mở rộng lề cắt lên 15% (thay vì 10%) để khắc phục các ca cắt quá sát làm mất tai/mặt
IMPROVED_MARGIN = 0.15

# ================================================================================================
# BƯỚC 3: HUẤN LUYỆN MÔ HÌNH V2 VỚI CÙNG SEED VÀ CÙNG HYPERPARAMETERS
# ================================================================================================
print("\n" + "="*70)
print("[*] BẮT ĐẦU HUẤN LUYỆN MODEL V2 (TARGETED IMPROVEMENT - CẢI TIẾN CÓ MỤC TIÊU)")
print("="*70)

loader_train_v2 = DataLoader(
    PetCropsDataset(ds_trainval, splits["train"], yolo_bboxes, tf_train_targeted_v2, margin=IMPROVED_MARGIN),
    batch_size=32, shuffle=True, num_workers=2
)
loader_black_v2 = DataLoader(
    PetCropsDataset(ds_trainval, splits["blackbox"], yolo_bboxes, tf_eval_standard, margin=IMPROVED_MARGIN),
    batch_size=64, shuffle=False, num_workers=2
)
loader_eye_v2 = DataLoader(
    PetCropsDataset(ds_trainval, splits["eyeball"], yolo_bboxes, tf_eval_standard, margin=IMPROVED_MARGIN),
    batch_size=64, shuffle=False, num_workers=2
)

model_v2 = build_classifier()
model_v2, history_v2 = train_classifier(
    model_v2, loader_train_v2, loader_black_v2, epochs=15, lr=3e-4, checkpoint_path="v2.pt"
)

# ================================================================================================
# BƯỚC 4: ĐÁNH GIÁ ĐỐI SÁNH KẾT QUẢ GIỮA MODEL V1 VÀ MODEL V2
# ================================================================================================
acc_v2_black, f1_v2_black, _, _ = evaluate_model(model_v2, loader_black_v2)
acc_v2_eye,   f1_v2_eye,   _, _ = evaluate_model(model_v2, loader_eye_v2)

v2_results = {
    "v1_blackbox": baseline_results["yolo_blackbox"],
    "v1_eyeball":  baseline_results["yolo_eyeball"],
    "v2_blackbox": {"accuracy": acc_v2_black, "macro_f1": f1_v2_black},
    "v2_eyeball":  {"accuracy": acc_v2_eye,   "macro_f1": f1_v2_eye},
}

with open("v2.json", "w") as f:
    json.dump(v2_results, f, indent=2)

print("\n[*] BẢNG SO SÁNH KẾT QUẢ GIỮA BASELINE V1 VÀ CẢI TIẾN V2:")
print(f"- Blackbox Dev Accuracy: {baseline_results['yolo_blackbox']['accuracy']*100:.2f}%  -->  {acc_v2_black*100:.2f}%")
print(f"- Blackbox Dev Macro F1: {baseline_results['yolo_blackbox']['macro_f1']:.4f}   -->  {f1_v2_black:.4f}")
print(f"- Eyeball Dev Accuracy:  {baseline_results['yolo_eyeball']['accuracy']*100:.2f}%  -->  {acc_v2_eye*100:.2f}%")
print(f"- Eyeball Dev Macro F1:  {baseline_results['yolo_eyeball']['macro_f1']:.4f}   -->  {f1_v2_eye:.4f}")
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] Checkpoint `v2.pt` được lưu thành công.
- [ ] Kết quả đo lường được xuất ra `v2.json`.

---

### Task 10: Kiểm tra Overfitting (Eyeball vs Blackbox Dev Gap) & Đánh giá Cuối cùng trên Test

> [!IMPORTANT]
> **TRIẾT LÝ KIỂM TRA OVERFITTING TRÊN EYEBALL DEV CỦA ANDREW NG:**
> 1. **Khái niệm "Human Overfitting":**  
>    Vì kỹ sư đã trực tiếp soi bằng mắt 100 mẫu lỗi của Eyeball Dev và thiết kế ra các bộ augmentation/margin để sửa các lỗi đó, não bộ con người và mô hình có nguy cơ cao bị "học vẹt" hoặc "quá khớp" vào tập Eyeball Dev.
> 2. **Tập Blackbox Dev làm trọng tài khách quan:**  
>    Tập Blackbox Dev (507 mẫu) là tập dữ liệu **tuyệt đối không bao giờ được soi bằng mắt**. Do đó, nó đóng vai trò tấm gương phản chiếu trung thực nhất:
>    $$\Delta_{\text{Overfit}} = \text{Accuracy}(\text{Eyeball Dev}) - \text{Accuracy}(\text{Blackbox Dev})$$
>    - Nếu $\Delta_{\text{Overfit}} > 5.0\%$: Dấu hiệu cảnh báo mô hình đang bị overfit vào Eyeball Dev (kết quả tăng ảo do ta điều chỉnh quá khớp với tập soi lỗi).
>    - Nếu $\Delta_{\text{Overfit}} \le 5.0\%$: Mô hình có năng lực tổng quát hóa xuất sắc.
> 3. **Nguyên tắc Niêm phong Tập Test:**  
>    Tập Test (3.669 ảnh) chỉ được mở niêm phong đúng 1 lần duy nhất để báo cáo kết quả cuối cùng. Tuyệt đối không dùng Test để chọn checkpoint hay điều chỉnh siêu tham số!

**Mã nguồn chuẩn cho Notebook (Kèm Chú thích Tiếng Việt Chi tiết):**
```python
# ================================================================================================
# BƯỚC 1: TÍNH TOÁN KHOẢNG CÁCH OVERFITTING (DEV GAP ANALYSIS)
# ================================================================================================
gap_v1 = baseline_results["yolo_eyeball"]["accuracy"] - baseline_results["yolo_blackbox"]["accuracy"]
gap_v2 = v2_results["v2_eyeball"]["accuracy"] - v2_results["v2_blackbox"]["accuracy"]

print("\n" + "="*70)
print("[*] PHÂN TÍCH KHOẢNG CÁCH OVERFITTING (EYEBALL DEV - BLACKBOX DEV GAP):")
print(f"- Overfitting Gap ở Model v1: {gap_v1*100:.2f}%")
print(f"- Overfitting Gap ở Model v2: {gap_v2*100:.2f}%")

if gap_v2 > 0.05:
    print("[!] CẢNH BÁO: Gap > 5%! Có dấu hiệu mô hình bị overfit vào các đặc trưng quan sát trên Eyeball Dev.")
else:
    print("[*] KẾT LUẬN AN TOÀN: Gap <= 5%. Mô hình v2 có khả năng tổng quát hóa tốt trên dữ liệu chưa từng nhìn thấy!")
print("="*70)

# ================================================================================================
# BƯỚC 2: CHỌN MÔ HÌNH XUẤT SẮC NHẤT DỰA TRÊN BLACKBOX DEV
# ================================================================================================
# Lưu ý quan trọng: Quyết định chọn model đi thi đấu trên tập Test phải dựa trên Blackbox Dev (không dựa trên Eyeball Dev)
if v2_results["v2_blackbox"]["macro_f1"] >= v2_results["v1_blackbox"]["macro_f1"]:
    best_model_name = "v2"
    chosen_model = model_v2
else:
    best_model_name = "v1"
    chosen_model = model_yolo

print(f"[*] Mô hình chiến thắng được chọn để đánh giá tập Test: MODEL {best_model_name.upper()}")

# ================================================================================================
# BƯỚC 3: TRÍCH XUẤT BOUNDING BOX CHO TẬP TEST (CHẠY DUY NHẤT 1 LẦN)
# ================================================================================================
print("\n[*] Đang phát hiện Bounding Box cho tập Test (3.669 ảnh) bằng YOLOv8n...")
test_bboxes = {}
for i in range(len(ds_test)):
    img_t, _ = ds_test[i]
    b_t, _ = detect_pet_yolo(img_t)
    test_bboxes[i] = b_t

# ================================================================================================
# BƯỚC 4: ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (FINAL TEST EVALUATION)
# ================================================================================================
loader_test = DataLoader(
    PetCropsDataset(ds_test, list(range(len(ds_test))), test_bboxes, tf_eval_standard, margin=IMPROVED_MARGIN),
    batch_size=64, shuffle=False, num_workers=2
)

test_acc, test_f1, test_preds, test_labels_true = evaluate_model(chosen_model, loader_test)

final_evaluation = {
    "chosen_model": best_model_name,
    "overfitting_gap_v1": gap_v1,
    "overfitting_gap_v2": gap_v2,
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1
}

with open("final_metrics.json", "w") as f:
    json.dump(final_evaluation, f, indent=2)

print("\n" + "="*70)
print("[*] KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (CHỈ CHẠY 1 LẦN DUY NHẤT):")
print(f"- Model được chọn:    Model {best_model_name.upper()}")
print(f"- Test Accuracy:      {test_acc*100:.2f}%")
print(f"- Test Macro F1:      {test_f1:.4f}")
print("="*70)
```

**Tiêu chuẩn nghiệm thu (Acceptance Criteria):**
- [ ] Tập Test chỉ được nạp và chạy suy luận đúng 1 lần.
- [ ] File `final_metrics.json` lưu giữ đầy đủ các chỉ số nghiệm thu.

---

### Task 11: Tổng hợp Báo cáo Khoa học Trực quan trong Notebook

> [!NOTE]
> **Ý NGHĨA CỦA MA TRẬN NHẦM LẪN (CONFUSION MATRIX ANALYSIS):**
> Trong bài toán 37 lớp phân loại giống chó mèo, một số giống có chung tổ tiên di truyền hoặc chung đặc điểm hình thái cận huyết:
> - Mèo lông xù mắt xanh: `Birman` vs `Ragdoll`.
> - Chó mặt xệ tai cụp/vểnh: `Pug` vs `French Bulldog`.
> - Mèo đốm vằn hoang dã: `Bengal` vs `Egyptian Mau`.
> Bằng cách xóa đường chéo chính (`np.fill_diagonal(cm, 0)`), ta triệt tiêu các trường hợp dự đoán đúng và lập tức cô lập được top 5 cặp giống loài hay bị nhầm lẫn nhất trong không gian đặc trưng.

**Mã nguồn chuẩn cho Notebook (Kèm Chú thích Tiếng Việt Chi tiết):**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ================================================================================================
# 1. VẼ ĐỒ THỊ SO SÁNH TIẾN TRÌNH HUẤN LUYỆN (TRAINING LOSS & VAL MACRO F1 CURVES)
# ================================================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Đồ thị Training Loss
axes[0].plot(history_yolo["train_loss"], label="Model v1 (Baseline)", linestyle="--", color="gray")
axes[0].plot(history_v2["train_loss"], label="Model v2 (Targeted)", color="royalblue", linewidth=2)
axes[0].set_title("So sánh Tiến trình Giảm Loss Huấn luyện (Train Loss)", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend()

# Đồ thị Validation Macro F1 trên Blackbox Dev
axes[1].plot(history_yolo["val_f1"], label="Model v1 Val F1", linestyle="--", color="gray")
axes[1].plot(history_v2["val_f1"], label="Model v2 Val F1", color="forestgreen", linewidth=2)
axes[1].set_title("So sánh Hiệu năng trên Blackbox Dev (Val Macro F1)", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Macro F1")
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# ================================================================================================
# 2. VẼ BIỂU ĐỒ CỘT THỂ HIỆN TRẦN CẢI THIỆN THEO NHÓM LỖI (CEILING GAIN CHART)
# ================================================================================================
plt.figure(figsize=(10, 5))
df_ceil_plot = pd.read_csv("ceiling.csv")
plt.barh(df_ceil_plot["loai_loi"], df_ceil_plot["can_tren_tang_accuracy (%)"], color="salmon", edgecolor="darkred")
plt.gca().invert_yaxis() # Đảo ngược trục Y để nhóm lỗi lớn nhất hiển thị ở trên cùng
plt.title("Tiềm năng Cải thiện Độ chính xác Tối đa theo Từng Nhóm Lỗi (Ceiling Gain %)", fontsize=13)
plt.xlabel("Mức tăng Accuracy ước lượng tối đa (%)")
plt.grid(axis="x", linestyle=":", alpha=0.7)
plt.tight_layout()
plt.savefig("error_ceiling_chart.png")
plt.show()

# ================================================================================================
# 3. TRÍCH XUẤT VÀ PHÂN TÍCH TOP-5 CẶP GIỐNG DỄ NHẦM LẪN NHẤT TRÊN TẬP TEST
# ================================================================================================
cm = confusion_matrix(test_labels_true, test_preds)
np.fill_diagonal(cm, 0) # Xoá đường chéo chính để chỉ tập trung vào các trường hợp dự đoán sai

top_confused = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm[i, j] > 0:
            top_confused.append((CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j])))
top_confused = sorted(top_confused, key=lambda x: x[2], reverse=True)[:5]

print("\n" + "="*70)
print("[*] TOP 5 CẶP GIỐNG DỄ NHẦM LẪN NHẤT TRÊN TẬP TEST (MA TRẬN NHẦM LẪN):")
print("="*70)
for rank, (actual, pred, count) in enumerate(top_confused, start=1):
    print(f"Top {rank}: Giống thật '{actual:22s}'  -->  Dự đoán nhầm thành '{pred:22s}' ({count} lần)")
print("="*70)
```

---

## 4. Cấu trúc Cell Báo cáo Markdown Mẫu ở Cuối Notebook

Ngay sau các cell code ở Task 11, một cell Markdown hoàn chỉnh sẽ được tạo ra với cấu trúc:

```markdown
# BÁO CÁO KHOA HỌC: PHÂN TÍCH LỖI HỆ THỐNG PHÂN LOẠI 37 GIỐNG CHÓ MÈO

## 1. Thiết lập Thử nghiệm & Chiến lược Phân chia Dữ liệu
- **Tập Train (2.873 mẫu):** Huấn luyện mô hình.
- **Tập Eyeball Dev (300 mẫu):** Phân tích lỗi thủ công và gắn nhãn định tính.
- **Tập Blackbox Dev (507 mẫu):** Đánh giá định lượng khách quan và chọn Best Checkpoint.
- **Tập Test (3.669 mẫu):** Đo lường năng lực tổng quát hóa cuối cùng.

## 2. Kết quả Đối sánh Baseline (Pipeline vs Oracle)
- **Model O (Oracle Input):** Accuracy = [Điền số], Macro F1 = [Điền số]
- **Model Y (YOLO Input):** Accuracy = [Điền số], Macro F1 = [Điền số]
- **Phân bổ Lỗi Pipeline:**
  - Lỗi do Bước A (Detector): [Điền %]
  - Lỗi do Bước B (Classifier): [Điền %]

## 3. Bảng Phân loại Lỗi & Cận trên Cải thiện (Ceiling Analysis)
(Chèn bảng từ file ceiling.csv và hình error_ceiling_chart.png)

## 4. Hiệu quả Cải tiến (Model v1 vs Model v2)
- Biện pháp can thiệp đã áp dụng: [Liệt kê Augmentation / Bbox margin]
- Mức tăng trên Blackbox Dev: từ [Điền %] lên [Điền %]
- Kiểm soát Overfitting Gap: [Điền số]% (Đạt ngưỡng an toàn < 5%)

## 5. Đánh giá Cuối cùng trên Tập Test
- **Final Test Accuracy:** [Điền %]
- **Final Test Macro F1:** [Điền số]
- **Các cặp giống hay nhầm nhất:** [Bengal - Egyptian Mau, Pug - French Bulldog, v.v.]

## 6. Hạn chế & Hướng phát triển Tương lai
- YOLOv8-nano dù nhanh nhưng còn gặp khó khi ảnh chụp nhiều con vật hoặc góc chụp quá xa.
- Hướng mở rộng: Thử nghiệm kiến trúc Vision Transformer (ViT) hoặc tinh chỉnh trực tiếp YOLO trên tập bounding box chó mèo.
```

---

## 5. Kế hoạch Thực hiện Tuần tự & Checklist Nghiệm thu

### 5.1. Kế hoạch Thời gian (Timeline dự kiến)
- **Giai đoạn 1 (Khởi động & Baseline - Task 0 đến 5):** ~1-2 giờ chạy trên GPU Colab T4.
- **Giai đoạn 2 (Phân tích Lỗi & Gắn nhãn - Task 6 đến 8):** ~15-30 phút (dùng Interactive Widget).
- **Giai đoạn 3 (Cải tiến & Nghiệm thu - Task 9 đến 11):** ~1 giờ huấn luyện và hoàn thiện báo cáo.

### 5.2. Checklist Nghiệm thu Toàn diện (Definition of Done)
- [x] File kế hoạch đã vá sạch toàn bộ 5 lỗi logic của bản cũ.
- [x] Đã bổ sung bình luận tiếng Việt chuyên sâu cho toàn bộ các Task phân tích lỗi (Task 5 đến Task 11).
- [ ] File notebook `pet_error_analysis.ipynb` được tạo hoàn chỉnh và sẵn sàng thực thi.
- [ ] Chạy tuần tự không gặp bất kỳ lỗi cú pháp (`SyntaxError`), lỗi biến (`NameError`), hay lỗi bộ nhớ (`CUDA OOM`).
- [ ] Xuất xưởng đầy đủ các file dữ liệu nghiệm thu tại thư mục gốc:
  - `splits.json`
  - `oracle_v1.pt`, `yolo_v1.pt`, `v2.pt`
  - `baseline.json`, `error_table.csv`, `pipeline_attribution.json`, `ceiling.csv`, `v2.json`, `final_metrics.json`
  - `training_history.png`, `error_ceiling_chart.png`, `check_trimap_bbox.png`
- [ ] Báo cáo tổng kết đầy đủ số liệu thực chứng, không sử dụng số liệu giả định.
