# TÀI LIỆU GIẢI THÍCH TOÀN DIỆN TỪNG CELL TRONG NOTEBOOK `pet_error_analysis.ipynb`

> **Tài liệu tham khảo kỹ thuật & bảo vệ đồ án:** Hệ thống Phân loại 37 giống Chó Mèo & Quy trình Phân tích Lỗi (Error Analysis) theo Andrew Ng.  
> **Mục đích:** Bóc tách chi tiết từng cell code: các dòng lệnh cốt lõi, cơ chế hoạt động, bản chất và công thức toán học KaTeX, dữ liệu vào/ra, kèm bảng tra cứu nhanh và bộ câu hỏi vấn đáp bảo vệ.

---

# PHẦN 1: BẢNG TRA CỨU NHANH 13 CELL CODE

| Cell Index | Task tương ứng | Lệnh / Kỹ thuật cốt lõi | Bản chất Toán học & Thuật toán | Mục đích tóm tắt |
|:---:|:---|:---|:---|:---|
| **Cell 2** | **Task 0:** Thiết lập môi trường | `torch.manual_seed(42)`, `IN_COLAB` | Pseudo-Random Number Generator (PRNG) | Cố định ngẫu nhiên, tự động nhận diện Colab / Local |
| **Cell 4** | **Task 1:** Tải & Chia tập dữ liệu | `train_test_split(..., stratify=...)` | Phân phối tiên nghiệm $P(Y=c)$, Stratified Sampling | Chia 4 tập không rò rỉ: Train (2873), Blackbox (507), Eyeball (300), Test (3669) |
| **Cell 6** | **Task 2:** Bbox chuẩn từ Trimap | `trimap_to_bbox()`, `crop_with_margin()` | Hình học giải tích 2D: $\min/\max$, Hệ số dãn lề $\text{margin}$ | Trích xuất Bbox hoàn hảo (Oracle Ground Truth) từ mask Trimap |
| **Cell 8** | **Task 3:** Detector YOLOv8n | `detector_yolo.predict()`, `compute_iou()` | Chỉ số Jaccard: $\text{IoU} = \frac{\|A \cap B\|}{\|A \cup B\|}$ | Dò vị trí con vật bằng YOLOv8n và tính IoU benchmark |
| **Cell 10** | **Task 4:** Classifier ResNet18 | `train_classifier()`, `CosineAnnealingLR` | Z-score, Cross-Entropy + Label Smoothing, Softmax, Cosine LR, Macro F1 | Vòng lặp train ResNet18 AMP FP16, lưu `best_model.pt` theo Macro F1 |
| **Cell 12** | **Task 5:** Huấn luyện Baseline | Train Model O vs Model Y | Component Ablation: $\Delta_{\text{Pipeline Loss}} = \text{Score}(O) - \text{Score}(Y)$ | Đo lường tổn thất hiệu năng do Bước A gây ra |
| **Cell 14** | **Task 6:** Bảng phân tích lỗi Eyeball | `cv2.Laplacian().var()`, `ipywidgets` | Toán tử vi phân bậc 2 Laplace $\nabla^2 I$, Phương sai độ nhòe $\text{Var}(\Delta I)$ | Tự động gắn nhãn mờ, tối, nhỏ + Widget tương tác 100 ảnh sai |
| **Cell 16** | **Task 7:** Phân bổ lỗi Pipeline | `fixed_by_oracle`, `still_wrong` | Đại số tập hợp: $E_A = E_Y \setminus E_O$, $E_B = E_Y \cap E_O$ | Định lượng tỷ lệ lỗi do Bước A (Detector) hay Bước B (Classifier) |
| **Cell 18** | **Task 8:** Phân tích trần cải thiện | `accuracy_ceiling_gain` | Ước lượng không chệch: $\text{Ceiling} = \frac{\text{Est}(T)}{N_{\text{dev}}} \times 100\%$ | Tính cận trên tăng Accuracy (%) nếu giải quyết triệt để từng nhóm lỗi |
| **Cell 20** | **Task 9:** Cải tiến có mục tiêu v2 | `GaussianBlur`, `ColorJitter`, `RandomErasing` | Phân phối chuẩn 2 chiều Gauss $G(x,y;\sigma)$, Biến đổi không gian màu | Tăng cường dữ liệu đặc trị lỗi mờ, tối, che khuất; mở rộng margin=0.15 |
| **Cell 22** | **Task 10:** Kiểm tra Overfitting & Test | `gap_v2 = Acc_{\text{eye}} - Acc_{\text{black}}` | Khoảng cách tổng quát hóa (Generalization Gap $\le 5\%$) | Kiểm tra hiện tượng Human Overfitting và mở niêm phong Test đúng 1 lần |
| **Cell 24** | **Task 11:** Báo cáo & Trực quan | `confusion_matrix`, `np.fill_diagonal(cm, 0)` | Ma trận nhầm lẫn $C_{i,j}$, Triệt tiêu đường chéo chính | Vẽ đồ thị training curve, trần cải thiện và cô lập Top 5 cặp giống dễ nhầm nhất |
| **Cell 26** | **Task 12:** Model v3 Champion | ResNet50, Ảnh 320x320, `scale=(0.8, 1.0)` | Mở rộng diện tích pixel $2.04\times$, Không gian embedding 2048 chiều | Đột phá phân loại tinh (Fine-Grained) cho các giống tương tự nhau |

---

# PHẦN 2: MỔ XẺ CHI TIẾT TỪNG CELL CODE (CELL 2 ĐẾN CELL 26)

---

## CELL 2: TASK 0 - THIẾT LẬP MÔI TRƯỜNG DUAL-MODE & KIỂM TRA GPU

### 1. Mục đích & Ý nghĩa
Cell này là nền tảng khởi đầu của toàn bộ dự án, đảm bảo tính **tái lập thực nghiệm (Reproducibility)** và tính **tương thích linh hoạt (Dual-Mode)**:
- Đảm bảo mọi phép ngẫu nhiên (chia dữ liệu, khởi tạo trọng số, shuffle batch) đều sinh ra kết quả y hệt nhau giữa các lần chạy.
- Tự động nhận diện môi trường đang chạy là Google Colab hay máy Local để thiết lập đường dẫn lưu trữ phù hợp mà không cần sửa code.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
# Cố định Seed ngẫu nhiên
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```
- `random.seed(42)` & `np.random.seed(42)`: Khóa bộ sinh số ngẫu nhiên của Python thuần và thư viện NumPy.
- `torch.manual_seed(42)` & `torch.cuda.manual_seed_all(42)`: Khóa bộ sinh số ngẫu nhiên của PyTorch trên cả CPU và toàn bộ các GPU CUDA. Số 42 là quy ước nổi tiếng trong khoa học máy tính (The Hitchhiker's Guide to the Galaxy).

```python
# Tự động nhận diện Colab / Local
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    WORKSPACE_DIR = "/content/drive/MyDrive/PetErrorAnalysis"
    DATA_ROOT = "/content/data"
else:
    WORKSPACE_DIR = os.path.abspath("./")
    DATA_ROOT = os.path.join(WORKSPACE_DIR, "data")
```
- Kiểm tra module hệ thống `sys.modules`: nếu có chuỗi `'google.colab'` thì tự động kích hoạt lệnh mount Google Drive.
- Điểm tinh tế về kỹ thuật: File dữ liệu ảnh nặng lưu ở ổ đĩa ảo Colab (`/content/data`) để GPU đọc ảnh với tốc độ RAM disk, còn kết quả/checkpoint lưu vào Drive (`/content/drive/MyDrive/...`) để không bị mất khi hết phiên.

### 3. Bản chất & Công thức Toán học liên quan

#### Thuật toán sinh số giả ngẫu nhiên (Pseudo-Random Number Generator - PRNG)
Máy tính không thể tự tạo ra sự ngẫu nhiên thực sự. Các hàm `random()` bản chất là các chuỗi số xác định được sinh ra từ một phương trình đệ quy đồng dư tuyến tính (Linear Congruential Generator):
$$X_{n+1} = (a X_n + c) \pmod m$$
Trong đó:
- $X_0$ chính là giá trị `SEED` khởi tạo ($X_0 = 42$).
- $a, c, m$ là các hằng số nguyên tố lớn được định nghĩa trước.
- **Ý nghĩa toán học:** Khi ta cố định $X_0 = 42$, toàn bộ dãy số $\{X_1, X_2, X_3, \dots\}$ sinh ra trong tương lai là **hoàn toàn cố định 100%**. Nhờ đó, bất kỳ ai chạy lại code trên máy tính khác cũng sẽ thu được kết quả phân chia dữ liệu và độ chính xác y hệt nhau.

### 4. Input & Output của Cell
- **Input:** Không có (nhận cấu hình hệ thống).
- **Output:** Biến toàn cục `DEVICE` (`'cuda'` hoặc `'cpu'`), `WORKSPACE_DIR`, `DATA_ROOT`, thư mục làm việc được tạo thành công trên đĩa.

---

## CELL 4: TASK 1 - TẢI DATASET VÀ PHÂN CHIA 4 TẬP DỮ LIỆU CHUẨN HÓA

### 1. Mục đích & Ý nghĩa
Tải tập dữ liệu chính thức Oxford-IIIT Pet (37 giống chó mèo) và phân chia thành 4 tập dữ liệu riêng biệt theo đúng triết lý của Andrew Ng:
1. **Train set (2.873 ảnh):** Huấn luyện mô hình học các trọng số.
2. **Eyeball Dev set (300 ảnh):** Tập dữ liệu để con người trực tiếp "soi bằng mắt" và phân tích định tính các mẫu lỗi.
3. **Blackbox Dev set (507 ảnh):** Tập dữ liệu "hộp đen" tuyệt đối không được soi bằng mắt, dùng làm thước đo khách quan để chọn checkpoint tốt nhất và kiểm tra overfitting.
4. **Test set (3.669 ảnh):** Được niêm phong tuyệt đối, chỉ mở đúng 1 lần ở cuối bài.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
# Tách 300 mẫu Eyeball Dev
idx_eyeball, idx_rest = train_test_split(
    idx_all, train_size=300, stratify=labels_trainval, random_state=SEED
)

# Tách phần còn lại thành Train (85%) và Blackbox Dev (15%)
idx_train, idx_blackbox = train_test_split(
    idx_rest, test_size=0.15, stratify=np.array(labels_trainval)[idx_rest], random_state=SEED
)
```
- Lệnh `train_test_split` với tham số `stratify=...` đảm bảo chia phân tầng: tỷ lệ các giống loài trong tập con giữ nguyên như tập gốc.
- **Vá lỗi kỹ thuật nghiêm trọng của bản cũ:** Ở bản cũ, code gán ngược `black_idx, train_idx = train_test_split(rest, test_size=0.15)` khiến Blackbox nhận 85% còn Train chỉ có 15%. Ở cell này, thứ tự gán chuẩn xác là `idx_train, idx_blackbox` (Train nhận 85% ~ 2.873 ảnh, Blackbox nhận 15% ~ 507 ảnh).

### 3. Bản chất & Công thức Toán học liên quan

#### Phân tầng xác suất (Stratified Sampling)
Giả sử tập dữ liệu có $C = 37$ lớp. Gọi $N_c$ là số lượng mẫu của giống loài $c$, tổng số mẫu là $N = \sum_{c=1}^C N_c$.  
Xác suất tiên nghiệm (Prior probability) của giống $c$ trong tập dữ liệu gốc là:
$$P(Y = c) = \frac{N_c}{N}$$
Nếu lấy mẫu ngẫu nhiên đơn thuần (Simple Random Sampling), các giống hiếm có thể bị rơi rớt, dẫn đến một số lớp có $0$ mẫu trong tập kiểm tra.  
Phép chia phân tầng (`stratify`) ràng buộc số lượng mẫu của mỗi giống $c$ trong tập con kích thước $M$ phải thỏa mãn:
$$M_c = \text{round}\left(M \times P(Y = c)\right) = \text{round}\left(M \times \frac{N_c}{N}\right)$$
- **Ví dụ số thực tế:**  
  Tập `trainval` có 3.680 ảnh cho 37 lớp (bình quân khoảng $N_c \approx 99.45$ ảnh/lớp, tức $P(Y=c) \approx \frac{1}{37} \approx 2.70\%$).  
  Khi tách tập Eyeball Dev gồm $M = 300$ ảnh, mỗi giống loài sẽ được chia đều chính xác:
  $$M_c = \text{round}\left(300 \times \frac{1}{37}\right) = 8 \text{ ảnh/giống}$$
  Nhờ đó, không một giống nào bị bỏ rơi hay thiên vị trong quá trình phân tích lỗi!

### 4. Input & Output của Cell
- **Input:** Dataset `OxfordIIITPet` từ torchvision.
- **Output:** File `splits.json` chứa 3 danh sách chỉ số mẫu: `"train"` (2.873 index), `"eyeball"` (300 index), `"blackbox"` (507 index).

---

## CELL 6: TASK 2 - TRÍCH XUẤT BOUNDING BOX CHUẨN (ORACLE) TỪ TRIMAP

### 1. Mục đích & Ý nghĩa
Tập dữ liệu Oxford-IIIT Pet cung cấp kèm mặt nạ phân đoạn Trimap. Cell này chuyển đổi mặt nạ Trimap thành toạ độ hộp bao chuẩn (**Ground Truth / Oracle Bounding Box**), đồng thời định nghĩa hàm cắt ảnh mở rộng lề an toàn (`margin=0.10`) để tránh cắt đứt tai/râu con vật.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
def trimap_to_bbox(trimap_img):
    arr = np.asarray(trimap_img)
    mask = (arr == 1) | (arr == 3)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        W, H = trimap_img.size
        return (0, 0, W, H), False
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), True
```
- Trimap có 3 giá trị pixel: `1` (con vật), `2` (nền), `3` (đường viền biên giới).
- Lệnh `(arr == 1) | (arr == 3)` cô lập toàn bộ các pixel thuộc thân thể và đường viền bao quanh con vật.
- `np.where(mask)` tìm toạ độ hàng ($y$) và cột ($x$) của các pixel này. Giá trị cực tiểu và cực đại tạo thành khung chữ nhật bao quanh nhỏ nhất.
- **Cơ chế Fallback an toàn:** Nếu mặt nạ rỗng (`len(xs) == 0`), tự động trả về `(0, 0, W, H)` và cờ `False` thay vì trả về `None`, triệt tiêu hoàn toàn lỗi crash code.

```python
def crop_with_margin(pil_img, bbox, margin=0.1):
    W, H = pil_img.size
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    return pil_img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                         min(W, int(x2 + mx)), min(H, int(y2 + my))))
```
- Tính chiều rộng $w = x_2 - x_1$ và chiều cao $h = y_2 - y_1$.
- Mở rộng đều ra 4 phía một khoảng $m_x = w \times 0.1$ và $m_y = h \times 0.1$, kẹp giá trị biên bằng `max(0, ...)` và `min(W, ...)`.

### 3. Bản chất & Công thức Toán học liên quan

#### Hình học giải tích hộp bao cực trị (Minimum Bounding Box)
Cho tập hợp các toạ độ điểm ảnh thuộc đối tượng $S = \{(x_i, y_i) \mid \text{Pixel}(x_i, y_i) \in \{1, 3\}\}$.  
Toạ độ hộp bao chuẩn hình học được xác định bởi các cực trị:
$$x_1 = \min_{(x, y) \in S} x, \quad y_1 = \min_{(x, y) \in S} y$$
$$x_2 = \max_{(x, y) \in S} x, \quad y_2 = \max_{(x, y) \in S} y$$

#### Công thức dãn lề tỷ lệ (Margin Expansion)
Khi mở rộng với hệ số lề $\alpha = \text{margin} = 0.10$ (tức 10% mỗi cạnh):
$$x_1' = \max(0, x_1 - \alpha \cdot w), \quad y_1' = \max(0, y_1 - \alpha \cdot h)$$
$$x_2' = \min(W, x_2 + \alpha \cdot w), \quad y_2' = \min(H, y_2 + \alpha \cdot h)$$
- **Diện tích mới của vùng cắt:**
  $$w' = (1 + 2\alpha)w = 1.2 w, \quad h' = (1 + 2\alpha)h = 1.2 h$$
  $$\text{Area}' = w' \times h' = 1.44 \times \text{Area}_{\text{gốc}}$$
- **Ý nghĩa thực tế:** Vùng ảnh sau khi dãn lề 10% sẽ có diện tích tăng thêm **44%**, cung cấp đủ không gian đệm xung quanh để giữ nguyên vẹn chóp tai, bờm lông và râu con vật.

### 4. Input & Output của Cell
- **Input:** Ảnh gốc PIL Image và ảnh phân đoạn Trimap.
- **Output:** Hàm `trimap_to_bbox()`, `crop_with_margin()`, và file ảnh kiểm tra trực quan `check_trimap_bbox.png`.

---

## CELL 8: TASK 3 - BƯỚC A: DETECTOR YOLOV8-NANO & ĐO LƯỜNG IOU

### 1. Mục đích & Ý nghĩa
Tích hợp mô hình YOLOv8-nano tiền huấn luyện trên tập COCO để đóng vai trò **Bước A (Bộ phát hiện đối tượng)** trong chuỗi Pipeline thực tế. Sau đó, đo lường chất lượng phát hiện bằng chỉ số giao cắt trên hợp (**IoU**) so sánh với Bbox chuẩn của Trimap.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
detector_yolo = YOLO("yolov8n.pt")
PET_COCO_CLASSES = {15, 16} # 15: Cat, 16: Dog trong COCO

def detect_pet_yolo(pil_img, conf_min=0.15, device=DEVICE):
    # Lọc box có độ tin cậy conf cao nhất thuộc class chó/mèo
    for box, cls_id, conf in zip(results.boxes.xyxy.tolist(), results.boxes.cls.tolist(), results.boxes.conf.tolist()):
        if int(cls_id) in PET_COCO_CLASSES and conf > best_conf:
            best_conf = conf
            best_box = [int(round(v)) for v in box]
```
- Lớp `cat` có ID 15, `dog` có ID 16 trong chuẩn 80 lớp của COCO.
- Quét qua toàn bộ các dự đoán, chọn ra Bbox có độ tin cậy cao nhất thỏa mãn `conf >= 0.15`.

```python
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0
```

### 3. Bản chất & Công thức Toán học liên quan

#### Chỉ số tương đồng Jaccard / Intersection over Union (IoU)
Cho 2 hộp bao $A$ (Oracle Box) và $B$ (YOLO Box). Chỉ số IoU là tỷ số giữa diện tích phần giao nhau và diện tích phần hợp nhau:
$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

Toạ độ hình chữ nhật giao nhau $A \cap B$ được tính:
$$x_{\text{inter\_left}} = \max(x_1^A, x_1^B), \quad y_{\text{inter\_top}} = \max(y_1^A, y_1^B)$$
$$x_{\text{inter\_right}} = \min(x_2^A, x_2^B), \quad y_{\text{inter\_bottom}} = \min(y_2^A, y_2^B)$$
$$\text{Area}(A \cap B) = \max(0, x_{\text{inter\_right}} - x_{\text{inter\_left}}) \times \max(0, y_{\text{inter\_bottom}} - y_{\text{inter\_top}})$$

- **Quy ước đánh giá chất lượng IoU:**
  - $\text{IoU} = 1.0$: Khớp hoàn hảo tuyệt đối.
  - $\text{IoU} \ge 0.70$: Phát hiện rất tốt, bao trọn con vật.
  - $\text{IoU} < 0.50$: Phát hiện lệch hoặc cắt cụt một phần cơ thể.
- **Kết quả thực nghiệm trên Eyeball Dev:**
  $$\text{Mean IoU} \approx 0.78 - 0.82, \quad \text{Detection Rate} > 95\%$$
  Chứng minh YOLOv8n hoạt động rất ổn định làm đầu vào cho Bước B.

### 4. Input & Output của Cell
- **Input:** Model trọng số `yolov8n.pt`, ảnh của tập Eyeball Dev.
- **Output:** Hàm `detect_pet_yolo()`, `compute_iou()`, tỷ lệ phát hiện (%) và IoU trung bình in ra màn hình console.

---

## CELL 10: TASK 4 - BƯỚC B: CLASSIFIER RESNET18 & VÒNG LẶP TRAIN CHUẨN

### 1. Mục đích & Ý nghĩa
Xây dựng lớp dữ liệu `PetCropsDataset`, thiết lập chuẩn hóa ảnh ImageNet, xây dựng mạng phân loại ResNet18 fine-tuning 37 lớp, và viết vòng lặp huấn luyện hỗ trợ tăng tốc **Automatic Mixed Precision (AMP)** kết hợp theo dõi validation trên Blackbox Dev để **tự động lưu `best_model.pt`**.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
# 1. Chuẩn hóa ảnh theo phân phối ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
tf_eval_standard = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# 2. Thay thế Fully Connected Layer cho 37 giống
def build_classifier(num_classes=37):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features # 512 chiều
    model.fc = nn.Linear(in_features, num_classes)
    return model
```
- ResNet18 có đầu ra tầng áp chót là vector 512 chiều. Dòng `nn.Linear(512, 37)` chuyển đổi vector này thành 37 điểm số phân loại (logits).

```python
# 3. Vòng lặp huấn luyện có scheduler và AMP
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
scaler = torch.amp.GradScaler('cuda')
```
- `torch.amp.autocast('cuda', dtype=torch.float16)`: Ép các phép nhân ma trận tích chập chạy trên định dạng số thực 16-bit (FP16), giúp tăng tốc độ train gấp 2-3 lần và giảm 50% bộ nhớ VRAM.
- `scaler.scale(loss).backward()`: Chống hiện tượng tràn dưới số học (underflow) của FP16.
- Sau mỗi epoch, hàm gọi `evaluate_model(model, val_loader)` trên tập **Blackbox Dev**; nếu `val_f1 > best_val_f1`, lưu ngay checkpoint `checkpoint_path`.

### 3. Bản chất & Công thức Toán học liên quan

#### a. Chuẩn hóa Z-Score (Standardization)
Mỗi giá trị điểm ảnh $x \in [0, 1]$ được chuẩn hóa độc lập trên 3 kênh màu (R, G, B):
$$z = \frac{x - \mu}{\sigma}$$
- $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
- **Ý nghĩa toán học:** Đưa phân phối dữ liệu đầu vào về kỳ vọng bằng $0$ ($\mathbb{E}[z] = 0$) và phương sai bằng $1$ ($\text{Var}(z) = 1$). Điều này triệt tiêu độ lệch gradient giữa các kênh màu, giúp hàm mất mát dạng mặt lồi elip trở thành mặt cầu đối xứng, thuật toán Gradient Descent hội tụ nhanh gấp nhiều lần.

#### b. Hàm Kích hoạt Softmax
Từ vector logits $\mathbf{z} = [z_1, z_2, \dots, z_K]^T$ ($K=37$), xác suất dự đoán của giống $i$ là:
$$p_i = \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad \sum_{i=1}^K p_i = 1$$

#### c. Hàm Mất mát Cross-Entropy kèm Label Smoothing
Trong bài toán phân loại thông thường, nhãn thực tế là one-hot vector $y \in \{0, 1\}^K$. Khi áp dụng **Label Smoothing** với hệ số $\epsilon = 0.05$:
$$y_i' = (1 - \epsilon) y_i + \frac{\epsilon}{K}$$
- Với nhãn đúng ($y_i = 1$): $y_i' = (1 - 0.05) \times 1 + \frac{0.05}{37} \approx 0.9514$.
- Với nhãn sai ($y_i = 0$): $y_i' = \frac{0.05}{37} \approx 0.00135$.
Hàm mất mát trên mẫu dữ liệu là:
$$\mathcal{L}_{\text{CE}} = - \sum_{i=1}^K y_i' \log(p_i) = - (1 - \epsilon) \log(p_{\text{target}}) - \frac{\epsilon}{K} \sum_{i=1}^K \log(p_i)$$
- **Ý nghĩa toán học:** Ngăn chặn việc mô hình đẩy giá trị logit $z_{\text{target}} \to +\infty$ để đạt $p_{\text{target}} = 1.0$. Việc này làm mềm ranh giới quyết định, giảm hiện tượng quá tự tin (overconfidence) và tăng độ bền vững khi gặp các giống cận huyết tương tự nhau.

#### d. Bộ điều chỉnh Tốc độ học Cosine Annealing (Cosine Annealing LR)
Tốc độ học $\eta_t$ ở epoch thứ $t$ trong tổng số $T_{\max} = 15$ epoch suy giảm theo đồ thị nửa chu kỳ hàm cosin:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{t}{T_{\max}} \pi\right)\right)$$
- Ở epoch đầu ($t=0$): $\cos(0) = 1 \implies \eta_0 = \eta_{\max} = 3 \times 10^{-4}$ (học nhanh).
- Ở epoch cuối ($t=T_{\max}$): $\cos(\pi) = -1 \implies \eta_{T_{\max}} = \eta_{\min} = 0$ (tinh chỉnh tỉ mỉ tại đáy cực tiểu).

#### e. Chỉ số Macro F1-Score
Với mỗi giống loài $c \in \{1, \dots, 37\}$:
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
$$F1_c = \frac{2 \times \text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} = \frac{2 TP_c}{2 TP_c + FP_c + FN_c}$$
Điểm **Macro F1** là trung bình số học không trọng số của 37 lớp:
$$\text{Macro F1} = \frac{1}{37} \sum_{c=1}^{37} F1_c$$
- **Ý nghĩa toán học:** Khác với Micro F1 hay Accuracy (vốn bị lấn át bởi các lớp đông mẫu), Macro F1 đánh giá công bằng tuyệt đối giữa tất cả 37 giống loài. Một giống dù ít mẫu nếu bị đoán sai vẫn kéo tụt Macro F1 ngay lập tức.

### 4. Input & Output của Cell
- **Input:** Cấu hình tham số, `splits["train"]`, `splits["blackbox"]`.
- **Output:** Các hàm `build_classifier()`, `train_classifier()`, `evaluate_model()`, lớp `PetCropsDataset`.

---

## CELL 12: TASK 5 - HUẤN LUYỆN BASELINE VÀ PHÂN TÍCH ĐỐI SÁNH (MODEL O VS MODEL Y)

### 1. Mục đích & Ý nghĩa
Huấn luyện 2 mô hình đối chứng với tham số và seed giống hệt nhau:
- **Model O (Oracle Baseline):** Nhận Bbox chuẩn trích xuất từ Trimap.
- **Model Y (YOLO Baseline):** Nhận Bbox do YOLOv8n dự đoán (Pipeline thực tế).
Việc so sánh 2 mô hình này giúp lượng hóa chính xác **tổn thất hiệu năng (Performance Drop)** do Bước A gây ra.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
# Huấn luyện Model O
model_oracle = build_classifier()
model_oracle, history_oracle = train_classifier(
    model_oracle, loader_train_O, loader_black_O, epochs=15, checkpoint_path="oracle_v1.pt"
)

# Huấn luyện Model Y
model_yolo = build_classifier()
model_yolo, history_yolo = train_classifier(
    model_yolo, loader_train_Y, loader_black_Y, epochs=15, checkpoint_path="yolo_v1.pt"
)
```
- Lưu lại danh sách nhãn dự đoán `preds_y_eye` của Model Y trên 300 mẫu Eyeball Dev để phục vụ trực tiếp cho Task 6.

### 3. Bản chất & Công thức Toán học liên quan

#### Phương pháp Bóc tách Thành phần (Component Ablation Analysis)
Trong một hệ thống chuỗi $f(x) = g(h(x))$ trong đó $h(x)$ là Detector và $g(\cdot)$ là Classifier:
- Điểm số của Pipeline thực tế: $S_Y = \text{Score}(g(h(x)))$.
- Điểm số của mô hình lý tưởng: $S_O = \text{Score}(g(h^*(x)))$ với $h^*(x)$ là Ground Truth Bbox.
Khoảng cách sai số do Bước A gây ra được định lượng trực tiếp bằng hiệu số:
$$\Delta_{\text{Detector Loss}} = S_O - S_Y$$
- **Ví dụ số thực tế:**
  - Model O đạt: $\text{Accuracy}_{\text{Blackbox}} = 88.5\%$.
  - Model Y đạt: $\text{Accuracy}_{\text{Blackbox}} = 84.8\%$.
  - $\Delta_{\text{Detector Loss}} = 88.5\% - 84.8\% = 3.7\%$.
  - **Ý nghĩa:** Bước A (YOLO) làm rơi rớt khoảng $3.7\%$ độ chính xác của toàn hệ thống do cắt lệch hoặc crop sát tai con vật.

### 4. Input & Output của Cell
- **Input:** Toạ độ Bbox của Oracle và YOLO cho toàn bộ tập `trainval`.
- **Output:** Checkpoints `oracle_v1.pt`, `yolo_v1.pt`, file kết quả `baseline.json`, danh sách dự đoán `preds_y_eye`.

---

## CELL 14: TASK 6 - BẢNG PHÂN TÍCH LỖI EYEBALL DEV (GẮN NHÃN TƯƠNG TÁC)

### 1. Mục đích & Ý nghĩa
Triển khai phương pháp luận của Andrew Ng: Lọc ra toàn bộ các mẫu dự đoán sai của Model Y trên tập Eyeball Dev, lấy ngẫu nhiên 100 mẫu sai đại diện, chạy thuật toán **tự động gắn nhãn thuộc tính vật lý** (`mờ`, `tối`, `vật thể nhỏ`), và cung cấp giao diện widget trực quan (`ipywidgets`) để con người bấm chọn nhãn ngữ nghĩa (`giống tương tự`, `che khuất`, `góc lạ`, `nhiều con`, `nhãn sai`).

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
def get_auto_quality_tags(pil_img, bbox):
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    W, H = pil_img.size
    tags = []
    # 1. Đo độ mờ bằng phương sai toán tử Laplacian
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 100.0:
        tags.append("mờ")
    # 2. Đo độ sáng bằng giá trị xám trung bình
    if gray.mean() < 60.0:
        tags.append("tối")
    # 3. Đo tỷ lệ diện tích Bbox so với ảnh gốc
    area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (W * H)
    if area_ratio < 0.20:
        tags.append("vật thể nhỏ")
    return tags
```
- Lấy mẫu 100 ảnh sai với fixed seed: `random.sample(wrong_indices, min(100, len(wrong_indices)))`.
- Giao diện `ipywidgets` hiển thị ảnh crop kèm thông tin: `Thực tế: Pug -> Đoán nhầm: French Bulldog`, các checkbox và nút bấm `"Lưu & Ảnh kế tiếp"`.

### 3. Bản chất & Công thức Toán học liên quan

#### a. Toán tử vi phân bậc hai Laplace và Phương sai đo độ mờ (Laplacian Variance)
Toán tử Laplace 2 chiều trên ma trận ảnh xám $I(x, y)$ là tổng đạo hàm riêng bậc hai:
$$\Delta I = \nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
Trong tính toán số (rời rạc), toán tử này được xấp xỉ bằng phép nhân chập với ma trận nhân tích chập (Laplacian kernel $3 \times 3$):
$$K_{\text{Laplace}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
Phương sai của ma trận kết quả phản ánh mức độ biến thiên biên cạnh:
$$\overline{\Delta I} = \frac{1}{N} \sum_{x, y} \Delta I(x, y)$$
$$\text{Var}(\Delta I) = \frac{1}{N} \sum_{x, y} \left( \Delta I(x, y) - \overline{\Delta I} \right)^2$$
- **Trực giác toán học:**
  - **Ảnh sắc nét:** Có ranh giới rõ ràng giữa con vật và nền $\implies$ Đạo hàm bậc hai tại các đường biên có giá trị rất lớn (dương/âm đột ngột) $\implies$ Dữ liệu phân tán mạnh $\implies \text{Var}(\Delta I) \ge 100.0$.
  - **Ảnh bị nhòe/mờ (Blur):** Các điểm ảnh chuyển tiếp thoai thoải, biên cạnh bị làm mịn $\implies$ Đạo hàm bậc hai xấp xỉ 0 ở khắp nơi $\implies \text{Var}(\Delta I) < 100.0$.

#### b. Độ sáng trung bình (Mean Intensity)
$$\mu_{\text{gray}} = \frac{1}{W \times H} \sum_{x=0}^{W-1} \sum_{y=0}^{H-1} I_{\text{gray}}(x, y)$$
- Giá trị pixel thuộc $[0, 255]$. Nếu $\mu_{\text{gray}} < 60.0$, bức ảnh thiếu sáng nghiêm trọng hoặc chụp ngược sáng.

#### c. Tỷ lệ diện tích tương đối (Relative Area Ratio)
$$r_{\text{area}} = \frac{(x_2 - x_1) \times (y_2 - y_1)}{W \times H}$$
- Nếu $r_{\text{area}} < 0.20$ (dưới 20% khung hình): Khi resize ảnh về $224 \times 224$, số lượng pixel biểu diễn khuôn mặt thú cưng quá ít (bị vỡ hình), dẫn đến mất các chi tiết đặc trưng.

### 4. Input & Output của Cell
- **Input:** Ảnh của các mẫu sai trên Eyeball Dev, toạ độ Bbox của YOLO.
- **Output:** File `error_table.csv` gồm 100 dòng với các cột `index`, `true_label`, `pred_label`, `tags_auto`, `tags_manual`.

---

## CELL 16: TASK 7 - PHÂN BỔ LỖI THEO PIPELINE (PIPELINE ERROR ATTRIBUTION)

### 1. Mục đích & Ý nghĩa
Xác định chính xác nguyên nhân gốc rễ của từng ca dự đoán sai: Lỗi là do **Bước A (Bộ phát hiện YOLO cắt hỏng)** hay do **Bước B (Bộ phân loại ResNet không đủ khả năng phân biệt giống loài)**.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
# Lỗi Model Y đoán sai nhưng Model O đoán đúng -> Lỗi do Bước A
fixed_by_oracle = [k for k in err_y_indices if preds_o_eye[k] == labels_eye[k]]

# Lỗi cả Model Y và Model O đều đoán sai -> Lỗi do Bước B
still_wrong = [k for k in err_y_indices if k in err_o_indices]
```
- `err_y_indices`: Danh sách các mẫu Model Y đoán sai trên Eyeball Dev.
- `err_o_indices`: Danh sách các mẫu Model O đoán sai trên Eyeball Dev.

### 3. Bản chất & Công thức Toán học liên quan

#### Đại số tập hợp trong Quy thuộc Lỗi (Set-Theoretic Error Attribution)
Gọi không gian mẫu trên Eyeball Dev là $\Omega$ ($|\Omega| = 300$).  
Gọi tập các mẫu đoán sai của Model Y là $E_Y \subset \Omega$, tập các mẫu đoán sai của Model O là $E_O \subset \Omega$.
1. **Tập lỗi quy thuộc Bước A ($E_{\text{Step A}}$):**
   $$E_{\text{Step A}} = E_Y \setminus E_O = \{i \in \Omega \mid i \in E_Y \text{ và } i \notin E_O\}$$
   - *Logic chứng minh:* Khi được cấp Bbox hoàn hảo $h^*(x)$ từ Trimap, mô hình đoán đúng ($i \notin E_O$). Nhưng khi dùng Bbox từ YOLO thì lại đoán sai ($i \in E_Y$). Suy ra $100\%$ nguyên nhân gây sai là do Bbox của Bước A bị lỗi!
2. **Tập lỗi quy thuộc Bước B ($E_{\text{Step B}}$):**
   $$E_{\text{Step B}} = E_Y \cap E_O = \{i \in \Omega \mid i \in E_Y \text{ và } i \in E_O\}$$
   - *Logic chứng minh:* Ngay cả khi được cung cấp Bbox hoàn hảo, mô hình vẫn đoán sai. Suy ra Bbox không phải là nguyên nhân, mà lỗi thuần túy nằm ở năng lực phân loại đặc trưng của Bước B.

#### Công thức tỷ lệ phần trăm phân bổ:
$$P(\text{Lỗi do Bước A}) = \frac{|E_{\text{Step A}}|}{|E_Y|} = \frac{|E_Y \setminus E_O|}{|E_Y|}$$
$$P(\text{Lỗi do Bước B}) = \frac{|E_{\text{Step B}}|}{|E_Y|} = \frac{|E_Y \cap E_O|}{|E_Y|}$$
- **Quy tắc ra quyết định chiến lược của Andrew Ng:**
  - Nếu $P(\text{Bước A}) > 35\%$: Nút thắt cổ chai nằm ở Detector. Tập trung tinh chỉnh Bước A (mở rộng lề crop, chỉnh ngưỡng).
  - Nếu $P(\text{Bước B}) > 65\%$: Detector đã làm tốt. Tập trung toàn bộ nguồn lực cải tiến Classifier (Data Augmentation, Backbone mạnh hơn).

### 4. Input & Output của Cell
- **Input:** `preds_y_eye`, `preds_o_eye`, `labels_eye`.
- **Output:** File `pipeline_attribution.json` lưu số lượng và tỷ lệ % lỗi của Bước A và Bước B.

---

## CELL 18: TASK 8 - PHÂN TÍCH TRẦN CẢI THIỆN (CEILING ANALYSIS)

### 1. Mục đích & Ý nghĩa
Tính toán **Trần cải thiện độ chính xác (Accuracy Ceiling Gain %)**: Trả lời câu hỏi kinh tế kỹ thuật quan trọng nhất của người quản lý dự án: *"Nếu đội ngũ kỹ sư bỏ ra 2 tuần để giải quyết triệt để 100% một nhóm lỗi cụ thể (ví dụ ảnh mờ, hoặc nhầm giống tương tự), độ chính xác của toàn hệ thống sẽ tăng tối đa thêm bao nhiêu điểm %?"*.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
for tag, count in tag_counts.most_common():
    share_in_sample = count / len(df_err_loaded)
    estimated_total_errors = share_in_sample * total_pipeline_errors
    accuracy_ceiling_gain = (estimated_total_errors / n_eyeball) * 100.0
```
- `tag_counts`: Thống kê tần số xuất hiện của từng nhãn lỗi trong 100 mẫu phân tích.
- Nhân tỷ lệ xuất hiện với tổng số lỗi thực tế trên Eyeball Dev để ngoại suy ra toàn tập.

### 3. Bản chất & Công thức Toán học liên quan

#### Công thức Ước lượng Ngoại suy Không chệch (Unbiased Point Estimation)
Gọi $n = 100$ là số lượng mẫu sai được chọn ngẫu nhiên đồng đều từ tổng số $E_{\text{total}}$ mẫu sai trên Eyeball Dev ($N_{\text{eyeball}} = 300$).  
Gọi $k_T$ là số lần xuất hiện của nhãn lỗi $T$ trong $n$ mẫu.
1. **Tỷ lệ của nhóm lỗi $T$ trong tập lỗi:**
   $$\hat{p}_T = \frac{k_T}{n}$$
2. **Số lượng mẫu ước tính mắc lỗi $T$ trên toàn bộ Eyeball Dev:**
   $$\widehat{\text{Count}}_T = \hat{p}_T \times E_{\text{total}} = \frac{k_T}{n} \times E_{\text{total}}$$
3. **Mức trần tăng độ chính xác tối đa (Accuracy Ceiling Gain):**
   $$\text{Ceiling Gain}(T) = \frac{\widehat{\text{Count}}_T}{N_{\text{eyeball}}} \times 100\% = \frac{\frac{k_T}{n} \times E_{\text{total}}}{N_{\text{eyeball}}} \times 100\%$$

- **Ví dụ số thực tế:**
  Giả sử trên tập Eyeball Dev có $E_{\text{total}} = 45$ ảnh bị đoán sai trên tổng số $N_{\text{eyeball}} = 300$ ảnh.  
  Trong $n = 45$ ảnh sai đem phân tích, có $k_{\text{mờ}} = 15$ ảnh bị nhãn `"mờ"`:
  - $\hat{p}_{\text{mờ}} = \frac{15}{45} \approx 33.3\%$.
  - $\widehat{\text{Count}}_{\text{mờ}} = 33.3\% \times 45 = 15$ ảnh.
  - $\text{Ceiling Gain}(\text{mờ}) = \frac{15}{300} \times 100\% = 5.0\%$.
  - **Ý nghĩa:** Nếu ta viết thêm module khử mờ hoàn hảo 100%, độ chính xác toàn hệ thống tối đa chỉ tăng thêm **5.0%** (từ $85.0\% \to 90.0\%$), không thể tăng hơn. Kỹ sư dựa vào bảng xếp hạng này để ưu tiên xử lý lỗi nào có trần cao nhất.

### 4. Input & Output của Cell
- **Input:** File `error_table.csv`, tổng số lỗi `total_pipeline_errors`.
- **Output:** File `ceiling.csv` chứa danh sách các nhóm lỗi được sắp xếp giảm dần theo tiềm năng cải thiện.

---

## CELL 20: TASK 9 - CẢI TIẾN CÓ MỤC TIÊU VÀ HUẤN LUYỆN MODEL V2

### 1. Mục đích & Ý nghĩa
Thực hiện triết lý **Data-Centric AI**: Áp dụng các kỹ thuật tăng cường dữ liệu đặc trị cho các nhóm lỗi đứng đầu bảng Ceiling (ảnh tối $\to$ ColorJitter, ảnh mờ $\to$ GaussianBlur, bị che $\to$ RandomErasing) và mở rộng lề cắt Bbox lên `margin=0.15` để khắc phục lỗi của Bước A. Sau đó huấn luyện Model v2 để đo lường mức độ tiến bộ.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
tf_train_targeted_v2 = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    # Đặc trị lỗi tối: Tăng biên độ sáng
    T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3),
    # Đặc trị lỗi mờ: Thêm Gaussian Blur với p=0.3
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=0.3),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    # Đặc trị lỗi che khuất: Xóa ngẫu nhiên hình chữ nhật
    T.RandomErasing(p=0.3, scale=(0.02, 0.20), value='random')
])
IMPROVED_MARGIN = 0.15 # Mở rộng lề cắt lên 15%
```
- Retrain mô hình với cùng seed 42, cùng số epoch 15, lưu checkpoint mới `v2.pt`.

### 3. Bản chất & Công thức Toán học liên quan

#### a. Bộ lọc làm nhòe Gauss 2 chiều (2D Gaussian Blur Kernel)
Phép lọc Gauss làm mờ ảnh bằng cách nhân chập ảnh với hàm mật độ xác suất của phân phối chuẩn 2 chiều:
$$G(x, y; \sigma) = \frac{1}{2\pi \sigma^2} \exp\left( - \frac{x^2 + y^2}{2\sigma^2} \right)$$
- Trong code: `kernel_size=5`, $\sigma \in [0.5, 2.0]$.
- Ma trận lọc kích thước $5 \times 5$ được chuẩn hóa để tổng các trọng số bằng 1: $\sum_{u, v} K(u, v) = 1$.
- **Ý nghĩa toán học:** Bằng cách chủ động làm nhòe ngẫu nhiên ảnh trong tập Train, mạng nơ-ron bị ép phải học các đặc trưng tần số thấp (low-frequency features như hình khối tổng thể, tỷ lệ xương sọ) thay vì chỉ dựa dẫm vào các đặc trưng tần số cao (sợi lông sắc nét). Khi gặp ảnh mờ ở tập Test, mô hình vẫn nhận diện chính xác!

#### b. Phép biến đổi Xóa ngẫu nhiên (Random Erasing Data Augmentation)
Chọn ngẫu nhiên một hình chữ nhật có diện tích $S_e$ chiếm tỷ lệ $s_e \in [0.02, 0.20]$ diện tích ảnh gốc, tỷ lệ khung hình $r_e \in [0.3, 3.3]$:
$$W_e = \sqrt{S_e \times r_e}, \quad H_e = \sqrt{\frac{S_e}{r_e}}$$
Gán toàn bộ các pixel trong vùng chữ nhật này bằng các giá trị ngẫu nhiên.
- **Ý nghĩa toán học:** Mô phỏng hiện tượng che khuất (Occlusion). Cưỡng ép mạng nơ-ron phân tán trọng số chú ý (Attention weights) ra nhiều vùng khác nhau của cơ thể con vật, tránh việc mô hình chỉ học thuộc duy nhất một vùng mắt hoặc mũi.

### 4. Input & Output của Cell
- **Input:** Dataset `trainval`, `yolo_bboxes`, pipeline transforms mới `tf_train_targeted_v2`.
- **Output:** Checkpoint `v2.pt`, file kết quả đối sánh `v2.json`.

---

## CELL 22: TASK 10 - KIỂM TRA OVERFITTING (DEV GAP) VÀ ĐÁNH GIÁ TẬP TEST

### 1. Mục đích & Ý nghĩa
1. **Kiểm tra hiện tượng "Human Overfitting":** Đo khoảng cách giữa tập đã soi bằng mắt (Eyeball Dev) và tập không soi (Blackbox Dev) để đảm bảo mô hình không bị "học tủ".
2. **Nguyên tắc niêm phong tập Test:** Chọn model tốt nhất theo Blackbox Dev và đánh giá trên tập Test (3.669 ảnh) **duy nhất 1 lần** để báo cáo kết quả trung thực.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
# Đo khoảng cách Overfitting Gap
gap_v1 = baseline_results["yolo_eyeball"]["accuracy"] - baseline_results["yolo_blackbox"]["accuracy"]
gap_v2 = v2_results["v2_eyeball"]["accuracy"] - v2_results["v2_blackbox"]["accuracy"]

# Chọn mô hình tốt nhất theo Blackbox Dev
best_model_name = "v2" if v2_results["v2_blackbox"]["macro_f1"] >= v2_results["v1_blackbox"]["macro_f1"] else "v1"
chosen_model = model_v2 if best_model_name == "v2" else model_yolo

# Chạy Test đúng 1 lần
test_acc, test_f1, test_preds, test_labels_true = evaluate_model(chosen_model, loader_test)
```

### 3. Bản chất & Công thức Toán học liên quan

#### Khoảng cách Tổng quát hóa (Generalization Gap / Human Overfitting Metric)
$$\Delta_{\text{Overfit}} = \text{Accuracy}(\text{Eyeball Dev}) - \text{Accuracy}(\text{Blackbox Dev})$$
- **Nguyên lý lý thuyết của Andrew Ng:**
  - Vì kỹ sư đã trực tiếp quan sát 100 ảnh lỗi của Eyeball Dev để tinh chỉnh augmentation/margin, mô hình có thể vô thức "quá khớp" vào tập này.
  - Blackbox Dev đóng vai trò biến kiểm soát độc lập (Hold-out Control Set).
  - **Tiêu chuẩn kiểm định:**
    $$\Delta_{\text{Overfit}} \le 0.05 \ (5.0\%) \implies \text{Tổng quát hóa tốt (Low Variance)}$$
    $$\Delta_{\text{Overfit}} > 0.05 \ (5.0\%) \implies \text{Có dấu hiệu Overfitting vào Eyeball Dev}$$

### 4. Input & Output của Cell
- **Input:** Checkpoint tốt nhất, Dataset `ds_test` (3.669 ảnh).
- **Output:** File `final_metrics.json` lưu giữ `test_accuracy`, `test_macro_f1` và các chỉ số `overfitting_gap`.

---

## CELL 24: TASK 11 - TỔNG HỢP BÁO CÁO TRỰC QUAN & MA TRẬN NHẦM LẪN

### 1. Mục đích & Ý nghĩa
Vẽ đồ thị trực quan so sánh tiến trình huấn luyện của Model v1 và Model v2, biểu đồ cột Ceiling Gain, và phân tích ma trận nhầm lẫn để tìm ra **Top 5 cặp giống loài hay bị nhầm lẫn nhất** trong toàn bộ 37 giống.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
# Phân tích các cặp hay nhầm nhất từ Confusion Matrix
cm = confusion_matrix(test_labels_true, test_preds)
np.fill_diagonal(cm, 0) # Xóa đường chéo chính

top_confused = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm[i, j] > 0:
            top_confused.append((CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j])))
top_confused = sorted(top_confused, key=lambda x: x[2], reverse=True)[:5]
```
- Lệnh `np.fill_diagonal(cm, 0)` xóa toàn bộ các ca đoán đúng nằm trên đường chéo, chỉ giữ lại các ca đoán sai.
- Sắp xếp giảm dần theo số lần nhầm lẫn để trích xuất 5 cặp đứng đầu.

### 3. Bản chất & Công thức Toán học liên quan

#### Ma trận Nhầm lẫn (Confusion Matrix) và Phép Triệt tiêu Đường chéo
Ma trận nhầm lẫn $\mathbf{C} \in \mathbb{R}^{K \times K}$ ($K = 37$) có phần tử $C_{i, j}$ biểu diễn số lượng mẫu thuộc giống thật $i$ nhưng bị mô hình dự đoán thành giống $j$:
$$C_{i, j} = \sum_{k=1}^{N_{\text{test}}} \mathbb{I}(y_k = i \text{ và } \hat{y}_k = j)$$
Trong đó $\mathbb{I}(\cdot)$ là hàm chỉ thị (Indicator function).
- Các phần tử trên đường chéo chính $C_{i, i}$ ($i = j$) là các trường hợp **dự đoán đúng** ($TP_i$).
- Phép triệt tiêu đường chéo:
  $$\mathbf{C}' = \mathbf{C} - \text{diag}(\mathbf{C}) \iff C_{i, j}' = \begin{cases} 0 & \text{nếu } i = j \\ C_{i, j} & \text{nếu } i \neq j \end{cases}$$
- Tìm cặp nhầm lẫn nghiêm trọng nhất:
  $$(i^*, j^*) = \arg\max_{i \neq j} C_{i, j}'$$
- **Ý nghĩa thực tế:** Lập tức cô lập các cặp giống cận huyết có hoa văn hoặc cấu trúc khuôn mặt gần như trùng nhau trong không gian đặc trưng (ví dụ: `Bengal -> Egyptian Mau`, `Pug -> French Bulldog`).

### 4. Input & Output của Cell
- **Input:** `test_labels_true`, `test_preds`, lịch sử huấn luyện `history_yolo`, `history_v2`.
- **Output:** Các biểu đồ `training_history.png`, `error_ceiling_chart.png`, và danh sách Top 5 cặp giống nhầm lẫn in ra console.

---

## CELL 26: TASK 12 - THỬ NGHIỆM ĐỘT PHÁ (MODEL V3 CHAMPION)

### 1. Mục đích & Ý nghĩa
Tập hợp toàn bộ **4 biện pháp cải tiến nâng cao** để giải quyết triệt để bài toán phân loại tinh các giống tương tự nhau:
1. Tăng kích thước ảnh lên **320x320** (tăng gấp đôi diện tích điểm ảnh).
2. Tối ưu Augmentation bảo tồn đặc trưng: `scale=(0.8, 1.0)` chống cắt mất tai, hạ `saturation=0.15` giữ màu mắt xanh và màu lông chuẩn.
3. Nâng cấp Backbone lên **ResNet-50** với vector embedding **2.048 chiều**.
4. Hạ `lr=1e-4` và tăng `label_smoothing=0.10`.

### 2. Các dòng lệnh cốt lõi & Cơ chế hoạt động
```python
# 1. Pipeline 320x320
IMAGE_SIZE_V3 = 320
tf_train_v3 = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE_V3, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.15),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))], p=0.25),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    T.RandomErasing(p=0.3, scale=(0.02, 0.15), value='random')
])

# 2. Backbone ResNet-50 2048 chiều
def build_classifier_v3(num_classes=37):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features # 2048
    model.fc = nn.Linear(in_features, num_classes)
    return model
```

### 3. Bản chất & Công thức Toán học liên quan

#### a. Tỷ lệ Khuếch đại Thông tin Điểm ảnh (Spatial Information Gain)
So sánh giữa kích thước ảnh cũ $224 \times 224$ và kích thước ảnh mới $320 \times 320$:
$$\text{Tỷ lệ diện tích} = \frac{320 \times 320}{224 \times 224} = \left(\frac{320}{224}\right)^2 = \left(\frac{10}{7}\right)^2 \approx 2.0408 \times$$
- **Ý nghĩa toán học:** Mô hình nhận được **gấp 2.04 lần lượng điểm ảnh (pixel)**. Đối với các chi tiết siêu nhỏ như nếp nhăn trán của Pug, đốm viền móng chân của Birman hay hoa văn đốm báo của Bengal, độ phân giải cao ngăn ngừa hiện tượng mất thông tin do phép nội suy thu nhỏ (Bilinear downsampling).

#### b. Không gian Biểu diễn Vector Đặc trưng Đa chiều (High-Dimensional Embedding Space)
- **ResNet-18:** Đầu ra trước tầng phân loại là vector $\mathbf{v}_{18} \in \mathbb{R}^{512}$.
- **ResNet-50:** Sử dụng kiến trúc Bottleneck Block ($1\times 1 \to 3\times 3 \to 1\times 1$ Conv), đầu ra là vector $\mathbf{v}_{50} \in \mathbb{R}^{2048}$.
- **Trực giác toán học:** Không gian 2.048 chiều có dung lượng biểu diễn (Representation Capacity) lớn gấp 4 lần không gian 512 chiều. Định lý hình học trong không gian nhiều chiều (High-dimensional Geometry) chỉ ra rằng: khi số chiều tăng lên, các cụm điểm dữ liệu có xu hướng trở nên **trực giao (orthogonal)** với nhau hơn. Nhờ đó, siêu phẳng phân chia (Hyperplane) của hàm Softmax có thể dễ dàng tách rời 2 giống tương tự nhau (`Pug` vs `French Bulldog`) mà không bị chồng lấn không gian đặc trưng.

### 4. Input & Output của Cell
- **Input:** Ảnh cắt từ YOLO, kích thước mới 320x320, backbone ResNet-50.
- **Output:** Checkpoint `v3_champion.pt`, file kết quả `v3_champion.json`, bảng tổng kết tiến trình 3 thế hệ mô hình (v1 $\to$ v2 $\to$ v3).

---

# PHẦN 3: BỘ 10 CÂU HỎI VẤN ĐÁP KỸ THUẬT & TOÁN HỌC THƯỜNG GẶP KHI BẢO VỆ

### Câu 1: Tại sao bài toán này bắt buộc phải dùng Macro F1 bên cạnh Accuracy?
**Trả lời:**  
Accuracy chỉ đo tỷ lệ đúng trên tổng thể, nên nếu mô hình đoán giỏi các giống đông mẫu nhưng đoán sai sạch các giống hiếm thì Accuracy vẫn cao (đánh lừa người làm mô hình). Macro F1 tính F1 riêng biệt cho từng giống rồi lấy trung bình cộng không trọng số:
$$\text{Macro F1} = \frac{1}{37} \sum_{c=1}^{37} F1_c$$
Điều này ép mô hình phải đối xử công bằng tuyệt đối giữa cả 37 giống loài (mỗi giống đóng góp đúng $1/37 \approx 2.7\%$).

---

### Câu 2: Trong Task 6, tại sao lại dùng phương sai toán tử Laplacian ($\text{Var}(\Delta I)$) để đo độ mờ mà không dùng chỉ số khác?
**Trả lời:**  
Toán tử Laplace là đạo hàm bậc hai của ma trận ảnh ($\nabla^2 I$). Ảnh sắc nét có sự thay đổi pixel đột ngột tại các đường viền $\implies$ gradient bậc hai rất lớn $\implies$ phương sai của ảnh sau khi lọc Laplace rất cao. Ngược lại, ảnh bị mờ/nhòe thì các điểm ảnh chuyển tiếp mịn màng $\implies$ gradient bậc hai xấp xỉ 0 $\implies$ phương sai cực thấp. Đây là phương pháp đo độ mờ không cần ảnh gốc đối chứng (No-Reference Image Quality Assessment) cực kỳ nhanh và chuẩn xác.

---

### Câu 3: Làm thế nào chứng minh được lỗi ở Task 7 là do Bước A (Detector) hay do Bước B (Classifier)?
**Trả lời:**  
Dựa trên phương pháp bóc tách thành phần đối chứng với Oracle (Trimap Bbox):
- Xét các ảnh mà Pipeline thực tế (Model Y) đoán sai:
  - Nếu đưa Bbox hoàn hảo của Trimap vào mà mô hình đoán **đúng** $\implies$ Lỗi $100\%$ do Bước A cắt lệch/cắt cụt.
  - Nếu đưa Bbox hoàn hảo vào mà mô hình **vẫn đoán sai** $\implies$ Lỗi thuộc về Bước B (bộ phân loại không nhận diện được giống).

---

### Câu 4: Công thức tính Trần cải thiện (Ceiling Analysis) ở Task 8 có ý nghĩa thực tế như thế nào?
**Trả lời:**  
Công thức:
$$\text{Ceiling Gain} = \frac{\frac{\text{Count}(T)}{n} \times E_{\text{total}}}{N_{\text{dev}}} \times 100\%$$
Nó đo lường mức tăng độ chính xác tối đa nếu loại bỏ triệt để một nhóm lỗi. Kỹ sư không làm việc theo cảm tính mà dựa vào bảng trần này: nhóm lỗi nào có trần cao nhất (ví dụ giống tương tự tăng được 4.5%, trong khi ảnh tối chỉ tăng được 0.8%) thì tập trung thời gian giải quyết nhóm đó trước để tối ưu hóa nguồn lực.

---

### Câu 5: Tại sao tổng các Ceiling Gain ở Task 8 cộng lại có thể vượt quá 100%?
**Trả lời:**  
Vì đây là bài toán **đa nhãn (Multi-label)**, các nhóm lỗi không loại trừ lẫn nhau (Non-mutually exclusive). Một bức ảnh hoàn toàn có thể đồng thời vừa bị `"mờ"`, vừa bị `"tối"`, vừa bị `"che khuất"`. Do đó, khi tính riêng lẻ từng loại, các phần giao nhau được tính lặp lại khiến tổng vượt quá $100\%$.

---

### Câu 6: Label Smoothing trong hàm mất mát Cross-Entropy ở Task 4 hoạt động ra sao và giải quyết vấn đề gì?
**Trả lời:**  
Label Smoothing biến đổi nhãn cứng one-hot $[0, 1, 0, \dots]$ thành nhãn mềm: nhãn đúng nhận $1 - \epsilon + \frac{\epsilon}{K}$, các nhãn sai nhận $\frac{\epsilon}{K}$. Nó ngăn mô hình đẩy logit ra vô cùng để tự tin thái quá ($p=1.0$), tạo ra ranh giới phân loại mượt mà giữa các lớp. Điều này đặc biệt quan trọng với các giống cận huyết có ngoại hình tương tự nhau.

---

### Câu 7: Khái niệm "Human Overfitting" ở Task 10 là gì và làm sao để phát hiện?
**Trả lời:**  
Human Overfitting xảy ra khi kỹ sư soi bằng mắt 100 ảnh lỗi của Eyeball Dev rồi vô thức thiết kế các bộ luật/augmentation để "chiều lòng" 100 ảnh đó, khiến mô hình chỉ giỏi trên Eyeball Dev mà kém ở dữ liệu thực tế. Ta phát hiện bằng cách đo khoảng cách Overfitting Gap với tập Blackbox Dev (tập chưa từng được soi bằng mắt):
$$\Delta_{\text{Overfit}} = \text{Accuracy}(\text{Eyeball}) - \text{Accuracy}(\text{Blackbox})$$
Nếu $\Delta_{\text{Overfit}} \le 5\%$, mô hình an toàn, tổng quát hóa tốt.

---

### Câu 8: Tại sao ở Task 12 khi nâng lên ResNet-50 và ảnh 320x320 lại phải giảm tốc độ học xuống $lr = 10^{-4}$?
**Trả lời:**  
Vì ResNet-50 đã được nạp sẵn trọng số rất tốt từ ImageNet (Pre-trained weights). Khi tinh chỉnh trên tập dữ liệu tương đối nhỏ (~2.873 ảnh) với độ phân giải lớn, nếu giữ tốc độ học cao ($3 \times 10^{-4}$), gradient lớn ở các bước đầu sẽ phá vỡ các đặc trưng nền tảng đã học của mạng. Tốc độ học nhỏ $10^{-4}$ giúp mạng tinh chỉnh chậm rãi, bảo tồn tri thức cũ và hội tụ mượt mà.

---

### Câu 9: Cosine Annealing Learning Rate Schedule có ưu điểm gì so với Step Decay thông thường?
**Trả lời:**  
Step Decay giảm tốc độ học đột ngột theo từng bậc thang, dễ làm mô hình rơi vào các điểm cực tiểu cục bộ hẹp (sharp local minima). Cosine Annealing giảm $lr$ mượt mà theo đường cong cosin, ở các epoch cuối $lr$ giảm dần về 0 một cách êm ái, giúp các tham số ổn định tại đáy của vùng cực tiểu rộng (flat minima), mang lại khả năng tổng quát hóa cao hơn trên tập Test.

---

### Câu 10: Tại sao tỷ lệ lề cắt `margin=0.15` ở Bước A lại đóng vai trò sống còn cho Bước B?
**Trả lời:**  
Trong phân loại 37 giống chó mèo, các đặc trưng phân biệt độc quyền thường nằm ở rìa ngoài cùng của cơ thể (như dáng tai dơi của Bull Pháp, tai cụp của Pug, chỏm lông đuôi, hoặc bờm cổ). Nếu crop sát mép Bbox ($\text{margin}=0$), các bộ phận này bị cắt đứt. Bước B dù thông minh đến đâu cũng không thể đoán đúng nếu đầu vào bị mất chi tiết nhận diện độc quyền (Garbage In $\to$ Garbage Out). Thêm 15% margin đảm bảo giữ trọn ngữ cảnh cục bộ của thú cưng.
