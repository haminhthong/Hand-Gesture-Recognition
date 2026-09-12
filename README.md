# Hand Gesture HCI Controller

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-FF6F00?logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![CI](https://img.shields.io/badge/CI-Passing-brightgreen?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

Ứng dụng tương tác người–máy (HCI) thời gian thực qua webcam bằng cử chỉ tay. Hệ thống trích xuất 21 landmarks bàn tay qua MediaPipe, chuẩn hóa hình học không gian, phân loại cử chỉ tĩnh bằng RBF-SVM (kèm Rule Baseline đối chứng), ổn định nhãn bằng bộ lọc trễ thời gian (temporal hysteresis), ánh xạ sự kiện sườn lên (rising-edge event mapping) và điều khiển canvas 2D trực quan.

---

## 1. Kiến trúc hệ thống

Dự án tập trung vào luồng kỹ thuật thực chất từ nhận thức thị giác máy tính đến tương tác giao diện:

```mermaid
flowchart TD
    FRAME[Webcam Frame] --> MP[MediaPipe Hands\n21 Landmarks (x, y, z)]
    MP --> NORM[Landmark Normalization\nWrist Origin + Palm Scale + Mirror + Rotation]
    NORM --> VEC[63D Feature Vector]
    
    VEC --> SVM[Static Gesture Classifier\nRBF-SVM + Confidence Threshold]
    NORM -.-> RULE[Rule Baseline\nGeometric Heuristic]
    MP --> FSM[Dynamic Gesture FSM\nOn/Off Sequence]
    
    SVM --> STABLE[Temporal Stabilizer\nActivation Dwell 120ms | Release Dwell 100ms]
    RULE -.-> STABLE
    
    STABLE --> MAPPER[Gesture Event Mapper\nRising Edge + Continuous Drag + Hand-loss Failsafe]
    FSM --> MAPPER
    
    MAPPER --> CANVAS[Interactive Canvas 2D\nObject Drag, Color, Delete, Shape Menu]
```

### Quy trình xử lý theo từng bước:
1. **Perception**: MediaPipe Hands xác định 21 tọa độ landmark 3D của bàn tay.
2. **Landmark Preprocessing**: 5 bước chuẩn hóa hình học triệt tiêu ảnh hưởng của vị trí, kích thước, tay trái/phải và góc nghiêng camera.
3. **Static Classifier & Rule Baseline**: Mô hình RBF-SVM dự đoán cử chỉ với ngưỡng tin cậy xác suất; Rule Baseline dựa trên hình học ngón tay phục vụ đối chứng.
4. **Dynamic Gesture FSM**: Máy trạng thái hữu hạn theo dõi chuỗi chuyển động theo thời gian để nhận diện cử chỉ On/Off.
5. **Temporal Stabilizer**: Bộ lọc trễ hai chiều (activation/release dwell) loại bỏ rung giật nhãn (flickering/jitter).
6. **Event Mapper**: Cơ chế rising-edge ngăn ngừa việc kích hoạt sự kiện liên tục mỗi frame; duy trì trạng thái kéo thả liên tục và failsafe tự động nhả khi mất dấu bàn tay.
7. **Canvas UI**: Cửa sổ tương tác 2D hỗ trợ tạo hình, di chuyển, đổi màu và xóa đối tượng.

---

## 2. Tiền xử lý Landmark (Core ML Engineering)

Thay vì đưa trực tiếp tọa độ thô vào mô hình khiến mô hình học phụ thuộc vào vị trí camera, `LandmarkPreprocessor` thực hiện 5 bước biến đổi hình học:

```text
Raw Landmarks (21, 3)
         ↓
1. Tịnh tiến cổ tay (Wrist index 0) về gốc tọa độ (0, 0, 0)
         ↓
2. Chuẩn hóa tỉ lệ theo kích thước lòng bàn tay: ||landmark[9] - landmark[0]||
         ↓
3. Lật tay trái (đổi dấu trục X) để dùng chung không gian đặc trưng với tay phải
         ↓
4. Xoay chuẩn hóa trong mặt phẳng (vector cổ tay → khớp MCP giữa hướng lên)
         ↓
Flatten → Vector đặc trưng 63 chiều
```

> **Ý nghĩa thực tế**: Giúp mô hình tập trung học hình dạng và thế ngón tay thay vì khoảng cách xa/gần hay vị trí bàn tay trên khung hình camera.

---

## 3. Cử chỉ & Bảng ánh xạ sự kiện HCI

Hệ thống hỗ trợ 6 cử chỉ tĩnh và 1 cử chỉ động:

| Cử chỉ | Loại | Ý nghĩa / Tư thế | Sự kiện HCI | Hành vi giao diện |
| :--- | :--- | :--- | :--- | :--- |
| **`Select`** | Static | Pinch ngón cái & trỏ | `START_DRAG` / `DRAG` | Bắt đầu kéo và di chuyển đối tượng |
| **`Fist`** | Static | Nắm chặt cả bàn tay | `STOP_DRAG` | Nhả đối tượng đang kéo |
| **`Options`** | Static | Pinch cái–trỏ và ngón giữa | `CHANGE_COLOR` | Đổi màu đối tượng (kích hoạt 1 lần - rising edge) |
| **`Stop`** | Static | Xòe 5 ngón tay | `DELETE_OBJECT` | Xóa đối tượng dưới con trỏ (rising edge) |
| **`Peace`** | Static | Ngón trỏ & giữa giơ thẳng | `OPEN_MENU` | Mở menu lựa chọn hình khối (rising edge) |
| **`NoAction`** | Static | Trạng thái nghỉ hoặc bị từ chối | `NONE` | Không kích hoạt hành động |
| **`On/Off`** | Dynamic | Chuỗi gập/duỗi ngón giữa trong thời gian quy định | `TOGGLE_CANVAS` | Bật/tắt hiển thị canvas tương tác |

### Cơ chế HCI giải quyết vấn đề thực tế:
- **Rising Edge Triggering**: Khi người dùng giơ cử chỉ `Stop` hoặc `Peace` trong 30 frame liên tiếp, menu hoặc lệnh xóa chỉ phát **đúng 1 lần** ở frame đầu tiên cử chỉ đạt trạng thái ổn định, kèm khoảng chờ cooldown để chống spam.
- **Continuous Drag State**: Cử chỉ `Select` phát sự kiện `START_DRAG` ở sườn lên, sau đó duy trì `DRAG` mỗi frame để cập nhật vị trí vật thể theo con trỏ chuột ảo.
- **Hand-loss Failsafe**: Nếu camera đột ngột mất dấu bàn tay trong lúc đang kéo vật thể, hệ thống tự động phát sinh `STOP_DRAG` và reset toàn bộ trạng thái để vật thể không bị kẹt.

---

## 4. Thực nghiệm & So sánh mô hình

Mô hình được đánh giá theo phương pháp **Subject-independent Evaluation** (chia Train/Val/Test hoàn toàn rời rạc theo đối tượng thu thập `subject_id` để ngăn ngừa hiện tượng rò rỉ dữ liệu - data leakage).

Kết quả kiểm thử chéo 5-fold GroupKFold trên tập dữ liệu thực tế (`data/raw/landmarks_dataset.csv`):

| Mô hình | Đặc trưng | Accuracy (Mean ± Std) | Macro-F1 | Nhận xét kỹ thuật |
| :--- | :--- | :---: | :---: | :--- |
| **Rule Baseline** | Heuristic hình học | 0.7219 ± 0.0201 | 0.6670 | Nhanh, không cần huấn luyện nhưng khó bao quát biến thiên kích thước bàn tay. |
| **KNN (k=5)** | Khoảng cách Euclidean | 0.8302 ± 0.0153 | 0.8268 | Khá, nhưng nhạy cảm với khoảng cách cục bộ và tốc độ chậm khi tập dữ liệu lớn. |
| **Random Forest (n=100)** | Ensemble cây quyết định | 0.9964 ± 0.0044 | 0.9964 | Độ chính xác cao, nhưng kích thước mô hình cồng kềnh, độ trễ suy luận lớn hơn. |
| **RBF-SVM (C=10)** | Kernel RBF + Scaler | **0.9945 ± 0.0077** | **0.9945** | **Được chọn**: Kích thước gọn nhẹ (~400KB), suy luận < 1ms, hỗ trợ tính xác suất `predict_proba`. |

---

## 5. Cấu trúc thư mục

Toàn bộ dự án được tổ chức phẳng và tinh gọn:

```text
hand-gesture-recognition/
├── configs/
│   ├── runtime.yaml           # Cấu hình camera, stabilizer, ngưỡng tin cậy
│   └── training.yaml          # Cấu hình huấn luyện mô hình
├── data/
│   ├── raw/                   # landmarks_dataset.csv (dữ liệu thu thập)
│   └── processed/             # splits.json (phân chia subject-independent)
├── experiments/
│   └── compare_models.py      # So sánh đối chứng Rule vs KNN vs RF vs SVM
├── models/
│   └── static_gesture_svm.joblib # Model artifact gọn nhẹ (model, scaler, config)
├── reports/
│   └── evaluation.json        # Báo cáo đánh giá chi tiết
├── scripts/
│   ├── collect_data.py        # Thu thập landmark MediaPipe theo subject
│   ├── prepare_data.py        # Kiểm tra tính toàn vẹn và tạo subject split
│   ├── train.py               # Huấn luyện SVM với GroupKFold
│   └── evaluate.py            # Đánh giá độc lập trên tập test
├── src/
│   └── hand_gesture_controller/
│       ├── app.py             # Vòng lặp runtime chính và xử lý CLI
│       ├── canvas.py          # Quản lý vật thể 2D, menu và tương tác
│       ├── classifier.py      # Bộ phân loại SVM + Confidence Threshold
│       ├── config.py          # Quản lý cấu hình Dataclass
│       ├── data_collection.py # Logic thu thập dữ liệu landmark
│       ├── dynamic_gesture.py # Máy trạng thái FSM nhận diện cử chỉ On/Off
│       ├── event_mapper.py    # Ánh xạ cử chỉ thành sự kiện HCI (Rising Edge, Drag)
│       ├── hand_detector.py   # Wrapper MediaPipe Hands trích xuất 21 landmarks
│       ├── preprocessing.py   # Chuẩn hóa landmark (Wrist, Scale, Mirror, Rotation)
│       ├── rule_baseline.py   # Bộ phân loại luật hình học đối chứng
│       ├── schemas.py         # Dataclass và Enum định nghĩa kiểu dữ liệu
│       └── telemetry.py       # Đo đạc FPS và độ trễ khung hình
├── tests/                     # Unit test và integration test suite
├── Main.py                    # Entrypoint khởi chạy ứng dụng
├── pyproject.toml             # Khai báo gói và cấu hình công cụ phát triển
└── requirements.txt           # Thư viện phụ thuộc
```

---

## 6. Cài đặt & Khởi chạy

### Cài đặt môi trường

Yêu cầu Python 3.10 hoặc 3.11:

```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
pip install -e .
```

### Khởi chạy ứng dụng

Ứng dụng hỗ trợ hai chế độ hoạt động rõ ràng qua tham số `--mode`:

1. **Chế độ SVM (Mặc định - Sử dụng Machine Learning)**:
   ```bash
   python Main.py --mode svm --model models/static_gesture_svm.joblib
   ```
   *Lưu ý: Nếu không tìm thấy file model, ứng dụng sẽ báo lỗi rõ ràng và dừng lại thay vì âm thầm chuyển chế độ.*

2. **Chế độ Rule Baseline (Sử dụng luật hình học)**:
   ```bash
   python Main.py --mode rules
   ```

**Phím tắt tương tác trên cửa sổ:**
- `Q`: Thoát ứng dụng.
- `D`: Bật / tắt HUD hiển thị thông tin debug (FPS, trạng thái cử chỉ).
- `C`: Xóa toàn bộ vật thể trên canvas.

---

## 7. Pipeline huấn luyện & Đánh giá Offline

Quy trình 4 bước huấn luyện và kiểm thử độc lập:

```bash
# 1. Thu thập dữ liệu mẫu từ camera
python scripts/collect_data.py --subject-id subject_001 --label Select --samples 200

# 2. Kiểm tra dữ liệu và phân chia tập train/val/test theo subject
python scripts/prepare_data.py --csv data/raw/landmarks_dataset.csv

# 3. Huấn luyện mô hình RBF-SVM với GroupKFold
python scripts/train.py --csv data/raw/landmarks_dataset.csv --splits-json data/processed/splits.json

# 4. Đánh giá mô hình trên tập dữ liệu kiểm thử độc lập
python scripts/evaluate.py --model models/static_gesture_svm.joblib --csv data/raw/landmarks_dataset.csv
```

---

## 8. Trả lời 7 câu hỏi phỏng vấn then chốt

1. **MediaPipe Hands trả về những gì?**
   - MediaPipe Hands trả về 21 điểm landmark 3D đại diện cho các khớp xương trên bàn tay. Mỗi điểm gồm tọa độ $(x, y)$ chuẩn hóa trong khoảng $[0, 1]$ theo kích thước ảnh và $z$ đại diện cho độ sâu tương đối so với cổ tay.

2. **Tại sao cần chuẩn hóa landmark trước khi đưa vào mô hình?**
   - Tọa độ thô phụ thuộc hoàn toàn vào vị trí đứng, khoảng cách đến camera và kích thước bàn tay người dùng. Bằng cách tịnh tiến cổ tay về gốc $(0,0,0)$ và chia cho kích thước lòng bàn tay, mô hình học được đặc trưng thuần túy về hình học cử chỉ. Việc lật tay trái giúp dùng chung một mô hình phân loại cho cả hai tay.

3. **Tại sao lựa chọn SVM thay vì Deep Learning hay Rule-based?**
   - Không gian đặc trưng sau chuẩn hóa là vector dạng bảng 63 chiều. RBF-SVM giải quyết bài toán phi tuyến trong không gian này cực kỳ hiệu quả, trọng số gọn nhẹ (~400KB), độ trễ suy luận dưới 1ms trên CPU và không gặp rủi ro overfitting lớn như mạng nơ-ron sâu khi tập dữ liệu có quy mô vừa phải.

4. **Tại sao cần chia dữ liệu theo `subject_id` (Subject-independent split)?**
   - Để kiểm thử khả năng tổng quát hóa thực sự của mô hình đối với người dùng mới. Nếu chia ngẫu nhiên các frame liên tiếp của cùng một người vào cả train và test, mô hình sẽ đạt điểm số ảo rất cao do các frame có bàn tay, tư thế và bối cảnh camera gần như đồng nhất.

5. **Tại sao cần có Temporal Stabilizer (Hysteresis)?**
   - Dự đoán thô trên từng frame thường xuyên bị nhiễu và nhảy nhãn cục bộ do tay chuyển động nhanh hoặc camera mờ nhòe. Cơ chế trễ thời gian (cần giữ ổn định 120ms để kích hoạt và 100ms vắng mặt để giải phóng) loại bỏ hoàn toàn hiện tượng rung giật giao diện.

6. **Cơ chế Rising Edge hoạt động như thế nào trong Event Mapper?**
   - Khi một cử chỉ tĩnh (như `Peace` hay `Stop`) được giữ liên tục trong 30-60 frame, cơ chế rising edge chỉ phát tín hiệu sự kiện `OPEN_MENU` hoặc `DELETE_OBJECT` duy nhất một lần tại frame đầu tiên trạng thái được xác nhận, ngăn chặn việc giao diện bị lặp hành động ngoài ý muốn.

7. **Sự khác biệt giữa Static Gesture và Dynamic Gesture là gì?**
   - Cử chỉ tĩnh (Static Gesture) được định nghĩa dựa trên hình dạng bàn tay tại một frame đơn lẻ. Cử chỉ động (Dynamic Gesture) đòi hỏi một chuỗi thay đổi trạng thái theo thứ tự thời gian kèm ràng buộc về thời gian chờ (timeout) thông qua máy trạng thái hữu hạn FSM.

---

## Giấy phép

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
