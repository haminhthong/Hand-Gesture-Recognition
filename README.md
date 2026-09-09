# Hand Gesture HCI Controller

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-FF6F00?logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
[![NumPy](https://img.shields.io/badge/NumPy-1.24--2.2-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyYAML](https://img.shields.io/badge/PyYAML-6%2B-2C3E50)](https://pyyaml.org/)
[![PyInstaller](https://img.shields.io/badge/Build-PyInstaller-0F7B93)](https://pyinstaller.org/)
[![Pytest](https://img.shields.io/badge/Test-Pytest-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![Ruff](https://img.shields.io/badge/Lint-Ruff-D7FF64?logo=ruff&logoColor=111111)](https://docs.astral.sh/ruff/)
[![Mypy](https://img.shields.io/badge/Type--check-Mypy-1674B1?logo=python&logoColor=white)](https://mypy-lang.org/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Nền tảng tương tác người–máy bằng cử chỉ tay, chạy cục bộ trên webcam. Hệ thống nhận diện một bàn tay bằng MediaPipe Hands, chuẩn hóa 21 landmark thành vector 63 chiều, phân loại tư thế tĩnh bằng model SVM nếu có artifact hoặc chuyển sang rule baseline, ổn định tín hiệu theo thời gian rồi phát sự kiện điều khiển canvas 2D.

README này là tài liệu chuẩn của repository: mọi mô tả dưới đây bám theo code hiện tại, cấu hình trong `configs/`, workflow CI và các lệnh CLI đang có.

## 1. Bài toán & phạm vi ứng dụng

### Bài toán

Một bộ phân loại từng frame chưa đủ để làm HCI usable. Tín hiệu webcam có thể bị nhiễu, nhãn thay đổi nhanh, một hành động có thể bị phát lặp ở mọi frame và cử chỉ động cần thứ tự theo thời gian. Dự án xử lý bốn điểm đó:

- Chuẩn hóa landmark theo cổ tay, kích thước lòng bàn tay, tay trái/phải và tùy chọn xoay trong mặt phẳng.
- Chuyển dự đoán thô thành trạng thái ổn định bằng activation dwell, release dwell và history horizon.
- Tách nhãn nhận diện khỏi `GestureEvent`; hành động một lần dùng rising edge và cooldown, kéo thả dùng event liên tục.
- Khi mất tay trong lúc kéo, phát hành `STOP_DRAG` và reset các bộ nhớ thời gian để không giữ trạng thái cũ.

### Phạm vi hiện tại

Trong phạm vi:

- Một bàn tay tại một thời điểm (`max_num_hands: 1`).
- Webcam OpenCV, cửa sổ GUI OpenCV và canvas vật thể 2D.
- Sáu nhãn tĩnh production: `Fist`, `Select`, `Options`, `Stop`, `Peace`, `NoAction`.
- Cử chỉ động production: chuỗi `On/Off`; `SOS` chỉ là experimental và mặc định tắt.
- Huấn luyện/đánh giá offline trên CSV landmark với split theo `subject_id`.
- Telemetry FPS, latency tổng và latency theo stage.

Ngoài phạm vi:

- Không điều khiển con trỏ hệ điều hành; con trỏ hiện tại chỉ là tọa độ ảo trong canvas.
- Không nhận diện nhiều bàn tay hoặc theo dõi danh tính bàn tay giữa các frame.
- Không lưu ảnh/video webcam trong pipeline thu thập; collector chỉ ghi landmark số.
- Không có model binary mặc định trong repository. Khi chưa có model, runtime dùng rule fallback nếu `fallback_to_rules: true`.

## 2. Cử chỉ và hành động

| Nhãn | Điều kiện/ý nghĩa trong code | Sự kiện ứng dụng |
| --- | --- | --- |
| `Fist` | Không có ngón nào được đánh dấu duỗi | Trạng thái nghỉ; nếu đang kéo thì kết thúc kéo |
| `Select` | Pinch cái–trỏ, ngón giữa tách khỏi trỏ và tổng ngón duỗi bằng 2 | `START_DRAG` một lần, sau đó `DRAG` mỗi frame |
| `Options` | Cái–trỏ và giữa cùng ở vùng pinch | `CHANGE_COLOR` tại rising edge |
| `Stop` | Cả 5 ngón duỗi | `DELETE_OBJECT` tại rising edge |
| `Peace` | Trỏ và giữa duỗi, cái gập, không phải pinch | `OPEN_MENU` tại rising edge |
| `NoAction` | Không đạt luật hoặc model bị reject | Không phát hành action |
| `On/Off` | Trỏ + giữa ở tư thế bắt đầu, sau đó giữa đổi trạng thái trong timeout | `TOGGLE_CANVAS` tại completion của FSM |
| `SOS` | 4 ngón (không có cái) rồi nắm đấm trong timeout | `EMERGENCY_SOS`, chỉ khi bật experimental |

`Wave`, `OK`, `Thumbs Up` và `Thumbs Down` còn xuất hiện trong một số module tương thích/benchmark cũ, nhưng không phải nhánh runtime canonical của `Main.py`. Không nên dùng chúng để mô tả khả năng production hiện tại.

## 3. Quy trình kỹ thuật canonical: logic, data flow và pipeline

Đây là flowchart duy nhất chi phối cách đọc source, cấu hình và báo cáo. Nhánh runtime chạy trên webcam; nhánh offline tạo dữ liệu/model/đánh giá và không tự động thay đổi runtime cho tới khi artifact được đặt đúng đường dẫn.

```mermaid
flowchart TD
    CFG[configs/runtime.yaml<br/>camera, thresholds, dwell, cooldowns] --> CLI[Main.py / hand-controller]
    TRAINCFG[configs/training.yaml<br/>labels, split, SVM candidates] --> OFFLINE

    subgraph OFFLINE[Offline data and model pipeline]
        COLLECT[scripts/collect_data.py<br/>canonical collector] --> CSV[data/raw/landmarks_dataset.csv<br/>metadata + 21 x 3 landmarks]
        SYNTH[scripts/generate_synthetic_dataset.py<br/>CI/smoke data] --> CSV
        CSV --> AUDIT[scripts/audit_data.py<br/>schema, NaN/Inf, duplicate ID, leakage]
        CSV --> SPLIT[scripts/create_splits.py<br/>subject-level Train/Val/Test]
        SPLIT --> TRAIN[scripts/train_static_model.py]
        CSV --> TRAIN
        TRAIN --> PREP_T[LandmarkPreprocessor<br/>wrist origin, palm scale, mirror, rotation]
        PREP_T --> SCALE[StandardScaler fit only on Train]
        SCALE --> SVM[Calibrated RBF-SVM<br/>threshold tuning on Val]
        SVM --> BUNDLE[models/static_gesture_svm_v1.joblib<br/>model + scaler + metadata + thresholds]
        BUNDLE --> EVAL[scripts/evaluate_static_model.py<br/>locked test report]
        CSV --> BASE[tools/train_baseline.py or<br/>experiments/compare_rule_knn_rf_svm.py]
        SPLIT --> BASE
    end

    subgraph RUNTIME[Runtime pipeline: one frame at a time]
        CAP[OpenCV VideoCapture] --> MIRROR[Optional horizontal flip]
        MIRROR --> MP[MediaPipe Hands<br/>max_num_hands = 1]
        MP --> OBS[HandObservation<br/>landmarks (21,3), handedness,<br/>palm_size, center, timestamp]
        OBS --> STATIC{Model artifact ready?}
        STATIC -->|yes| PREP_R[LandmarkPreprocessor from bundle config<br/>63D float32]
        PREP_R --> PRED[StaticGesturePredictor<br/>probability + reject policy]
        STATIC -->|no + fallback| RULE[RuleStaticBaseline<br/>geometry + precedence]
        STATIC -->|no + no fallback| NOACTION[NoAction / rejected]
        OBS --> DYN[DynamicGestureFSM<br/>On/Off; optional SOS]
        PRED --> STABLE[GestureStabilizer<br/>activation 120ms, release 100ms,<br/>history 250ms]
        RULE --> STABLE
        NOACTION --> STABLE
        STABLE --> MAP[GestureEventMapper<br/>edge/continuous + cooldown]
        DYN --> MAP
        OBS --> CURSOR[CursorFilter<br/>time-aware EMA, tau 0.08]
        MAP --> APP[Application state]
        CURSOR --> APP
        APP --> CANVAS[DraggableObjectManager<br/>visibility, z-order, drag, color, delete]
        APP --> MENU[ShapeMenu<br/>Peace opens; Select chooses shape]
        CANVAS --> DRAW[OpenCV render + HUD]
        MENU --> DRAW
        OBS --> DRAW
        DRAW --> TELEMETRY[PerformanceMonitor<br/>FPS, total latency, stage breakdown]
        TELEMETRY --> JSON[Optional benchmark-output JSON]
        OBS -->|missing beyond timeout| FAILSAFE[reset stabilizer/FSM<br/>STOP_DRAG if needed]
        FAILSAFE --> MAP
    end

    CLI --> CAP
    BUNDLE -. optional artifact .-> STATIC
    RUNTIME --> REPORT[Runtime telemetry / HCI benchmark output]
    OFFLINE --> REPORT
```

### 3.1 Luồng dữ liệu runtime

1. `VideoCapture.read()` tạo frame BGR. `mirror_camera` lật frame trước khi đưa vào MediaPipe.
2. `HandDetector.process()` chỉ lấy hand đầu tiên và trả `HandObservation`. Tọa độ `x`, `y`, `z` của MediaPipe vẫn ở hệ tọa độ normalized; `frame_width` và `frame_height` giữ kích thước ảnh thực tế.
3. Nhánh static dùng model bundle nếu tồn tại. Preprocessor phải đồng nhất với `preprocessor_config` trong bundle. Nếu model không sẵn sàng và fallback được bật, `RuleStaticBaseline` được dùng.
4. Nhánh dynamic nhận cùng `HandObservation` và chỉ phát nhãn khi FSM hoàn tất chuỗi.
5. `GestureStabilizer` lọc nhãn static theo thời gian. Cử chỉ cần giữ đủ `activation_dwell_ms`; khi vắng mặt đủ `release_dwell_ms`, trạng thái được trả về `NoAction`.
6. `GestureEventMapper` hợp nhất static và dynamic. `Select` là continuous; `Options`, `Stop`, `Peace`, `On/Off`, `SOS` là edge-triggered/cooldown.
7. `DraggableObjectManager` áp event trên canvas. `ShapeMenu` dùng đúng cursor đã lọc và nhận `OPEN_MENU` từ event mapper.
8. Mỗi frame được vẽ lại và telemetry ghi tổng latency cùng stage breakdown. Không có số benchmark cố định nào được hard-code vào README.

### 3.2 Chuẩn hóa landmark và feature contract

`LandmarkPreprocessor.transform()` nhận mảng `(21, 3)` và thực hiện theo thứ tự:

1. Trừ landmark cổ tay index `0`.
2. Chia cho `||landmark[9] - landmark[0]||` với epsilon `1e-6`.
3. Nếu `handedness == "Left"` và `mirror_left_hand=True`, đổi dấu trục X.
4. Nếu `normalize_rotation=True`, xoay mặt phẳng để vector cổ tay → Middle MCP hướng lên trục `-Y` của hệ ảnh.
5. Flatten thành vector `float32` 63 chiều.

`StandardScaler` chỉ được fit trên Train trong script huấn luyện; Val/Test chỉ gọi `transform`. Split luôn theo `subject_id`, không split ngẫu nhiên các frame liên tiếp của cùng một người.

## 4. Cấu trúc thư mục dự án

```text
.
├── Main.py                         # entrypoint chạy app từ root
├── data_collector.py               # wrapper tương thích cho collector
├── pyproject.toml                  # package, console scripts, pytest, Ruff, Mypy
├── requirements.txt                # dependency runtime/build
├── requirements-dev.txt            # runtime + test/lint/type-check
├── configs/
│   ├── runtime.yaml                # cấu hình camera, model, FSM, dwell, cooldown
│   └── training.yaml               # cấu hình offline SVM và labels
├── src/hand_gesture_controller/
│   ├── app.py                      # vòng lặp runtime và CLI
│   ├── config.py                   # RuntimeConfig/TrainingConfig
│   ├── schemas.py                  # HandObservation, Prediction, Event contracts
│   ├── data_collection.py          # collector canonical, có console entrypoint
│   ├── perception/                 # MediaPipe -> HandObservation
│   ├── features/                   # landmark preprocessing và motion utilities
│   ├── recognition/                # SVM predictor, rule baseline, dynamic FSM
│   ├── temporal/                   # cursor filter và gesture stabilizer
│   ├── events/                     # event mapping, edge, cooldown, failsafe
│   ├── application/                # canvas object manager và shape menu
│   └── telemetry/                  # FPS/latency monitor và JSON export
├── scripts/
│   ├── collect_data.py             # wrapper tới collector canonical
│   ├── generate_synthetic_dataset.py
│   ├── audit_data.py
│   ├── create_splits.py
│   ├── train_static_model.py
│   ├── evaluate_static_model.py
│   └── benchmark_hci.py
├── tools/
│   ├── collect_landmarks.py        # wrapper deprecated, giữ tương thích
│   └── train_baseline.py            # GroupKFold/LOSO baseline comparison
├── experiments/                    # so sánh Rule/KNN/RF/SVM
├── tests/                          # unit + integration tests
├── Image/                          # icon tài nguyên cho module FingerNumber tương thích
├── .github/workflows/ci.yml        # CI Windows/Linux, Python 3.10–3.12
└── LICENSE                         # MIT
```

Các thư mục phát sinh không phải source chính:

- `data/raw/`, `data/private/`: dữ liệu cục bộ, bị `.gitignore` loại khỏi Git.
- `data/processed/`: `splits.json` và `manifest.json` tạo bởi `create_splits.py`.
- `models/`: model bundle tạo bởi training script; không có sẵn mặc định.
- `reports/`: JSON/report cục bộ; bị ignore.
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`: cache công cụ, không commit.

## 5. Hướng dẫn cài đặt

Yêu cầu Python `3.10+`. CI hiện kiểm tra Python `3.10`, `3.11`, `3.12` trên Ubuntu và Windows.

```powershell
git clone <repository-url>
cd hand-gesture-recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Trên Linux/macOS, thay lệnh kích hoạt bằng `source .venv/bin/activate`. OpenCV GUI cần môi trường có display; collector và runtime cần webcam khi chạy thật.

## 6. Hướng dẫn chạy runtime

### Chạy nhanh bằng rule fallback

Model artifact là tùy chọn. Nếu chưa có `models/static_gesture_svm_v1.joblib`, chạy:

```powershell
python Main.py --config configs/runtime.yaml
```

Hoặc dùng console script sau khi `pip install -e .`:

```powershell
hand-controller --config configs/runtime.yaml
```

Tham số CLI ghi đè đúng trường được truyền vào; nếu bỏ qua `--camera`, `--width`, `--height` hoặc `--model`, giá trị trong YAML/default được giữ nguyên.

```powershell
python Main.py --camera 0 --width 640 --height 480
python Main.py --config configs/runtime.yaml --model models/static_gesture_svm_v1.joblib
python Main.py --config configs/runtime.yaml --benchmark-output reports/runtime.json
```

Phím tắt trong cửa sổ:

- `Q`: thoát và giải phóng camera/MediaPipe.
- `D`: bật/tắt HUD debug.
- `C`: xóa toàn bộ object trong canvas.

Canvas bắt đầu ở trạng thái ẩn. Thực hiện chuỗi `On/Off` để bật/tắt canvas. Khi canvas hiện:

- `Peace` mở menu hình học.
- `Select` trên một object để kéo; `Select` trên item menu để tạo `Rectangle`, `Circle`, `Triangle` hoặc `Star`.
- `Options` đổi màu object dưới cursor.
- `Stop` xóa object dưới cursor.

## 7. Luồng dữ liệu, thu thập và huấn luyện offline

### 7.1 Tạo dữ liệu

Collector canonical ghi một dòng cho mỗi mẫu, gồm 12 trường metadata và 63 trường tọa độ:

`sample_id`, `subject_id`, `session_id`, `sequence_id`, `gesture`, `frame_index`, `timestamp_ms`, `handedness`, `camera_width`, `camera_height`, `device_id`, `lighting`, sau đó là `x0,y0,z0,...,x20,y20,z20`.

Thu thập từ webcam:

```powershell
python scripts/collect_data.py `
  --subject-id subject_001 `
  --session-id session_001 `
  --label Select `
  --samples 200 `
  --interval 150 `
  --output data/raw/landmarks_dataset.csv
```

Phím trong collector: `S` bật/tắt burst, `N` sang sequence mới, `Q` kết thúc. Mỗi người nên có nhiều session và điều kiện ánh sáng; không dùng tên thật trong `subject_id`.

Tạo dữ liệu tổng hợp phục vụ smoke test, không dùng thay cho dữ liệu người thật để báo cáo độ chính xác:

```powershell
python scripts/generate_synthetic_dataset.py
```

### 7.2 Audit và split chống leakage

```powershell
python scripts/audit_data.py --csv data/raw/landmarks_dataset.csv
python scripts/create_splits.py `
  --csv data/raw/landmarks_dataset.csv `
  --splits-out data/processed/splits.json `
  --manifest-out data/processed/manifest.json
```

Audit kiểm tra schema, NaN/Inf, duplicate `sample_id` và disjointness của split. `create_splits.py` dùng seed `42` và chia `Train/Val/Test` ở cấp người (`subject_id`). Điều kiện bắt buộc:

`Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`.

### 7.3 Huấn luyện model production

```powershell
python scripts/train_static_model.py --config configs/training.yaml
```

Script sẽ:

1. Đọc CSV và giữ các labels trong `training.yaml`.
2. Nạp hoặc tạo subject split.
3. Fit `StandardScaler` trên Train.
4. Dùng GroupKFold trên Train để tìm `C` và `gamma` cho RBF-SVM.
5. Fit SVM hiệu chuẩn xác suất trên Train.
6. Dò ngưỡng reject trên Val theo Macro-F1, actionable precision và false-action rate.
7. Đánh giá một lần trên Locked Test.
8. Gói model, scaler, labels, thresholds, preprocessing config, manifest hash và metrics vào `models/static_gesture_svm_v1.joblib`.

Đánh giá artifact trên CSV test:

```powershell
python scripts/evaluate_static_model.py `
  --model models/static_gesture_svm_v1.joblib `
  --test-csv data/raw/landmarks_dataset.csv
```

So sánh baseline subject-independent:

```powershell
python tools/train_baseline.py `
  --dataset data/raw/landmarks_dataset.csv `
  --cv groupkfold `
  --splits 5 `
  --compare-rules

python experiments/compare_rule_knn_rf_svm.py `
  --csv data/raw/landmarks_dataset.csv `
  --cv groupkfold `
  --splits 5
```

## 8. Cấu hình chính

### Runtime (`configs/runtime.yaml`)

| Nhóm | Trường | Ý nghĩa |
| --- | --- | --- |
| Camera | `camera_index`, `target_width`, `target_height`, `mirror_camera` | Nguồn và cách hiển thị frame |
| MediaPipe | `min_detection_confidence`, `min_tracking_confidence`, `max_num_hands` | Độ tin cậy và giới hạn một tay |
| Model | `model_path`, `fallback_to_rules`, `default_accept_threshold`, `accept_thresholds` | Model ưu tiên, fallback và reject policy |
| Temporal | `activation_dwell_ms`, `release_dwell_ms`, `history_horizon_ms` | Ổn định nhãn theo thời gian |
| Failsafe | `hand_loss_timeout_ms` | Reset khi mất tay |
| Dynamic FSM | `on_off_timeout_seconds`, `sos_timeout_seconds`, `enable_experimental_gestures` | Timeout và bật SOS thử nghiệm |
| Cursor | `cursor_tau` | Hằng số time-aware EMA |
| Events | `cooldowns.*` | Cooldown theo `GestureEvent` |
| Telemetry | `show_debug_hud`, `benchmark_output` | HUD và JSON hiệu năng |

YAML dùng key chuỗi như `change_color`; `GestureEventMapper` chuẩn hóa chúng về enum trước khi chạy. Ngưỡng runtime trong YAML được ưu tiên khi app khởi tạo predictor.

### Training (`configs/training.yaml`)

Các trường quan trọng là `dataset_path`, `manifest_path`, `splits_path`, `model_output_path`, `mirror_left_hand`, `normalize_rotation`, `labels`, `cv_splits`, `c_candidates`, `gamma_candidates`, `target_precision_actionable` và `max_false_action_rate`.

## 9. Kiểm thử và CI

Chạy các kiểm tra tương ứng CI từ root repository:

```powershell
python -m compileall -q src/ scripts/ tools/ tests/
python -m ruff check .
python -m mypy src/
python -m pytest -q
python -m pytest --cov=src/hand_gesture_controller tests/
```

Workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) chạy trên Ubuntu và Windows với Python `3.10–3.12`, cài dependency runtime/dev, kiểm tra `pip check`, compile source, Ruff, Mypy theo đúng phiên bản Python của từng matrix job và Pytest coverage. Webcam không được yêu cầu trong CI; các test logic dùng dữ liệu tổng hợp/mocks.

## 10. Báo cáo, telemetry và giới hạn diễn giải

`PerformanceMonitor` ghi các stage: `frame_capture`, `mediapipe`, `preprocess`, `static_classifier`, `dynamic_fsm`, `temporal_filter`, `event_mapper` và `render`. Dùng `--benchmark-output reports/runtime.json` để lưu summary gồm `total_frames`, FPS trung bình, latency mean/p50/p95 và stage breakdown.

Không đưa số accuracy, FPS hay HCI success rate giả định vào README. Muốn báo cáo số liệu, hãy chạy đúng script trên dataset và phần cứng tương ứng, lưu artifact/report cùng thông tin dataset manifest.

Giới hạn hiện tại:

- MediaPipe có thể suy luận sai khi tay bị che khuất, ánh sáng kém hoặc xoay ngoài mặt phẳng lớn.
- Rule baseline là heuristic 2D; model SVM chỉ đáng tin khi dataset bao phủ đủ subject/session.
- Canvas và menu là GUI OpenCV trong cửa sổ local, chưa phải accessibility layer của hệ điều hành.
- `SOS` là experimental; `Wave` chỉ còn trong compatibility detector, chưa nằm trong FSM runtime canonical.

## 11. Repo cleanliness và license

Source, config, test và workflow được giữ trong Git. Dataset raw, model binary, report và cache bị ignore để tránh commit dữ liệu cá nhân hoặc artifact lớn. Collector cũ trong `tools/collect_landmarks.py` chỉ là wrapper; implementation duy nhất nằm ở `src/hand_gesture_controller/data_collection.py`.

Dự án phát hành theo [MIT License](LICENSE).
