# Ground-Truth Asset Layer Design

> Date: 2026-04-03
> Status: Draft
> Scope: 設計可重用的人工標註資產層，讓後續所有 evaluation 共用同一份 human ground truth，而不是在每個實驗階段重複標註。

## 1. 背景與問題定義

目前專案已具備以下能力：

1. `Phase 1`：從 COCO 篩出符合條件的多動物圖片，並標準化為 dataset 結構。
2. `Phase 2`：對每個動物預測左右眼座標。
3. `Phase 3 baseline`：從已預測的雙眼座標推導雙眼距離與前後距離 proxy。
4. `Evaluation Framework`：可對預測結果產生統計與報表。

目前的缺口是：

- COCO 不提供動物雙眼 ground-truth keypoints。
- COCO 不提供畫面內多隻動物的前後深度排序 ground truth。
- 若把人工標註放進每一個 evaluator 或每一次實驗流程中，將造成大量重複工作，且無法形成可重用的資產。

因此需要將人工標註獨立設計為一層可持續保存、可增量更新、可被多個 evaluation 共用的 `ground-truth asset layer`。

## 2. 設計目標

本設計稿的目標如下：

1. 讓人工標註只做一次，未來所有實驗共用。
2. 讓 `dataset`、`human ground truth`、`prediction`、`evaluation` 四層責任分離。
3. 讓 `run_annotation.py` 成為獨立的 GT 建置工具，而不是 evaluator 的附屬流程。
4. 讓標註結果可被版本化、可追蹤、可增量續標。
5. 讓後續新增新的 evaluator 時，不需要重新做人工作業。

## 3. 非目標

本設計稿目前不處理以下事項：

1. 多人協作標註衝突解決機制。
2. GUI 標註工具。
3. 真實物理深度的恢復或相機標定。
4. 主動學習、自動挑圖或半自動標註。

## 4. 核心原則

1. `Dataset Asset` 與 `Human GT` 必須分開保存。
2. `Prediction` 不得覆寫 `Human GT`。
3. `Evaluation` 只負責 join 與評估，不負責標註。
4. 所有資產都必須有穩定主鍵：
   - `dataset_id`
   - `image_id`
   - `annotation_id`
5. 標註結果必須支援增量保存與續標。
6. 單一 object 對應單一 human-label row，避免把整張圖的標註塞進單一 blob 欄位。

## 5. 系統分層

整體系統採四層設計：

### 5.1 A 層：Dataset Asset Layer

用途：
- 保存 Phase 1 過濾與標準化後的資料集。

來源：
- `run_data_pipeline.py`

內容：
- `image_id`
- `image_path`
- `annotation_id`
- `category`
- `bbox`
- `mask/contour metadata`

角色：
- 作為所有下游流程的共同基底。

### 5.2 B 層：Human Ground-Truth Layer

用途：
- 保存人工標註的左右眼與深度排序真值。

來源：
- `run_annotation.py`

內容：
- 每個 object 的 `left_eye`, `right_eye`, `depth_rank`

角色：
- 作為 localization / measurement / ranking evaluator 共用的 GT 資產。

### 5.3 C 層：Prediction Layer

用途：
- 保存模型或演算法輸出。

來源：
- `run_eye_detection.py`
- 未來的 measurement predictor
- 其他模型實驗

內容：
- `pred_left_eye`, `pred_right_eye`
- `pred_eye_distance_px`
- `pred_front_back_proxy_gap_px`
- `confidence`
- `run_id`, `model_name`, `method`

角色：
- 作為可重跑、可比較的實驗產物。

### 5.4 D 層：Evaluation Layer

用途：
- 讀取 `A + B + C` 後產生統計、報告與 debug 輸出。

來源：
- `run_evaluate.py`

角色：
- 不做人工作業
- 不生成原始 ground truth
- 僅做 join、對比、計算指標與輸出報告

## 6. 建議的資產目錄結構

```text
assets/
├── datasets/
│   └── <dataset_id>/
│       ├── manifest.json
│       └── instances.csv
├── ground_truth/
│   └── <dataset_id>/
│       ├── human_labels.csv
│       └── meta.json
└── predictions/
    └── <run_id>/
        ├── run_meta.json
        ├── localization.csv
        └── measurement.csv
```

說明：

1. `assets/datasets/`
   - 保存 Phase 1 的標準化資料資產。
2. `assets/ground_truth/`
   - 保存人工標註真值。
3. `assets/predictions/`
   - 保存每次模型或實驗執行的預測輸出。

不建議將 human label 放在：

- `output/`
  - 因為這裡偏向暫時性結果與報表輸出。
- `data/`
  - 因為這裡偏向原始來源資料或下載內容。

## 7. Dataset Asset 格式

### 7.1 `manifest.json`

用途：
- 描述此 dataset asset 的全域資訊。

建議欄位：

```json
{
  "dataset_id": "coco_val2017_cat_dog_v1",
  "source": "COCO val2017",
  "created_at": "2026-04-03T00:00:00Z",
  "categories": ["cat", "dog"],
  "filtering": {
    "min_instances": 2,
    "min_categories": 2,
    "min_area": 2000,
    "max_overlap_ratio": 0.8
  },
  "n_images": 8,
  "n_annotations": 18,
  "schema_version": 1
}
```

### 7.2 `instances.csv`

用途：
- 保存 dataset 中每個 object 的結構化資料。

建議欄位：

```text
dataset_id,image_id,image_path,annotation_id,category,bbox_x,bbox_y,bbox_w,bbox_h
```

備註：
- `mask` 與 `contours` 不適合直接塞進 CSV，可在 runtime 時由原始 COCO annotation 重建，或在未來另存 pickle/json asset。
- 初版先以「可重建、可 join」為優先。

## 8. Human Ground-Truth 格式

### 8.1 `human_labels.csv`

每個 object 一列。

建議欄位：

```text
dataset_id,
image_id,
annotation_id,
category,
bbox_x,bbox_y,bbox_w,bbox_h,
left_eye_x,left_eye_y,
right_eye_x,right_eye_y,
depth_rank,
label_status,
annotator,
labeled_at,
notes
```

欄位說明：

1. `dataset_id`
   - 用來指定這份 GT 屬於哪個 dataset asset。
2. `image_id`
   - COCO image id。
3. `annotation_id`
   - COCO annotation id，作為 object 主鍵。
4. `left_eye_x,left_eye_y`
   - 左眼影像座標。
5. `right_eye_x,right_eye_y`
   - 右眼影像座標。
6. `depth_rank`
   - 同一張圖內的深度排序。建議固定規則：`1 = 最靠前`，數字越大越遠。
7. `label_status`
   - 例如：`LABELED`, `SKIPPED`, `UNCERTAIN`
8. `annotator`
   - 標註者名稱。
9. `labeled_at`
   - ISO 8601 timestamp。
10. `notes`
   - 補充說明，例如「眼睛被遮擋」、「只看得到側臉」。

### 8.2 `meta.json`

用途：
- 保存此 GT asset 的全域資訊。

建議欄位：

```json
{
  "dataset_id": "coco_val2017_cat_dog_v1",
  "created_at": "2026-04-03T00:00:00Z",
  "updated_at": "2026-04-03T00:00:00Z",
  "annotators": ["alice"],
  "schema_version": 1,
  "depth_rank_rule": "1 = closest to camera"
}
```

## 9. Prediction Asset 格式

Prediction 與 Human GT 分開儲存。

### 9.1 `localization.csv`

建議欄位：

```text
run_id,
dataset_id,
image_id,
annotation_id,
method,
model_name,
status,
pred_left_eye_x,pred_left_eye_y,
pred_right_eye_x,pred_right_eye_y,
confidence
```

### 9.2 `measurement.csv`

建議欄位：

```text
run_id,
dataset_id,
image_id,
annotation_id,
pred_eye_distance_px
```

以及 pairwise 表：

```text
run_id,
dataset_id,
image_id,
annotation_a_id,
annotation_b_id,
pred_front_back_proxy_gap_px,
pred_front_back_proxy_ratio,
pred_relation
```

## 10. Runtime Join 模型

在 evaluator 執行時，應建立以下 join 關係：

1. `Dataset Asset (A)` join `Human GT (B)`
   - key: `dataset_id + image_id + annotation_id`
2. `Dataset Asset (A)` join `Prediction (C)`
   - key: `dataset_id + image_id + annotation_id`
3. 對 pairwise measurement evaluator，則使用：
   - `dataset_id + image_id + annotation_a_id + annotation_b_id`

最終 evaluator 的輸入應是：

```text
RuntimeRecord = DatasetRecord + Optional[HumanGT] + Optional[Prediction]
```

這表示：

- 沒有 GT 時，可以跑 baseline report
- 有 GT 時，才跑真實 accuracy / error / ranking metrics

## 11. `run_annotation.py` 設計

### 11.1 目標

建立一個輕量級、純文字互動的標註工具，用來為 dataset asset 補上 human GT。

### 11.2 嚴格限制

1. 不使用 tkinter / PyQt / Gradio / Web UI。
2. 互動僅使用：
   - `print()`
   - `input()`
3. 可選擇是否使用 `cv2.imshow()` 顯示圖片。

### 11.3 CLI 責任

`run_annotation.py` 只做以下事：

1. 載入 dataset asset
2. 載入既有 human label
3. 找出尚未標註或需重標的 object
4. 透過終端機逐一詢問並保存

它不應做：

1. 模型推論
2. evaluation 計算
3. 報表彙整

### 11.4 建議 CLI 參數

```text
--dataset-id <id>
--image-id <id>
--from-csv <path>
--annotator <name>
--skip-labeled
--overwrite
--no-imshow
--resume
```

說明：

1. `--dataset-id`
   - 指定要標哪一批 dataset asset。
2. `--image-id`
   - 只標單張圖片。
3. `--from-csv`
   - 指定一份 image id 清單或 subset 清單。
4. `--annotator`
   - 記錄標註者名稱。
5. `--skip-labeled`
   - 已存在 human label 的 object 自動跳過。
6. `--overwrite`
   - 允許重標。
7. `--no-imshow`
   - 不彈出圖片，只看終端機資訊。
8. `--resume`
   - 從先前中斷位置續標。

### 11.5 問答流程

每張圖每個 object 依序詢問：

1. 左眼座標
   - 使用者輸入：`x,y`
2. 右眼座標
   - 使用者輸入：`x,y`
3. 深度排序
   - 使用者輸入正整數

圖片顯示時，建議畫上：

1. `bbox`
2. `annotation_id`
3. `category`

### 11.6 防呆要求

必須使用 try-except 與輸入驗證，至少支援：

1. 非數字輸入不 crash
2. `x,y` 格式錯誤不 crash
3. 空輸入時重新提示
4. 支援 `skip`
5. 支援 `quit`
6. 支援 `redo`

### 11.7 存檔策略

建議採用：

1. 每完成一個 object 就立即寫入 CSV
2. 寫入前先載入既有 CSV
3. 以 `dataset_id + annotation_id` 為去重 key
4. 若 `--overwrite` 才允許覆寫既有列

這樣可以避免：

- 中途斷掉導致整張圖標註遺失
- 長流程標註時資料全部只存在記憶體

## 12. Evaluation 與 GT 的關係

未來 evaluator 應分成兩類：

### 12.1 無 GT 也可跑的 baseline evaluator

例如：

1. detection success rate
2. confidence summary
3. 可量測比例

### 12.2 需要 GT 才能跑的 evaluator

例如：

1. 眼睛座標誤差
   - `L2 distance`
   - `MAE`
2. 雙眼距離誤差
   - `abs(pred_eye_distance_px - gt_eye_distance_px)`
3. 深度排序正確率
   - pairwise ranking accuracy

因此 `run_evaluate.py` 未來應支援兩種模式：

1. 無 GT 模式
   - 只跑 baseline metrics
2. 有 GT 模式
   - 載入 `human_labels.csv`
   - 跑 GT-based metrics

## 13. 對現有專案的建議改動

### 13.1 `run_data_pipeline.py`

新增職責：

1. 在 Phase 1 完成後輸出 dataset asset
2. 產生 `dataset_id`
3. 輸出：
   - `manifest.json`
   - `instances.csv`

### 13.2 `src/data/`

建議新增：

1. `asset_exporter.py`
   - 負責把 Phase 1 dataset 匯出成可追蹤 asset
2. `gt_loader.py`
   - 載入 `human_labels.csv`
3. `gt_merge.py`
   - 將 human label merge 回 runtime dataset

### 13.3 `run_annotation.py`

新增，作為 human GT 的唯一建置入口。

### 13.4 `run_evaluate.py`

未來擴充：

1. 支援 `--dataset-id`
2. 支援 `--with-gt`
3. 支援載入 `human_labels.csv`
4. 對不同 validator 自動判斷是否有 GT 可用

## 14. MVP 落地順序

### Phase A：先定型 Asset 結構

1. 定義 `dataset_id`
2. 讓 Phase 1 輸出 `manifest.json + instances.csv`

### Phase B：先做 `run_annotation.py`

1. 支援單張圖
2. 支援 `skip-labeled`
3. 支援 `cv2.imshow()` 可選
4. 支援即時寫入 `human_labels.csv`

### Phase C：做 GT 載入與合併

1. 實作 `gt_loader`
2. 實作 `gt_merge`
3. 讓 runtime dataset 可帶入 GT

### Phase D：升級 evaluator

1. localization evaluator 比對預測眼睛與 GT 眼睛
2. measurement evaluator 比對預測距離與 GT 距離 / GT 排序

## 15. 風險與待決問題

1. `dataset_id` 的命名策略要固定
   - 建議包含類別、過濾版本、日期或 hash
2. 若未來修改 Phase 1 過濾條件，舊的 GT 是否還能沿用
   - 原則上不能假設一定沿用，需以 `dataset_id` 區分
3. `depth_rank` 是否允許同名次
   - 初版建議不允許，先要求唯一正整數
4. 是否需要標註 `visibility / occlusion`
   - 初版可先放進 `notes`，後續再正規化

## 16. 結論

本設計的核心結論是：

1. 人工標註不是 evaluation 流程的一部分，而是一層獨立的 `ground-truth asset layer`
2. 後續所有評估應建立在：
   - `Dataset Asset`
   - `Human Ground Truth`
   - `Prediction`
   - `Evaluation`
3. `run_annotation.py` 是建立 GT 的工具，不是模型推論工具，也不是 evaluator
4. 只要這層資產設計好，未來新增更多 evaluator 時，不需要重新做人工作業

