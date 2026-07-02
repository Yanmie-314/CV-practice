# GEM-YOLO 改进实现说明

本次实现对应 `改进方案与实现指南.md` 中的 Phase 0 和 Phase 1。

## Phase 0：TTA + WBF 集成

现有 `scripts/predict_rgbt_to_coco.py` 已支持：

- `--augment`：启用 Ultralytics TTA
- `--agnostic-nms`：启用 class-agnostic NMS

三模型默认集成入口：

```bash
conda run -n yolo_gaiic python scripts/run_phase0_ensemble.py \
  --split val \
  --augment \
  --device 0 \
  --batch-size 4 \
  --fuse-iou-thr 0.55 \
  --score-thr 0.001
```

## Pretrained GEM Adapter notes

The adapter path is the preferred "GEM on pretrained YOLO" route. It keeps the
existing YOLO detector loaded from `work_dirs/yolo11s_rgbt_aug_domain-2/weights/best.pt`
and learns only a lightweight RGB/TIR gated residual image adapter by default.

Channel convention is important: fused JPG files are written by OpenCV as BGR
`[TIR, G, R]`, then Ultralytics decodes them to RGB tensors. Therefore the
pretrained detector actually receives tensor channels `[R, G, TIR]`. The adapter
baseline uses exactly `[R, G, TIR]`, and its final residual conv is zero
initialized so an untrained adapter is equivalent to the pretrained fused-image
input.

Run this sanity check before adapter training if the dataset or fusion pipeline
changes:

```bash
conda run -n yolo_gaiic python scripts/check_gem_adapter_baseline.py \
  --split val \
  --max-images 16
```

Prediction checkpoints save `checkpoint_type`, `model_meta`, adapter weights,
YOLO weights, and optimizer state. If the saved YOLO path moves, override it:

```bash
conda run --no-capture-output -n yolo_gaiic python -u scripts/predict_gem_adapter_yolo_to_coco.py \
  --checkpoint work_dirs/gem_adapter_yolo/weights/best.pt \
  --yolo-weights work_dirs/yolo11s_rgbt_aug_domain-2/weights/best.pt \
  --out submissions/gem_adapter_yolo_val.json \
  --split val \
  --device 0
```

默认会使用：

- `work_dirs/yolo11s_rgb/weights/best.pt`
- `work_dirs/yolo11s_tir/weights/best.pt`
- `work_dirs/yolo11s_rgbt_aug_domain/weights/best.pt`

输出位于 `submissions/phase0/`。验证集会自动调用 `scripts/eval_coco_results.py`。

## Phase 1：GEM-YOLO 双流训练

新增文件：

- `scripts/models/gem_yolo.py`：GFM、SPDConv、GhostBottleneck、GSConv、GhostPAN、`GemYoloDetector`
- `scripts/datasets/rgbt_dual_dataset.py`：原始 RGB/TIR 双流数据集
- `configs/gem_yolo_rgbt.yaml`：默认训练配置
- `scripts/train_gem_yolo.py`：纯 PyTorch 训练入口，复用 Ultralytics v8 detection loss
- `scripts/predict_gem_yolo_to_coco.py`：GEM checkpoint 的 COCO JSON 推理导出

冒烟训练：

```bash
conda run -n yolo_gaiic python scripts/train_gem_yolo.py \
  --epochs 1 \
  --batch 2 \
  --workers 0 \
  --imgsz 256 \
  --device cpu \
  --max-images 2 \
  --name gem_yolo_rgbt_smoke
```

完整训练建议：

```bash
conda run --no-capture-output -n yolo_gaiic python -u scripts/train_gem_yolo.py \
  --config configs/gem_yolo_rgbt.yaml \
  --device 0
```

训练时会同时写入 `work_dirs/gem_yolo_rgbt/train.log`。实时查看：

```powershell
Get-Content work_dirs\gem_yolo_rgbt\train.log -Tail 80 -Wait
```

如果希望每个 step 都输出进度，可以增加：

```bash
--progress-interval 1
```

GEM 推理导出：

```bash
conda run -n yolo_gaiic python scripts/predict_gem_yolo_to_coco.py \
  --checkpoint work_dirs/gem_yolo_rgbt/weights/best.pt \
  --out submissions/gem_yolo_val.json \
  --split val \
  --device 0
```

然后评估：

```bash
conda run -n yolo_gaiic python scripts/eval_coco_results.py \
  --pred submissions/gem_yolo_val.json
```

## 推荐新路线：预训练 YOLO + GEM Adapter

`scripts/train_gem_adapter_yolo.py` 在现有 YOLO 权重前增加一个轻量 GEM Adapter，默认加载：

```text
work_dirs/yolo11s_rgbt_aug_domain-2/weights/best.pt
```

Adapter 初始输出接近已有 `[TIR, G, R]` 输入分布，并用残差学习动态融合，避免从零训练检测器。

先冻结 YOLO，只训练 Adapter：

```bash
conda run --no-capture-output -n yolo_gaiic python -u scripts/train_gem_adapter_yolo.py \
  --yolo-weights work_dirs/yolo11s_rgbt_aug_domain-2/weights/best.pt \
  --epochs 30 \
  --batch 8 \
  --device 0 \
  --name gem_adapter_yolo \
  --progress-interval 20
```

如果 Adapter-only 有提升，再从第 11 轮开始小学习率解冻 YOLO：

```bash
conda run --no-capture-output -n yolo_gaiic python -u scripts/train_gem_adapter_yolo.py \
  --yolo-weights work_dirs/yolo11s_rgbt_aug_domain-2/weights/best.pt \
  --epochs 40 \
  --batch 8 \
  --device 0 \
  --name gem_adapter_yolo_unfreeze \
  --unfreeze-epoch 11 \
  --yolo-lr-scale 0.02
```

推理导出：

```bash
conda run --no-capture-output -n yolo_gaiic python -u scripts/predict_gem_adapter_yolo_to_coco.py \
  --checkpoint work_dirs/gem_adapter_yolo/weights/best.pt \
  --out submissions/gem_adapter_yolo_val.json \
  --split val \
  --device 0 \
  --batch-size 4 \
  --conf 0.001 \
  --iou 0.7
```
