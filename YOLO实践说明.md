# Ultralytics YOLO RGB-T 实践说明

## 1. 核心路线

本项目主线改为 RGB-T 单模型：每个样本同时读取同名 RGB 与 TIR 图像，先生成一张输入级融合图，再用一个 YOLO 模型训练和推理。

默认融合方式 `rgbt` 会生成 3 通道图像：`[TIR, G, R]`。这样模型一次前向即可同时利用红外强度、可见光绿色通道和可见光红色通道；bbox 坐标仍保持原始 `640 x 512`，不需要缩放。

该路线不依赖 `mmcv`、`mmengine`、`mmdet`。

## 2. 环境检查

```powershell
cd D:\cv
conda run --no-capture-output -n yolo_gaiic python -c "import torch, ultralytics; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('ultralytics', ultralytics.__version__)"
```

如果使用 `conda run` 启动长时间训练或推理，必须加 `--no-capture-output`，否则 conda 可能缓存 stdout/stderr，终端看不到实时 epoch 日志。

## 3. 准备 RGB-T 数据集

先做 3 张图的冒烟测试：

```powershell
conda run --no-capture-output -n yolo_gaiic python scripts/prepare_rgbt_yolo_dataset.py --fusion rgbt --out-dir datasets/rgbt_smoke --yaml-out configs/rgbt_smoke.yaml --max-images 3 --overwrite
```

正式生成完整 RGB-T 数据集：

```powershell
conda run --no-capture-output -n yolo_gaiic python scripts/prepare_rgbt_yolo_dataset.py --fusion rgbt --out-dir datasets/gaiic_yolo_rgbt --yaml-out configs/gaiic_yolo_rgbt.yaml --overwrite
```

输出：

```text
datasets/gaiic_yolo_rgbt/
configs/gaiic_yolo_rgbt.yaml
```

## 4. 训练单个 RGB-T 模型

```powershell
conda run --no-capture-output -n yolo_gaiic python scripts/train_rgbt_yolo.py --model yolo11s.pt --data configs/gaiic_yolo_rgbt.yaml --imgsz 640 --epochs 100 --batch 8 --device 0 --project D:/cv/work_dirs --name yolo11s_rgbt
```

正常时终端会持续输出 `epoch`、`GPU_mem`、`box_loss`、`cls_loss`、`dfl_loss` 等日志。

## 5. 验证

```powershell
conda run --no-capture-output -n yolo_gaiic yolo detect val model=D:/cv/work_dirs/yolo11s_rgbt/weights/best.pt data=configs/gaiic_yolo_rgbt.yaml imgsz=640 batch=8 device=0 project=D:/cv/work_dirs name=yolo11s_rgbt_val
```

生成验证集 COCO detection results 并评估：

```powershell
conda run --no-capture-output -n yolo_gaiic yolo detect predict model=D:/cv/work_dirs/yolo11s_rgbt/weights/best.pt source=datasets/gaiic_yolo_rgbt/images/val imgsz=640 conf=0.001 iou=0.7 max_det=300 save_json=True project=D:/cv/work_dirs name=yolo11s_rgbt_val_pred
conda run --no-capture-output -n yolo_gaiic python scripts/yolo_json_to_coco.py --split val --pred-json work_dirs/yolo11s_rgbt_val_pred/predictions.json --out submissions/rgbt_val.json
conda run --no-capture-output -n yolo_gaiic python scripts/eval_coco_results.py --pred submissions/rgbt_val.json
```

## 6. 测试集提交

```powershell
conda run --no-capture-output -n yolo_gaiic yolo detect predict model=D:/cv/work_dirs/yolo11s_rgbt/weights/best.pt source=datasets/gaiic_yolo_rgbt/images/test imgsz=640 conf=0.001 iou=0.7 max_det=300 save_json=True project=D:/cv/work_dirs name=yolo11s_rgbt_test_pred
conda run --no-capture-output -n yolo_gaiic python scripts/yolo_json_to_coco.py --split test --pred-json work_dirs/yolo11s_rgbt_test_pred/predictions.json --out submissions/rgbt_test.json
conda run --no-capture-output -n yolo_gaiic python scripts/make_submission.py --input submissions/rgbt_test.json --out submissions/final_submission.json
```

最终提交 `submissions/final_submission.json`。

## 7. 对照实验

RGB-only 和 TIR-only 仍可作为对照实验保留，主提交使用 RGB-T 单模型结果。
