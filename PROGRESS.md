# 项目进度同步

> 本文档记录当前已完成的环境、代码、数据、训练与结果，方便后续继续推进。

## 1. 环境与硬件

| 项目 | 状态 |
| --- | --- |
| Conda 环境 | `yolo_gaiic` |
| Python | 3.10 |
| PyTorch | `torch 2.11.0+cu128` |
| torchvision | `torchvision 0.26.0+cu128` |
| CUDA 版本 | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| 计算能力 | (12, 0) |
| Ultralytics | 8.4.67 |

验证命令：

```bash
conda run -n yolo_gaiic python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
conda run -n yolo_gaiic python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
```

## 2. 数据集与配置

### 2.1 已生成数据集

| 数据集 | 路径 | 说明 |
| --- | --- | --- |
| RGB-only | `datasets/gaiic_yolo_rgb/` | 图像为符号链接，标注已转 YOLO 格式 |
| TIR-only | `datasets/gaiic_yolo_tir/` | 图像为符号链接，标注已转 YOLO 格式 |
| RGB-T 融合 | `datasets/gaiic_yolo_rgbt/` | 融合后 3 通道图像（TIR + G + R），完整规模 |
| RGB-T 小规模 | `datasets/rgbt_smoke/` | 小规模测试集，用于快速冒烟测试 |

### 2.2 已生成配置

| 配置 | 路径 | 说明 |
| --- | --- | --- |
| RGB-only | `configs/gaiic_yolo_rgb.yaml` | 单模态 RGB 训练配置 |
| TIR-only | `configs/gaiic_yolo_tir.yaml` | 单模态 TIR 训练配置 |
| RGB-T 融合 | `configs/gaiic_yolo_rgbt.yaml` | 融合模态训练配置 |
| RGB-T smoke | `configs/rgbt_smoke.yaml` | 小规模融合训练配置 |

## 3. 脚本清单

| 脚本 | 用途 |
| --- | --- |
| `scripts/common.py` | 公共常量与工具函数 |
| `scripts/prepare_yolo_dataset.py` | RGB/TIR 单模态数据集准备 |
| `scripts/prepare_rgbt_yolo_dataset.py` | RGB-T 融合数据集准备，支持 `rgbt` / `tir_rgb_gray` / `weighted` |
| `scripts/check_dataset.py` | 数据集完整性检查 |
| `scripts/visualize_coco.py` | COCO 标注可视化 |
| `scripts/train_rgbt_yolo.py` | RGB-T 融合模型训练封装 |
| `scripts/predict_rgbt_to_coco.py` | RGB-T 模型推理并导出 COCO results |
| `scripts/yolo_json_to_coco.py` | Ultralytics JSON 转 COCO detection results |
| `scripts/fuse_coco_results.py` | RGB + TIR 后融合 |
| `scripts/eval_coco_results.py` | COCO 评估 |
| `scripts/make_submission.py` | 提交文件格式校验与输出 |

## 4. 已训练模型

| 模型 | 路径 | 说明 |
| --- | --- | --- |
| RGB-only YOLO11s | `work_dirs/yolo11s_rgb/weights/best.pt` | RGB 单模态基线 |
| RGB-T YOLO11s | `work_dirs/yolo11s_rgbt/weights/best.pt` | RGB-T 融合模型 |

## 5. 已生成结果

| 结果 | 路径 | 说明 |
| --- | --- | --- |
| RGB-T val 预测 | `work_dirs/yolo11s_rgbt_val_pred_conf025/` | 验证集预测可视化 |
| RGB-T val COCO | `submissions/rgbt_val_conf025.json` | 验证集 COCO detection results |

## 6. 关键命令备忘

### 6.1 数据集准备

```bash
# RGB / TIR 单模态
conda run -n yolo_gaiic python scripts/prepare_yolo_dataset.py --modality rgb
conda run -n yolo_gaiic python scripts/prepare_yolo_dataset.py --modality tir

# RGB-T 融合（完整）
conda run -n yolo_gaiic python scripts/prepare_rgbt_yolo_dataset.py --fusion rgbt

# RGB-T 融合（小规模 smoke test）
conda run -n yolo_gaiic python scripts/prepare_rgbt_yolo_dataset.py --fusion rgbt --out-dir datasets/rgbt_smoke --yaml-out configs/rgbt_smoke.yaml --max-images 100
```

### 6.2 训练

```bash
# RGB-only
conda run -n yolo_gaiic yolo detect train model=yolo11s.pt data=configs/gaiic_yolo_rgb.yaml imgsz=640 epochs=100 batch=8 device=0 project=work_dirs name=yolo11s_rgb

# TIR-only
conda run -n yolo_gaiic yolo detect train model=yolo11s.pt data=configs/gaiic_yolo_tir.yaml imgsz=640 epochs=100 batch=8 device=0 project=work_dirs name=yolo11s_tir

# RGB-T 融合
conda run -n yolo_gaiic python scripts/train_rgbt_yolo.py --data configs/gaiic_yolo_rgbt.yaml --name yolo11s_rgbt
```

### 6.3 验证 / 推理

```bash
# RGB-T 验证集推理
conda run -n yolo_gaiic python scripts/predict_rgbt_to_coco.py --model work_dirs/yolo11s_rgbt/weights/best.pt --source datasets/gaiic_yolo_rgbt/images/val --out submissions/rgbt_val.json --split val --conf 0.001

# RGB-T 测试集推理
conda run -n yolo_gaiic python scripts/predict_rgbt_to_coco.py --model work_dirs/yolo11s_rgbt/weights/best.pt --source datasets/gaiic_yolo_rgbt/images/test --out submissions/rgbt_test.json --split test --conf 0.001

# 评估
conda run -n yolo_gaiic python scripts/eval_coco_results.py --pred submissions/rgbt_val.json
```

### 6.4 后融合提交

```bash
# 分别生成 rgb 与 tir 测试集结果后
conda run -n yolo_gaiic python scripts/fuse_coco_results.py --rgb submissions/rgb_test.json --tir submissions/tir_test.json --out submissions/fused_test.json
conda run -n yolo_gaiic python scripts/make_submission.py --input submissions/fused_test.json --out submissions/final_submission.json
```

## 7. 下一步建议

1. **跑通 TIR-only 基线**：目前 RGB-only 与 RGB-T 已有模型，TIR-only 尚缺。
2. **评估当前模型**：对 `work_dirs/yolo11s_rgb/weights/best.pt` 和 `work_dirs/yolo11s_rgbt/weights/best.pt` 在 val 上跑评估，得到 mAP 对比。
3. **RGB + TIR 后融合**：生成 rgb_test.json 与 tir_test.json 后，用 `fuse_coco_results.py` 得到最终 fused 提交。
4. **调优 RGB-T 融合方式**：当前使用 `rgbt`（TIR + G + R），可对比 `tir_rgb_gray` 与 `weighted`。
5. **模型升级**：在稳定基线后尝试 `yolo11m.pt`。

## 8. 注意事项

- 所有命令统一使用 `conda run -n yolo_gaiic`。
- `datasets/gaiic_yolo_rgb/` 与 `datasets/gaiic_yolo_tir/` 中图像为符号链接，删除时不会删除原始数据。
- RGB-T 融合数据集为实际复制生成的 3 通道 JPG，占用空间较大，注意磁盘空间。
