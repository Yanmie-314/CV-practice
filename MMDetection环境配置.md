# MMDetection 环境配置（RTX 5070 Ti）

## 1. 问题说明

RTX 5070 Ti 属于新架构 GPU。旧版 PyTorch，例如 `torch 2.1 + cu118`，可能不支持该显卡的计算架构，容易出现 CUDA 不可用、算子不兼容或运行时报错。

因此本项目环境配置优先原则是：

1. 先保证 PyTorch 能正确识别并使用 RTX 5070 Ti。
2. 再解决 MMDetection / MMCV 的版本和编译问题。

不要再使用 `torch 2.1.2 + cu118` 作为推荐方案。

## 2. 推荐环境方案

建议新建干净 conda 环境：

```bash
conda create -n mmdet_gaiic python=3.10 -y
conda activate mmdet_gaiic
```

安装支持新架构 GPU 的 CUDA 版 PyTorch。优先使用 CUDA 12.8 轮子。

目标检测不需要 `torchaudio`，不要把它放进基础安装命令，否则 pip 可能解析到旧版 `torchaudio` 并造成依赖冲突。

```bash
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

验证 PyTorch：

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'none')"
```

期望结果：

```text
available True
gpu NVIDIA GeForce RTX 5070 Ti
```

## 3. 安装 MMDetection 依赖

安装基础依赖：

```bash
pip install -U openmim
pip install opencv-python pycocotools matplotlib tqdm pillow numpy scipy pandas ensemble-boxes
pip install mmengine
```

## 4. MMCV 安装策略

MMDetection 依赖 MMCV。对于 RTX 5070 Ti + CUDA 12.8 + 新版 PyTorch，最大风险是 MMCV 没有完全匹配的 Windows 预编译 wheel。

先尝试 OpenMIM 安装：

```bash
mim install mmcv
```

如果成功，再安装 MMDetection：

```bash
mim install mmdet
```

验证：

```bash
python -c "import mmcv, mmengine, mmdet; print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__); print('mmdet', mmdet.__version__)"
```

## 5. 如果 MMCV 安装失败

如果 `mim install mmcv` 找不到匹配 wheel，通常会尝试源码编译。Windows 下源码编译 MMCV 需要：

- Visual Studio 2022 Build Tools
- MSVC C++ 编译工具
- Windows SDK
- CUDA Toolkit 12.x
- 与 PyTorch CUDA 版本兼容的本机 CUDA 编译环境

源码编译可尝试：

```bash
pip install -U ninja packaging wheel setuptools
set MMCV_WITH_OPS=1
pip install -v -U --no-build-isolation mmcv
```

如果需要显式指定显卡架构，可设置：

```bash
set TORCH_CUDA_ARCH_LIST=12.0
```

然后再执行：

```bash
pip install -v -U --no-build-isolation mmcv
```

注意：`TORCH_CUDA_ARCH_LIST=12.0` 是否被当前 PyTorch/CUDA 编译链接受，取决于已安装 PyTorch 和 CUDA Toolkit 是否支持该架构。

## 6. 更稳的替代方案

如果 Windows 下 MMCV 编译失败，推荐使用 WSL2 / Linux 环境：

```bash
conda create -n mmdet_gaiic python=3.10 -y
conda activate mmdet_gaiic
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -U openmim
mim install mmcv
mim install mmdet
pip install opencv-python pycocotools matplotlib tqdm pillow numpy scipy pandas ensemble-boxes
```

Linux/WSL2 下编译和安装 MMCV 通常比 Windows 稳定。

## 7. 可接受的最终环境状态

环境配置完成后，需要满足：

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python -c "import mmcv, mmengine, mmdet; print(mmcv.__version__, mmengine.__version__, mmdet.__version__)"
```

如果这两条命令都能成功，环境即可用于本项目。

## 8. 后续运行约定

你配好环境后，把实际 conda 环境名告诉我。之后我会统一使用：

```bash
conda run -n 环境名 python ...
conda run -n 环境名 mim ...
```

不会使用 base 环境或旧的 `pytorch_gpu` 环境运行项目代码，除非你明确要求。
