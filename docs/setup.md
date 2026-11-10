# FocalRoute安装与模型下载

所有命令在`focalroute`仓库根目录执行。源码通过Git克隆；LIBERO子模块仅在仿真时需要。模型与数据单独下载，不把它们加入Git。安装包名为`focalroute`，Python导入名为`openpi`。

## 1. 安装训练与推理环境

| 环境 | Python／核心依赖 | 入口 |
|---|---|---|
| NVIDIA训练／模型服务 | Python3.11、PyTorch2.7.1、Transformers4.53.2＋仓库替换实现 | `.venv/bin/python` |
| Ascend训练／模型服务 | 平台匹配的PyTorch2.7.1／torch_npu2.7.1／CANN；参考组合为CANN8.3 | `.venv-npu/bin/python` |
| Qwen8B→SAM预标注 | PyTorch2.8.0、Transformers4.57.1；独立CUDA环境 | `.venv-label/bin/python` |
| LIBERO仿真客户端 | 独立环境；通过WebSocket调用模型 | `examples/libero/.venv/bin/python` |
| CPU检查 | 已安装的训练环境，`OPENPI_DEVICE=cpu` | 仅导入／格式检查，不作为全量训练方案 |

### NVIDIA CUDA

先安装与PyTorch CUDA运行时兼容的NVIDIA驱动，确认`nvidia-smi`正常；准备Git和[uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --frozen --no-dev
SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
cp -R src/openpi/models_pytorch/transformers_replace/. "$SITE/transformers/"

OPENPI_DEVICE=cuda JAX_PLATFORMS=cpu .venv/bin/python -c \
  'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__); assert torch.cuda.is_available()'
```

替换实现包括SigLIP、PaliGemma和Gemma。重新安装Transformers后须再次复制；不要把Qwen使用的4.57.1装进这个环境。`JAX_PLATFORMS=cpu`防止PyTorch流程里的JAX辅助模块抢占GPU，不会把PyTorch训练改成CPU。

### Ascend NPU

先按[Ascend PyTorch](https://www.hiascend.com/software/ai-frameworks/pytorch)的版本矩阵配置驱动、CANN和匹配的PyTorch／torch_npu，并加载平台提供的CANN环境脚本。以下沿用平台已有的Python3.11及设备库；**不在NPU环境执行CUDA版`uv sync`**。

```bash
export NPU_BASE_PYTHON=/path/to/platform-env/bin/python
uv venv --python "$NPU_BASE_PYTHON" --system-site-packages .venv-npu

# 从项目依赖生成CPU-JAX版本，保留平台torch，不安装CUDA版JAX。
mkdir -p outputs/setup
.venv-npu/bin/python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import tomllib

config = tomllib.loads(Path("pyproject.toml").read_text())
assert version("torch").startswith("2.7.1")
assert version("torch_npu").startswith("2.7.1")
requirements = []
for item in config["project"]["dependencies"]:
    if item.startswith(("torch==", "jax[", "openpi-client")):
        continue
    if item == "lerobot":
        source = config["tool"]["uv"]["sources"]["lerobot"]
        item = f"lerobot @ git+{source['git']}@{source['rev']}"
    requirements.append(item)
requirements += ["jax==0.5.3", "ml-dtypes==0.4.1", "tensorstore==0.1.74"]
Path("outputs/setup/npu-requirements.txt").write_text("\n".join(requirements) + "\n")
Path("outputs/setup/npu-constraints.txt").write_text(
    f"torch=={version('torch')}\ntorch_npu=={version('torch_npu')}\n"
)
PY
uv pip install --python .venv-npu/bin/python \
  -c outputs/setup/npu-constraints.txt -r outputs/setup/npu-requirements.txt
uv pip install --python .venv-npu/bin/python -e packages/openpi-client
uv pip install --python .venv-npu/bin/python --no-deps -e .
SITE=$(.venv-npu/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
cp -R src/openpi/models_pytorch/transformers_replace/. "$SITE/transformers/"

OPENPI_DEVICE=npu JAX_PLATFORMS=cpu .venv-npu/bin/python -c \
  'import torch, torch_npu; from openpi.shared.torch_runtime import device_type; print(torch.__version__, torch_npu.__version__, device_type()); assert device_type() == "npu"'
```

生成的constraints固定平台torch／torch_npu版本；若依赖解析冲突，先对照平台版本矩阵，不通过升级设备库绕过。此安装配方需在目标NPU镜像验收，源码发行检查不会替你重装驱动或整套CANN。

训练时设置`OPENPI_DEVICE=npu`、`ASCEND_RT_VISIBLE_DEVICES`，并使用较短的临时目录：

```bash
export TMPDIR=/tmp/openpi-npu
mkdir -p "$TMPDIR"
```

CANN编译器可能在临时目录建立Unix socket；过深的路径会超过系统长度限制。

### 标注与仿真

- Qwen8B／SAM：[独立标注环境](prelabeling.md#1-准备独立标注环境)
- LIBERO：[独立仿真客户端](evaluation.md#2-安装libero仿真客户端)
- CPU检查：使用训练解释器并设置`OPENPI_DEVICE=cpu CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu`；不需要另装模型或启动服务

## 2. 下载官方模型与数据

以下以CUDA主环境为例；NPU环境把`.venv/bin/python`换成`.venv-npu/bin/python`。需要可访问GCS和Hugging Face，已有资产可直接复用。

### π0.5 LIBERO与分词器

```bash
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
JAX_CKPT=$(JAX_PLATFORMS=cpu .venv/bin/python -c \
  'from openpi.shared.download import maybe_download; print(maybe_download("gs://openpi-assets/checkpoints/pi05_libero", gs={"token": "anon"}))')
JAX_PLATFORMS=cpu .venv/bin/python -c \
  'from openpi.shared.download import maybe_download; print(maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))'

# 转换产物保留FP32源权重；训练／推理按原π0.5规则选择BF16骨干和FP32模块。
OPENPI_DEVICE=cpu JAX_PLATFORMS=cpu .venv/bin/python \
  examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir "$JAX_CKPT" --config-name pi05_libero \
  --output-path models/pi05_libero_pytorch --precision float32

mkdir -p models/pi05_libero_pytorch/assets assets/pi05_libero
cp -R "$JAX_CKPT/assets/." models/pi05_libero_pytorch/assets/
cp -R "$JAX_CKPT/assets/." assets/pi05_libero/
```

转换会读取完整JAX权重，需要足够的主机内存和磁盘。转换产物没有Router；新训练先构造目标Router，再加载官方参数。不要通过把原权重整体转BF16来替代运行时的混合精度边界。

归一化统计应位于：

```text
models/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json
assets/pi05_libero/physical-intelligence/libero/norm_stats.json
```

直接复现LIBERO时使用上述官方统计，不要从伪标签重算；其他数据集的统计流程见[norm_stats.md](norm_stats.md)。

### LIBERO训练数据

训练与标注共用同一个固定LeRobot v2快照，区别于仿真环境的任务／初态文件：

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "physical-intelligence/libero", repo_type="dataset",
    revision="a4336d589d589045d1c56423ffdf3b88a0e19b1f",
    local_dir="data/physical-intelligence/libero",
)
PY
export HF_LEROBOT_HOME="$(realpath data)"
```

使用`realpath`固定数据目录身份；同一数据通过不同软链接路径传入，可能触发Hugging Face重建Arrow缓存。只做推理／仿真不需要下载训练演示数据。

### Qwen8B和SAM

模型revision、SAM下载地址及完整命令在[预标注指南](prelabeling.md#2-下载qwen8b与sam)。不需要32B模型、8B二次一致性复核或在线teacher。

## 3. 路径约定

```text
models/                 原模型与标注模型
data/                   固定训练数据快照
assets/pi05_libero/      官方归一化统计
outputs/prelabel8/       新标签manifest、原回答和mask
checkpoints/pi05_libero/<实验名>/<步数>/
outputs/                日志、评测、视频和本地调试
```

以上目录和所有`.venv*`环境均被Git忽略。`uv.lock`、依赖声明、源码和使用指南保留在发布仓库中；`.env.example`可提交，真实`.env`和密钥不可提交。
