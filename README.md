# FocalRoute

FocalRoute是基于 [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac) 的 PyTorch π0.5 视觉 Token Router。用少量 Hard＋Global token 压缩视觉前缀，保留完整 LLM 和 Action Expert 推理路径。

这是独立研究分支，不是 Physical Intelligence 官方发布。

## 最新方案

- **Hard保留原RoPE位置**：被选patch不重新连续编号，文本和动作也保留原位置。
- **Global放在尾部位置**：RoPE编号追加到原动作范围之后，实际仍进入前缀KV缓存；前缀顺序为`[Hard, text, Global]`。
- **完整后端训练**：Router、整个LLM和整个Action Expert参与训练，仅冻结SigLIP及视觉投影器。
- **稀疏伪标签**：Qwen3-VL-8B单轮四角色定位→SAM ViT-H；每8帧取1帧，Hard辅助损失前15k开启、后15k关闭，flow全程保留。
- **默认预算H96＋G32＝R128**：Hard按LIBERO加权标注分数至少保留90%选取，Global取512个有效视觉token的1/16；均向上对齐32。详见[预算口径](docs/router.md#默认预算的依据)。

新模式必须显式启用；旧checkpoint继续使用保存时的拓扑和位置模式。保留位置的效果须由对应新模型闭环评测验证，不能沿用旧Router成绩。

## 从这里开始

| 要做什么 | 指南 |
|---|---|
| 安装CUDA、Ascend或CPU环境 | [安装与模型下载](docs/setup.md#1-安装训练与推理环境) |
| 下载π0.5、转换PyTorch权重、准备LIBERO | [模型与数据](docs/setup.md#2-下载官方模型与数据) |
| 下载Qwen8B、SAM并生成标签 | [预标注](docs/prelabeling.md) |
| 训练原RoPE R128／R64、Hard-only或恢复断点 | [Router训练](docs/router.md) |
| 启动模型服务、LIBERO仿真与标准评测 | [推理与评测](docs/evaluation.md) |

## 安装主环境

需要Linux、Python 3.11、Git和[uv](https://docs.astral.sh/uv/getting-started/installation/)。仓库名及安装包名为`focalroute`：

```bash
git clone https://github.com/leoboynju/focalroute.git
cd focalroute
git submodule update --init third_party/libero
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --frozen --no-dev
SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
cp -R src/openpi/models_pytorch/transformers_replace/. "$SITE/transformers/"
```

这是**CUDA训练／推理环境**。Python导入名仍为`openpi`，客户端为`openpi-client`。Ascend使用平台匹配的PyTorch／torch_npu／CANN环境；Qwen预标注和仿真客户端使用独立环境，安装命令见对应指南。不要混装不同版本的Transformers。

## 模型与运行入口

| 模型／组件 | 获取方式 | 用途 |
|---|---|---|
| 官方π0.5 LIBERO | `gs://openpi-assets/checkpoints/pi05_libero`；[下载与转换](docs/setup.md#2-下载官方模型与数据) | Router-free初始化、原版推理 |
| 原RoPE Router | 从官方权重按[训练命令](docs/router.md#2-启动新训练)独立训练 | R128、R64及Hard-only |
| Qwen3-VL-8B-Instruct | [预标注指南](docs/prelabeling.md#2-下载qwen8b与sam) | 离线定位source／goal／arm／gripper |
| SAM ViT-H | 同上 | 将定位框转成像素mask和稀疏patch标签 |

本源码仓库不包含模型权重、数据集或预先训练好的原RoPE Router权重。下载、训练和评测产物默认放入已忽略的`models/`、`data/`、`assets/`、`checkpoints/`、`outputs/`。

准备好checkpoint后，启动策略服务：

```bash
CUDA_VISIBLE_DEVICES=0 OPENPI_DEVICE=cuda JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/serve_policy.py --env LIBERO --port 8000 \
  policy:checkpoint --policy.config pi05_libero \
  --policy.dir checkpoints/pi05_libero/router_originalrope_r128_h96g32/30000
```

原版π0.5可把`--policy.dir`换成`models/pi05_libero_pytorch`；模型服务会读取checkpoint中的Router拓扑、mask及RoPE配置。客户端运行方式见[推理与评测](docs/evaluation.md)。

## 代码导航

- `src/openpi/models_pytorch/`：Router、π0.5及位置／精度实现
- `scripts/train_pytorch.py`：训练、保存与恢复
- `scripts/prelabel_lerobot.py`：公开预标注入口
- `scripts/serve_policy.py`、`packages/openpi-client/`：模型服务与客户端
- `examples/libero/`：标准LIBERO客户端

上游其他示例保留在`examples/`，其说明放在各自目录；不属于本分支原RoPE方案的复现步骤。本机实验控制器、TensorRT调试和归档脚本保留本地，不随源码发布。

## 许可

源码见[LICENSE](LICENSE)及[NOTICE](NOTICE)。基础模型、Qwen、SAM和数据集分别遵循其发布方许可。
