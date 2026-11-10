# 模型服务与LIBERO评测

模型服务和仿真客户端分开运行。先完成[安装与权重下载](setup.md)，无须为评测下载LIBERO训练演示数据。

## 1. 启动模型服务

### 原RoPE Router

```bash
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
CUDA_VISIBLE_DEVICES=0 OPENPI_DEVICE=cuda JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/serve_policy.py --env LIBERO --port 8000 \
  policy:checkpoint --policy.config pi05_libero \
  --policy.dir checkpoints/pi05_libero/router_originalrope_r128_h96g32/30000
```

### 官方π0.5或其他Router变体

同一条命令只改`--policy.dir`：

| 模型 | checkpoint目录 |
|---|---|
| 官方π0.5 LIBERO（已转换） | `models/pi05_libero_pytorch` |
| mixed R64／R128、Hard-only | 对应实验的数字步数目录，包含`model_config.json`和`assets/` |
| 旧连续RoPE模型 | 原目录；保持旧模式，不手改配置文件 |

NPU服务使用`.venv-npu/bin/python`，将设备变量改成`OPENPI_DEVICE=npu ASCEND_RT_VISIBLE_DEVICES=0`。训练和服务不要争用同一设备。

`action_horizon=10`是动作块长度；默认Expert去噪为10次，客户端`replan_steps=5`表示每执行5步重新规划。这三个参数不同，修改重规划间隔不会改变去噪次数。服务CLI目前不提供去噪次数开关；若做5次消融，应显式通过策略API的`sample_kwargs={"num_steps": 5}`配置并单独记录结果。

## 2. 安装LIBERO仿真客户端

在第二个终端执行，使用子模块附带的任务和初态。此环境沿用上游客户端锁定依赖，与Python3.11的模型环境隔离：

```bash
git submodule update --init third_party/libero
uv venv --python 3.8 examples/libero/.venv
uv pip sync --python examples/libero/.venv/bin/python \
  examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client
uv pip install --python examples/libero/.venv/bin/python --no-deps -e third_party/libero
```

首次使用LIBERO会创建本机路径配置，默认可用`~/.libero/config.yaml`；换机器或移动目录时须更新它。无显示器渲染需要可用的EGL驱动，`MUJOCO_GL=egl`；显示服务器环境可按平台配置改为`glx`。

## 3. 四套件标准2000条

保持第一个终端的模型服务运行：

```bash
export PYTHONPATH="$PWD/third_party/libero:$PWD/packages/openpi-client/src"
export MUJOCO_GL=egl
mkdir -p outputs/libero
set -o pipefail
for suite in libero_spatial libero_object libero_goal libero_10; do
  examples/libero/.venv/bin/python examples/libero/main.py \
    --args.host localhost --args.port 8000 \
    --args.task-suite-name "$suite" --args.num-trials-per-task 50 \
    --args.replan-steps 5 --args.seed 7 \
    --args.video-out-path "outputs/libero/$suite/videos" \
    2>&1 | tee "outputs/libero/$suite.log"
done
```

每套件10任务×50个官方初态＝500条，总计2000条；`libero_10`也称Long。记录四套件成功率和总体平均，同时保留任务记录、模型身份和实际运行参数。先跑单套件`--args.num-trials-per-task 1`可检查连通，但不计入正式2000分母。

该公开客户端是上游串行路径，结果在日志和视频目录。它不等同本机带分片恢复、完整轨迹收据和CSV交付的实验控制器；视频命名按任务和成败复用，不提供2000个独立视频的留存保证。

## 4. 其他环境与工具边界

- 自己的机器人／仿真：安装`packages/openpi-client`，按[远程推理接口](remote_inference.md)提供图像、状态和任务文本；动作维度及归一化必须与checkpoint对应
- ALOHA／DROID：上游示例仍保留在`examples/`各目录，本分支原RoPE路线在LIBERO验证，不能据此声称这些环境已验收
- LIBERO-Plus：官方扩展需独立资源及prompt映射；本仓库当前不发布本机PLUS控制器。不能把标准LIBERO命令改个套件名就当作PLUS评测
- TensorRT：本仓库公开PyTorch推理；本机引擎和调试脚本不入Git，旧引擎不能直接运行新原RoPE模型

开环数据回放只用于可视化；成功率必须来自模型动作实际驱动`env.step`。任何少量smoke或未完成结果均不能标为最终成绩。
