# 原RoPE Router：训练与恢复

先完成[环境、官方权重和数据准备](setup.md)，需要辅助监督时完成[Qwen8B→SAM预标注](prelabeling.md)。这里的默认复现路线是**混合R128（H96＋G32）＋原位置RoPE＋完整LLM／Expert训练**。

## 1. 模型合同

| 档位 | Hard | Global | 两有效视图合计预算 |
|---|---:|---:|---:|
| mixed R128（新训练默认） | 96 | 32 | 128 |
| mixed R128（既有实验） | 64 | 64 | 128 |
| mixed R64 | 32 | 32 | 64 |
| Hard-only128 | 128 | 0，分支关闭 | 128 |
| Hard-only64 | 64 | 0，分支关闭 | 64 |

SigLIP与视觉projector仍编码输入图像，Router位于它们与LLM之间。LIBERO输入共3×256个patch；第三相机是占位图，开启mask后只有前两路512个位置参与Hard选择与Global聚合。

### 默认预算的依据

- Hard：在当前LIBERO Qwen8B→SAM标签上，选择平均加权标注分数保留率**至少90%**的最小预算，再向上对齐32。H64保留87.90%，H80保留91.30%但不满足32对齐，故选H96，保留93.09%。这是数据集平均值，不是每帧保证。
- Global：`G = 32 × ceil(N_valid / (16 × 32))`；LIBERO的`N_valid=512`，所以G32。第三张占位图不计入；若是768个有效视觉token，向上对齐后应为G64。
- H96＋G32仍是R128，两项都是32的倍数；既有H64＋G64只满足总预算128，不满足这次分配规则。默认值在新训练初始化时固定，不在线读伪标签或逐帧改变token数量；其他数据集须用自身标签和有效视图重新校准。
- 统计来源：34,918个stride8时刻中的34,221个非空标注，使用K64截断前的支持patch和分数，source／goal／gripper权重1／0.6／0.8，排除纯arm和非正标签补位。保留率是理想标签排序下每帧分数占比的平均，不是实际Router召回或闭环成功率。
- 新默认需独立训练；已有checkpoint仍恢复保存的H/G，不能把H64/G64权重改标签后当成H96/G32。两者总预算相同但Global输出头形状不同。

### 手动指定H／G

省略数量参数时使用H96/G32；也可分别覆盖，两项都省略或只改一项均可。手动值原样生效，不被90%或1/16规则再次换算，R始终为H＋G。

```bash
# 添加到下方训练命令，或替换其中的同名参数；例如既有mixed R128配比
--model.vision-router-hard-tokens 64 --model.vision-router-soft-tokens 64
```

默认和推荐实验预算均按32对齐。旧checkpoint或显式自定义数量仍由原加载／验证规则处理，不静默舍入；resume优先恢复已保存拓扑，换H／G需用新实验。

### 原位置RoPE

- `--model.vision-router-preserve-hard-rope`与有效相机mask一起启用
- Hard取未压缩有效mask的`cumsum(mask)-1`对应位置，不把Top-K重新编号
- 文本和动作保持原位置；Global新增编号排在原动作horizon之后
- 前缀物理顺序`[Hard, text, Global]`；Global仍缓存到前缀KV，被每一步Expert去噪读取
- 例：512个有效图像位置、8个有效文本、10个动作槽；文本512–519，动作520–529，Global从530开始
- 保留较大的位置编号不会补回被删除token；新旧同预算推理层数相同，实际延迟仍需同后端测量

### 训练范围与精度

`--pytorch-train-mode router_llm_expert`训练Router、整个LLM、整个Action Expert及动作／时间投影，**仅冻结SigLIP和视觉projector**。旧`router_first4`保留为兼容选项，不是本路线默认。

新默认H96/G32拓扑有390个可训练张量、3,202,529,985个参数；既有H64/G64为3,202,534,113个。两者均有7个结构性张量不在动作loss路径，因此Adam状态为383组。`requires_grad=True`不代表每个张量都会被动作loss使用。

沿原π0.5混合精度：骨干Attention／MLP为BF16；动作、噪声、时间投影、Norm／AdaRMS及Router保留相应FP32边界。不要对整个模型统一调用BF16转换。

## 2. 启动新训练

两种硬件共用同一入口，先按实际环境选择**一组**变量：

```bash
# NVIDIA：指定有使用权限且空闲的卡
export TRAIN_PYTHON="$PWD/.venv/bin/python"
export OPENPI_DEVICE=cuda CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC=4 BATCH=64
```

```bash
# Ascend：参考8卡实验合同，全局BS256，每rank32
export TRAIN_PYTHON="$PWD/.venv-npu/bin/python"
export OPENPI_DEVICE=npu ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8 BATCH=256 TMPDIR=/tmp/openpi-npu
mkdir -p "$TMPDIR"
```

然后执行：

```bash
export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
export HF_LEROBOT_HOME="$(realpath data)"

"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$NPROC" \
  scripts/train_pytorch.py pi05_libero \
  --exp-name router_originalrope_r128_h96g32 \
  --pytorch-weight-path models/pi05_libero_pytorch \
  --checkpoint-base-dir checkpoints --assets-base-dir assets \
  --batch-size "$BATCH" --num-workers 4 \
  --num-train-steps 30000 --save-interval 10000 --seed 42 \
  --no-overwrite --no-resume --no-wandb-enabled \
  --pytorch-train-mode router_llm_expert --pytorch-training-precision bfloat16 \
  --model.pytorch-compile-mode None \
  --model.vision-router-hard-enabled --model.vision-router-hard-tokens 96 \
  --model.vision-router-soft-enabled --model.vision-router-soft-tokens 32 \
  --model.vision-router-mask-invalid-cameras \
  --model.vision-router-preserve-hard-rope \
  --vision-router-label-manifest outputs/prelabel8/manifest.json \
  --vision-router-label-variant gripper_k64 --vision-router-label-weight 0.01 \
  --vision-router-label-end-step 15000
```

CUDA示例的BS64是起步配置，不等同BS256实验；显存不足应在新实验启动前选择合理batch。`--batch-size`是全局值，必须能被rank数整除。不能拿first4训练的显存预算套完整后端训练。

- R64：两项token数量改为32，换新实验名
- 复现既有H64/G64：两项数量显式改为64，使用独立实验名；不能与新默认共用checkpoint目录
- Hard-only：Hard改为64或128，用`--model.no-vision-router-soft-enabled --model.vision-router-soft-tokens 0`真正移除Global，仍可保留原Hard RoPE
- 不用标签：移除三个`vision-router-label-*`来源／variant／weight参数及截止参数；该实验不再属于上述半程标签合同
- 每档从官方router-free权重独立初始化；仅允许缺少目标Router参数，不复用其他拓扑的优化器

训练数据仍是全部273,465帧，标签只覆盖stride8的采样帧。前15k为`L_flow + 0.01 L_Hard`，之后为`L_flow`；Global没有标签loss或teacher蒸馏。DDP按全局有效标签帧数归一化，后半程Hard／Global继续从flow获得梯度。

当前标签训练接口接受固定LIBERO快照`a4336d589d589045d1c56423ffdf3b88a0e19b1f`。通用预标注支持LeRobot v2双视图图片，但其他数据集仍需适配训练相机、动作和标签身份接口；不能伪报revision。`partial`小样manifest不能用于正式训练。

## 3. 保存与恢复

```text
checkpoints/pi05_libero/router_originalrope_r128_h96g32/10000/
  model.safetensors
  model_config.json
  optimizer.pt
  metadata.pt
  assets/physical-intelligence/libero/norm_stats.json
```

`model_config.json`格式3记录原RoPE拓扑；旧checkpoint缺省使用压缩后连续RoPE。推理自动恢复保存的模式，不能给旧模型临时开启新位置后沿用其成绩。

恢复时保留原实验名、batch、数据和训练时长，把上面的`--no-resume`换成`--resume`，移除`--pytorch-weight-path`，其余合同保持一致。拓扑／训练模式／标签来源／截止步从元数据恢复；resume不能换成另一个合同，也不会重新计15k。

此checkpoint包含模型、Adam和global step，但没有完整DataLoader游标／RNG续传证据，不承诺与未中断训练逐位相同。

## 4. 推理与兼容边界

- [模型服务与标准LIBERO评测](evaluation.md)：原版和Router共用服务入口，checkpoint决定拓扑
- 原RoPE仅用于本仓库PyTorch路径；JAX训练入口不支持该开关
- 旧连续位置TensorRT引擎不支持格式3；需适配显式位置／动作offset并重新编译，不能直接复用旧加速比
- `scripts/calibrate_token_budget.py`只做可选离线预算统计；不是此固定预算路线的前置步骤或在线teacher
- `scripts/render_token_attention.py`是开环诊断；任务成功率必须来自模型动作驱动的闭环仿真

已完成的旧混合／Hard-only结果不能当作新原RoPE模型成绩。原RoPE与全后端训练同时变化时，也不能把差异全部归因于RoPE。
