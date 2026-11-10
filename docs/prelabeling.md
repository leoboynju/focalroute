# Qwen8B预标注 → Router训练

**三步：下载标注模型 → 生成稀疏正标签 → 用独立训练环境训练。**

公开入口：[`scripts/prelabel_lerobot.py`](../scripts/prelabel_lerobot.py)。不依赖本机的`experiments/`、服务或绝对路径。

## 1. 准备独立标注环境

以下是Linux／Python3.11／CUDA12.8的标注路径。训练依赖固定`transformers==4.53.2`及仓库替换实现，Qwen3-VL使用`4.57.1`；**不要把标注依赖装进训练环境**。

```bash
uv venv --python 3.11 .venv-label
uv pip install --python .venv-label/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv-label/bin/python -r scripts/requirements-label.txt
```

在仓库根目录执行后续命令。先用`nvidia-smi`选真正空闲的卡；8B以BF16运行，SAM以FP32运行，两个阶段不同时驻留。48GB卡可先用BS16，小显存从BS1试起；不会自动量化或回退CPU。

## 2. 下载Qwen8B与SAM

已存在相同资产时直接复用，不必重新下载。

```bash
.venv-label/bin/hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir models/Qwen3-VL-8B-Instruct

mkdir -p models/sam
curl --fail --location --continue-at - \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  --output models/sam/sam_vit_h_4b8939.pth
```

脚本只从本地读取模型，不会在执行中偷偷下载。模型权重不随源码发布；许可分别以[Qwen模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)和[SAM](https://github.com/facebookresearch/segment-anything)为准。

跨硬件／batch的回答不保证逐字相同；对照实验应冻结实际manifest SHA和配置，而不只记录prompt。

## 3. 准备数据与通用prompt

本入口读取**LeRobot v2图片格式**：`meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`及对应Parquet。默认图片列为`image`、`wrist_image`；其他列用`--image-keys <base列> <wrist列>`指定。暂不解码视频列或LeRobot v3文件布局。

复现LIBERO时固定数据快照；已有此快照可跳过下载：

```bash
.venv-label/bin/hf download physical-intelligence/libero --repo-type dataset \
  --revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
  --local-dir data/physical-intelligence/libero
```

[`router_label_prompt.txt`](../scripts/router_label_prompt.txt)定义四角色：

| 角色 | 含义 |
|---|---|
| source | 当前被操作／改变状态的物体或部件；已抓起时标当前位置 |
| goal | 可见目标、容器、目标表面；没有独立目标时不编造 |
| arm | 机械臂连杆和关节，不含末端与被抓物 |
| gripper | 末端本体、手指、夹爪或吸盘，不预设机器人外形与朝向 |

每张图只做一次全图四角色查询，任务文本直接取该帧`task_index`对应的原始`task`，替换`{{INSTRUCTION_JSON}}`。无手写物体表、无32B、无第二次裁剪定位或8B一致性复核。无法确定的区域保持unknown，不把缺失框当背景负标签。

**通用的是标注语义，不是任意数据集训练适配器。** 现有训练标签加载器严格接受上述LIBERO快照；换其他数据集还须适配数据／相机／动作／标签加载接口，不能把其他数据伪报为这个revision来绕过校验。

## 4. 先小样检查，再生成全量

纯CPU格式自检，不加载模型：

```bash
PYTHONPATH=packages/openpi-client/src \
  .venv-label/bin/python scripts/prelabel_lerobot.py --self-check
```

小样：两视图合成一个时刻，仅取episode0第一帧。`CUDA_VISIBLE_DEVICES`填实际空闲物理卡；`cuda:0`表示其中第一张可见卡。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=packages/openpi-client/src \
.venv-label/bin/python scripts/prelabel_lerobot.py \
  --dataset data/physical-intelligence/libero \
  --dataset-revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
  --qwen8b models/Qwen3-VL-8B-Instruct \
  --sam-checkpoint models/sam/sam_vit_h_4b8939.pth \
  --device cuda:0 --batch-size 1 \
  --episodes 0 --max-moments 1 --output outputs/prelabel8_smoke
```

先查看`ground/`原始回答以及`labels/`中的mask和正标签。小样只产生`preview.complete`，manifest为`partial`，**不能拿它冒充全量标注训练**。

全量：每episode每8帧取1帧，两路各查询一次。该快照为34,918时刻／69,836图片，覆盖40任务、1,693episode；BS16指单视图图片数，不是训练batch。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=packages/openpi-client/src \
.venv-label/bin/python scripts/prelabel_lerobot.py \
  --dataset data/physical-intelligence/libero \
  --dataset-revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
  --qwen8b models/Qwen3-VL-8B-Instruct \
  --sam-checkpoint models/sam/sam_vit_h_4b8939.pth \
  --device cuda:0 --batch-size 16 --stride 8 --output outputs/prelabel8
```

### 产物与断点

- `source_identity.json`：数据metadata、模型配置／文件stat、prompt、图像列和采样范围。固定revision是下载来源声明；脚本不冒充对全部模型权重做了上游SHA审计。
- `ground/`：原始`.txt`与解析记录；有效回答不重推，坏格式只补失败图片，旧回答改名保留。回退只换输出表示／抑制重复，不手选坐标。
- `labels/.../masks.npz`：`maps[4,2,224,224]`，角色顺序为source、goal、gripper、arm；原图按训练同款resize-with-pad映射到224。
- `labels.npz`：`gripper_k64_positive[768]`；两有效视图共512位置，第三相机256位置全0。只保存最多64个正patch，**不把补位token变成正标签**。
- `manifest.json`与`annotation.complete`：全量检查通过才发布，marker绑定manifest SHA。格式失败时整个时刻隔离为零Hard标签，仍保留flow；隔离比例大于0.1%不发布完成标记。

重新执行相同命令可续跑；已完成的产物只核小收据，不重新加载模型或全扫数组。`--prepare-only`仅抽帧；`--finish-existing`要求所有Qwen回答已有效缓存，只接SAM，不加载Qwen。

SAM采用单mask输出、像素概率≥0.5，权重为概率×预测质量；质量门槛0.55不是语义准确率。mask裁剪在框内，source／goal额外保留3像素边缘；不做context14扩张，context=0。正patch要求至少25%像素支持，source／goal／gripper按1.0／0.6／0.8形成正标签分布，纯arm不作为正目标。H128也复用K64正标签，不补成128个伪正例。腕部／关节仍可能混淆：这些是粗伪标签，不是人工真值。

## 5. 接入最新原RoPE训练

全量`annotation.complete`生成后，使用[原RoPE R128训练命令](router.md#2-启动新训练)。该命令已经接入本页的`outputs/prelabel8/manifest.json`，CUDA／Ascend仅切换解释器、设备和batch；**不使用Qwen环境训练**。

| 项目 | 当前路线 |
|---|---|
| Router | 默认H96/G32，跨两有效视图共享128预算；90%分数口径及R64／Hard-only修改见训练指南 |
| 位置 | Hard、文本、动作保留原RoPE；Global追加到原动作位置之后 |
| 可训练范围 | `router_llm_expert`；仅SigLIP和视觉projector冻结 |
| 辅助监督 | `--vision-router-label-end-step 15000`；前15k权重.01，后15k为0 |
| 动作训练 | 所有帧保留flow；Global无伪标签loss或teacher蒸馏 |

训练读取`assets/pi05_libero/physical-intelligence/libero/norm_stats.json`中的官方统计，不从伪标签重算。跨机器时一起复制数据metadata、manifest和全部标签子目录，保留相对目录与原dataset index。

同一checkpoint不能换拓扑、训练范围或截止合同；每档从官方权重独立初始化。参考8卡全局BS256＝每rank32，30k步、seed42；默认LR warmup10k至5e-5、随后保持5e-5。AdamW β=(.9,.95)、ε=1e-8、weight_decay=1e-10、clip=1.0。恢复从保存的global step继续，不重新计15k。

标注完成不等于训练成功，训练完成也不等于任务成功。每个新模型另做40任务×50初态的标准LIBERO闭环；smoke、开环叠图和未完成的PLUS结果不计作最终成绩。
