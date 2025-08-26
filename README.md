# 2025全球AI攻防挑战赛-赛道一：图片全要素交互认证-生成赛 实现与报告

## 简介
本仓库记录我在[2025全球AI攻防挑战赛-赛道一：图片全要素交互认证-生成赛](https://tianchi.aliyun.com/competition/entrance/532389/information)的参赛实现与工程方案。


生成赛包含四个任务，总的来说目的均是生成高质量的“虚假”图片：
1. **T2I**（Text-to-Image）文字生成图像（根据给定的文本提示，生成高质量目标图像）
2. **TIE**（Text-Instructed Image Editing）指令式图像编辑（根据提供的指令，对原图进行编辑）
3. **VTTIE**（Vision-Text-To-Image Editing）视觉+文字指令编辑  （同TIE，但是仅针对文字信息）
4. **Deepfake** 人脸替换（将target的脸部高质量地移植到origin的脸部）

我的目标是在尽量使用只**开源**模型的前提下，尝试一组包含多模态内容生成、编辑以及评估的完整流水线，并以比赛结果作为对整个流水线的有效性验证。

详细的实现参考下面代码和示例结果以及说明文档参考[docs](./docs)目录。

## 仓库结构
```text
root/
├── code/                 # 核心代码
│   ├── common/           # 通用工具：翻译、自动化评分、公共函数等
│   ├── deepfake/         # Deepfake 任务实现
│   ├── t2i/              # T2I 任务实现
│   ├── tie/              # TIE 任务实现
│   └── vttie/            # VTTIE 任务实现
│
├── configs/              # 各任务的 YAML 配置文件
│   ├── deepfake.yaml
│   ├── t2i.yaml
│   ├── tie.yaml
│   └── vttie.yaml
│
├── data/                 # 输入数据与中间文件
│   ├── imgs/             # 比赛提供的原始图片
│   ├── imgs_deepfake_sr/ # Deepfake任务超分后的清晰化图像
│   ├── task.csv          # 比赛提供的原始任务表
│   ├── task_en.csv       # 翻译后的任务表
│   ├── prompt_map.csv    # 翻译过的Prompt映射表
│   └── translate_cache.json # 翻译缓存
│
├── results/              # 各任务的运行结果（仅部分示例）
│   ├── deepfake/         # Deepfake 生成结果
│   ├── t2i/              # T2I 生成结果
│   ├── tie/              # TIE 编辑结果
│   └── vttie/            # VTTIE 编辑结果
│
├── docs/                 # 报告与详细说明
│   ├── report_zh.pdf     # 中文报告
│   └── report_en.pdf     # 英文报告
│
├── README.md             # 中文README
└── README_en.md          # 英文README
```

## 所用开源模型及工具

### 任务1: T2I - 文生图
  - SDXL-base/refiner-1.0
  - Qwen-Image
  - Real-ESRGAN 

### 任务2/3: TIE/VTTIE - 图像编辑
  - FLUX.1-Kontext-Dev
  - Qwen-Image-Edit

### 任务4: Deepfake - 人脸替换
  - Insightface+Inswapper
  - GFPGAN
  - SDXL-base-1.0 + InstantID
  - FLUX.1-Kontext-Dev

### 模拟测评
  - Qwen2.5-VL-72B-Instruct


## 比赛结果
[最终成绩](https://tianchi.aliyun.com/competition/entrance/532389/rankingList)：7/891 