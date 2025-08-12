# Datawhale AI夏令营 2025
# 2025全球AI攻防挑战赛-赛道一：图片全要素交互认证-生成赛

## 临时笔记

## 项目概述
本项目包含比赛的四个任务：
1. **T2I**（Text-to-Image）文字生成图像  
2. **TIE**（Text-Instructed Image Editing）指令式图像编辑  
3. **VTTIE**（Vision-Text-To-Image Editing）视觉+文字指令编辑  
4. **Deepfake** 人脸替换  

目标是全部利用**开源模型**或者本地方法完成任务。

---

## 任务 1: T2I - 文字生成图像

### 方法
- 使用 **CogView4**（`diffusers` 提供的 `CogView4Pipeline`）进行生成。  
- 配置：
  - `guidance_scale=3.5`
  - `num_inference_steps=50`
  - 输出分辨率：`512×512`

### 遇到的问题 & 解决
- 暂无
- 可以尝试Stable Diffusion进行对比
- prompt支持中文，但是可以用英文prompt对比
- 对prompt进行人工增强，对比效果

---

## 任务 2: TIE - 指令式图像编辑

### 方法
- 使用 **FLUX.1-Kontext**（`FluxKontextPipeline`）进行编辑。
- 配置：
  - `height=1024`, `width=1024`
  - `guidance_scale=2.5`
  - `num_inference_steps=28`

### 遇到的问题 & 解决
1. **无法直接导入 FluxKontextPipeline**  
   - 原因：`diffusers` 版本过低 → 升级到支持 Flux 模型的版本。
2. **显存 OOM**  
   - `enable_model_cpu_offload()` 会导致 OOM → 改用 `enable_sequential_cpu_offload()`，虽然速度下降但能稳定运行。  
   - 调整分辨率为 768 或 512 可以进一步降低显存占用。

---

## 任务 3: VTTIE - 视觉+文字指令编辑

### 方法
- 同 TIE 使用 **FLUX.1-Kontext**，只是编辑参数略有调整：
  - `guidance_scale=2.3`
  - `num_inference_steps=30`

### 遇到的问题 & 解决
- 与 TIE 一致
- 可尝试更多参数变化对比效果
- FLUX不支持中文prompt，需翻译为英文
- 同任务一一样，可以对prompt进行人工增强

---

## 任务 4: Deepfake - 人脸替换

### 方法
#### 第一阶段：传统方法（OpenCV + dlib）
- 使用 **dlib** 检测人脸 + 68 关键点 → Delaunay 三角剖分 + 仿射变换 → 拼接到目标人脸位置。
- 使用 `cv2.seamlessClone` 做融合

#### 问题
- **效果差**，融合不自然（肤色差异、脸型不匹配、姿态不匹配、未考虑图片曝光差异）。
- 逻辑错误：初始版本将 ori 的脸换到 target 中，和赛题要求相反

#### 第二阶段：深度学习方法（InsightFace - INSwapper）
- 使用 **inswapper_128.onnx** 模型
- 流程：
  1. 检测并对齐人脸（ArcFace / RetinaFace）
  2. 编码 target 脸部特征
  3. 替换 ori 中的脸
  4. 融合结果
- 优点：融合自然、细节保留好，较第一阶段方法有明显提升
- 问题：个别图片组检测不到人脸 → 后续需要分析失败案例单独分析

---

## 通用问题与优化记录
1. **中文 Prompt 支持**  
2. **运力不足，显存优化**  

---

## 当前进展
- **任务 1~4** 已完成批处理脚本，成功输出所有任务的初步结果

## 参考
- [Datawhale AI夏令营 2025](https://www.datawhale.cn/activity/360/learn/204/4491)
- [Datawhale 2025全球AI攻防挑战赛 GitHub template code](https://github.com/datawhalechina/competition-baseline/tree/master/competition/2025%E5%85%A8%E7%90%83AI%E6%94%BB%E9%98%B2%E6%8C%91%E6%88%98%E8%B5%9B)