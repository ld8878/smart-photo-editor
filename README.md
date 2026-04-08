# 智能修图 - Smart Photo Editor

<div align="center">

**基于 AI 分析的智能修图配方提取与应用工具**

[English](#english) | [中文](#中文)

</div>

---

## 中文

### 项目简介

智能修图是一款创新的图片处理工具，能够自动分析「原图 → 修后图」的变化，提取出可复用的**修图配方（Recipe）**，然后将相同风格一键应用到其他图片上。

### 核心功能

- **智能分析** - 自动对比原图与修后图，提取曝光、对比度、饱和度、色温等参数
- **人像处理** - 检测人脸区域，提取磨皮、肤色、唇色等美颜参数
- **颜色迁移** - 基于直方图匹配的色彩风格迁移
- **配方复用** - 将提取的配方一键应用到新图片
- **AI 增强** - 可选的 OpenAI Vision API 深度分析
- **生成式编辑** - 可选的 DALL·E 图像编辑增强

### 技术栈

- **后端**: Python 3.9+ / FastAPI / OpenCV / NumPy
- **前端**: 原生 HTML/CSS/JavaScript（无框架依赖）
- **AI**: OpenAI Vision API（可选）

### 快速开始

#### 1. 克隆项目

```bash
git clone https://github.com/your-username/smart-photo-editor.git
cd smart-photo-editor
```

#### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 启动服务

```bash
./run.sh
# 或者
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 5. 访问应用

打开浏览器访问 http://localhost:8000

### 使用方法

1. **第一步 - 分析**
   - 上传一张原图和对应的修后图
   - 点击「开始分析」，系统自动提取修图配方

2. **第二步 - 应用**
   - 上传需要处理的新图片
   - 选择「应用配方」或「AI 增强」
   - 等待处理完成

3. **第三步 - 下载**
   - 查看处理前后对比
   - 点击「下载结果」保存图片

### 配置选项

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥（启用 AI 分析） | 无 |
| `OPENAI_VLM_MODEL` | Vision 模型 | `gpt-4o-mini` |
| `OPENAI_IMAGE_EDIT_MODEL` | 图像编辑模型 | `dall-e-2` |

### API 接口

#### 分析图片对

```http
POST /api/analyze
Content-Type: multipart/form-data

original: <原图文件>
edited: <修后图文件>
use_vlm: "true" | "false"
```

#### 应用配方

```http
POST /api/apply
Content-Type: multipart/form-data

new_image: <新图片文件>
recipe_json: <配方 JSON>
ref_original: <参考原图> (可选)
ref_edited: <参考修后图> (可选)
```

#### AI 增强

```http
POST /api/apply-generative
Content-Type: multipart/form-data

new_image: <新图片文件>
recipe_json: <配方 JSON>
description: <描述文本>
```

### 配方结构

```json
{
  "version": 1,
  "global": {
    "exposure": 0.15,
    "contrast": 1.1,
    "saturation": 1.2,
    "temperature": -0.1,
    "highlights": 0.05,
    "shadows": 0.1,
    "grain": 0.0,
    "vignette": 0.0
  },
  "face": {
    "skin_smooth_strength": 0.3,
    "skin_warmth": 0.1,
    "lip_saturation": 0.05,
    "face_slim_approx": 0.0,
    "notes": ""
  },
  "color_grading": {
    "use_lut": true,
    "lut_id": "pair_histogram",
    "histogram_match_strength": 0.7
  }
}
```

### 项目结构

```
smart-photo-editor/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 入口
│   ├── analysis.py      # 图片分析逻辑
│   ├── recipe.py        # 配方定义与应用
│   ├── pipeline.py      # 处理流水线
│   ├── cv_metrics.py    # CV 指标计算
│   ├── face.py          # 人脸检测与处理
│   ├── lut.py           # 颜色查找表
│   └── gen_fallback.py  # AI 生成增强
├── static/
│   └── index.html       # 前端界面
├── requirements.txt
├── run.sh
└── README.md
```

### 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 开源协议

本项目采用 MIT 协议开源 - 详见 [LICENSE](LICENSE) 文件

---

## English

### Overview

Smart Photo Editor is an innovative image processing tool that automatically analyzes the changes between "original → edited" images, extracts reusable **editing recipes**, and applies the same style to other images with one click.

### Key Features

- **Smart Analysis** - Automatically compare original and edited images to extract exposure, contrast, saturation, temperature, etc.
- **Portrait Processing** - Detect face regions, extract beauty parameters like skin smoothing, skin tone, lip color
- **Color Transfer** - Histogram matching based color style transfer
- **Recipe Reuse** - Apply extracted recipes to new images with one click
- **AI Enhancement** - Optional OpenAI Vision API for deep analysis
- **Generative Editing** - Optional DALL·E image editing enhancement

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/smart-photo-editor.git
cd smart-photo-editor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
./run.sh
```

Visit http://localhost:8000 in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Analyze image pair and extract recipe |
| `/api/apply` | POST | Apply recipe to new image |
| `/api/apply-generative` | POST | AI-enhanced image editing |
| `/api/health` | GET | Health check |

### License

This project is licensed under the MIT License.

---

<div align="center">

Made with ❤️ by [Your Name]

</div>
