# CardioInsight AI 安装和使用指南

本文档提供了CardioInsight AI系统的详细安装步骤和基本使用说明。

## 安装步骤

### 1. 系统要求

- Python 3.7+
- 推荐使用GPU进行深度学习模型训练和推理
- 至少4GB RAM（推荐8GB+）
- 至少10GB磁盘空间

### 2. 安装依赖

#### 方法一：使用pip直接安装

```bash
# 克隆代码库
git clone https://github.com/cardioinsight/cardioinsight-ai.git
cd cardioinsight-ai

# 安装依赖
pip install -r requirements.txt
```

#### 方法二：使用setup.py安装

```bash
# 克隆代码库
git clone https://github.com/cardioinsight/cardioinsight-ai.git
cd cardioinsight-ai

# 安装包及其依赖
pip install -e .
```

#### 方法三：创建虚拟环境（推荐）

```bash
# 克隆代码库
git clone https://github.com/cardioinsight/cardioinsight-ai.git
cd cardioinsight-ai

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\\Scripts\\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 验证安装

安装完成后，可以运行以下命令验证安装是否成功：

```bash
# 如果使用方法二安装
cardioinsight --help

# 或者直接运行主程序
python -m cardioinsight_ai.main --help
```

如果显示帮助信息，则表示安装成功。

## 基本使用

### 1. 命令行使用

```bash
# 分析心电图文件
cardioinsight --ecg path/to/ecg_file.csv --output results.json --use-dl --explain

# 优化模型用于远程部署
cardioinsight --optimize
```

### 2. Python API使用

```python
from cardioinsight_ai.main import CardioInsightAI

# 初始化系统
system = CardioInsightAI()

# 加载心电图数据
ecg_data, metadata = system.load_ecg_data("path/to/ecg_file.csv")

# 分析心电图
results = system.analyze_ecg(ecg_data, metadata, use_dl=True, explain=True)

# 打印结果
print(f"预测结果: {results['prediction']}")
print(f"置信度: {results['confidence']}")

# 保存结果
system.save_results(results, "analysis_results.json")
```

## 配置系统

CardioInsight AI系统可以通过配置文件进行配置。配置文件是一个JSON文件，包含以下选项：

```json
{
  "data_dir": "data",
  "models_dir": "models",
  "results_dir": "results",
  "use_gpu": true,
  "default_model": "dl_model",
  "server_url": "https://api.cardioinsight.ai",
  "api_key": "your_api_key"
}
```

可以通过以下方式指定配置文件：

```bash
cardioinsight --config path/to/config.json --ecg path/to/ecg_file.csv
```

或者在Python API中：

```python
system = CardioInsightAI("path/to/config.json")
```

## 常见问题

### 1. 安装依赖时出错

如果在安装依赖时遇到问题，可以尝试以下解决方案：

- 确保已安装最新版本的pip：`pip install --upgrade pip`
- 对于TensorFlow安装问题，请参考[TensorFlow官方安装指南](https://www.tensorflow.org/install)
- 对于特定平台的问题，可能需要安装额外的系统依赖

### 2. GPU支持

要启用GPU支持，请确保已安装CUDA和cuDNN，并安装tensorflow-gpu：

```bash
pip install tensorflow-gpu
```

### 3. 数据格式支持

CardioInsight AI支持以下格式的心电图数据：

- CSV文件（每列代表一个导联，可选时间列）
- NumPy数组文件（.npy）
- MATLAB文件（.mat）
- WFDB格式（使用wfdb库读取）

## 获取帮助

如果您在安装或使用过程中遇到任何问题，请联系我们：

- 项目主页：https://github.com/cardioinsight/cardioinsight-ai
- 问题反馈：issues@cardioinsight.ai

