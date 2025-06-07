# CardioInsight AI

CardioInsight AI是一个多模态智能心电分析与辅助诊疗系统，旨在提供高精度、可解释的心电图分析，支持多导联分析、可视化解释、多模态融合、动态ECG标注、医生辅助教学和远程医疗适配。

## 系统特点

1. **多导联智能诊断**：支持12导联心电图分析，提高诊断广度与准确度
2. **AI诊断 + 可解释性**：使用Grad-CAM和Attention可视化异常波段，提供置信度评估
3. **多模态融合**：结合心电图数据、病史和症状描述，模拟医生临床思路
4. **动态ECG自动标注**：支持24小时Holter数据分析，自动标注异常事件
5. **AI医生辅助教学系统**：提供案例库、模拟考试和错题分析功能
6. **基层/远程医疗适配优化**：模型轻量化，支持离线处理和远程会诊

## 安装说明

### 系统要求

- Python 3.7+
- 推荐使用GPU进行深度学习模型训练和推理

### 安装步骤

1. 克隆代码库：

```bash
git clone https://github.com/cardioinsight/cardioinsight-ai.git
cd cardioinsight-ai
```

2. 安装依赖：

```bash
pip install -e .
```

或者直接安装依赖：

```bash
pip install -r requirements.txt
```

## 使用示例

### 基本心电图分析

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

### 多模态分析

```python
# 准备临床数据
clinical_data = {
    "symptoms": ["胸痛", "气短", "心悸"],
    "medical_history": ["高血压", "糖尿病"],
    "demographics": {
        "age": 65,
        "gender": "male",
        "bmi": 28.5
    }
}

# 多模态分析
results = system.analyze_with_multimodal(ecg_data, clinical_data, metadata)
```

### Holter数据分析

```python
# 加载Holter数据
holter_data, metadata = system.load_ecg_data("path/to/holter_data.mat")

# 分析Holter数据
results = system.analyze_holter(holter_data, metadata)

# 获取报告
report = results["report"]
```

### 医生辅助教学

```python
# 创建教学测验
quiz = system.create_teaching_quiz(n_questions=10, difficulty="medium")

# 开始测验会话
session = system.teaching_system.start_quiz_session(quiz)

# 获取当前问题
question = session.get_current_question()

# 回答问题
is_correct = session.answer_question("Normal Sinus Rhythm")

# 获取测验结果
results = session.get_results()
```

### 远程医疗功能

```python
# 处理心电图数据用于远程会诊
result = system.process_for_remote(ecg_data, metadata={
    "patient_id": "12345",
    "requires_consultation": True,
    "priority": "high"
})

# 检查会诊状态
status = system.check_consultation_status(result["consultation_case_id"])

# 获取会诊结果
consultation = system.get_consultation_result(result["consultation_case_id"])
```

### 命令行使用

```bash
# 分析心电图文件
cardioinsight --ecg path/to/ecg_file.csv --output results.json --use-dl --explain

# 优化模型用于远程部署
cardioinsight --optimize
```

## 模块结构

- **ecg_preprocessing.py**: 心电图预处理模块
- **feature_extraction.py**: 特征提取模块
- **ml_models.py**: 机器学习模型模块
- **dl_models.py**: 深度学习模型模块
- **explainability.py**: 可解释性模块
- **multimodal_fusion.py**: 多模态融合模块
- **dynamic_ecg_annotation.py**: 动态ECG自动标注模块
- **teaching_module.py**: AI医生辅助教学模块
- **remote_healthcare.py**: 基层/远程医疗适配模块
- **main.py**: 主程序

## 许可证

MIT License

## 联系方式

- 项目主页：https://github.com/cardioinsight/cardioinsight-ai
- 问题反馈：issues@cardioinsight.ai

