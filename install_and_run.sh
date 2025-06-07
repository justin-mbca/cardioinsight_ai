#!/bin/bash

# CardioInsight AI - 安装和运行脚本

# 显示彩色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  CardioInsight AI - 安装和运行脚本   ${NC}"
echo -e "${BLUE}=======================================${NC}"

# 检查Python版本
echo -e "\n${YELLOW}检查Python版本...${NC}"
python_version=$(python3 --version 2>&1)
if [[ $python_version == *"Python 3"* ]]; then
    echo -e "${GREEN}检测到 $python_version${NC}"
else
    echo -e "${RED}错误: 未检测到Python 3。请安装Python 3.7或更高版本。${NC}"
    exit 1
fi

# 创建虚拟环境
echo -e "\n${YELLOW}创建虚拟环境...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}虚拟环境已存在，跳过创建步骤。${NC}"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 创建虚拟环境失败。请确保已安装venv模块。${NC}"
        exit 1
    fi
    echo -e "${GREEN}虚拟环境创建成功。${NC}"
fi

# 激活虚拟环境
echo -e "\n${YELLOW}激活虚拟环境...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 激活虚拟环境失败。${NC}"
    exit 1
fi
echo -e "${GREEN}虚拟环境激活成功。${NC}"

# 安装依赖
echo -e "\n${YELLOW}安装依赖...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 安装依赖失败。${NC}"
    exit 1
fi
echo -e "${GREEN}依赖安装成功。${NC}"

# 安装开发模式
echo -e "\n${YELLOW}以开发模式安装CardioInsight AI...${NC}"
pip install -e .
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 安装CardioInsight AI失败。${NC}"
    exit 1
fi
echo -e "${GREEN}CardioInsight AI安装成功。${NC}"

# 创建必要的目录
echo -e "\n${YELLOW}创建必要的目录...${NC}"
mkdir -p data/raw data/processed data/case_library models/ml_models models/dl_models models/optimized
echo -e "${GREEN}目录创建成功。${NC}"

# 运行示例
echo -e "\n${YELLOW}是否运行基本分析示例? (y/n)${NC}"
read -r run_example
if [[ $run_example == "y" || $run_example == "Y" ]]; then
    echo -e "\n${YELLOW}运行基本分析示例...${NC}"
    python examples/basic_analysis.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 运行示例失败。${NC}"
    else
        echo -e "${GREEN}示例运行成功。${NC}"
    fi
fi

echo -e "\n${BLUE}=======================================${NC}"
echo -e "${GREEN}CardioInsight AI 安装和设置完成!${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "\n使用说明:"
echo -e "1. 激活虚拟环境: ${YELLOW}source venv/bin/activate${NC}"
echo -e "2. 运行基本分析示例: ${YELLOW}python examples/basic_analysis.py${NC}"
echo -e "3. 运行多模态分析示例: ${YELLOW}python examples/multimodal_analysis.py${NC}"
echo -e "4. 运行远程医疗示例: ${YELLOW}python examples/remote_healthcare.py${NC}"
echo -e "5. 运行教学模块示例: ${YELLOW}python examples/teaching_module.py${NC}"
echo -e "\n更多信息请参考 ${YELLOW}README.md${NC} 和 ${YELLOW}INSTALL.md${NC} 文件。"

