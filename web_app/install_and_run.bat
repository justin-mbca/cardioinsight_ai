@echo off
REM CardioInsight AI - 安装和运行脚本 (Windows版)

echo =======================================
echo   CardioInsight AI - 安装和运行脚本
echo =======================================

REM 检查Python版本
echo.
echo 检查Python版本...
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未检测到Python。请安装Python 3.7或更高版本。
    exit /b 1
)

REM 创建虚拟环境
echo.
echo 创建虚拟环境...
if exist venv (
    echo 虚拟环境已存在，跳过创建步骤。
) else (
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo 错误: 创建虚拟环境失败。请确保已安装venv模块。
        exit /b 1
    )
    echo 虚拟环境创建成功。
)

REM 激活虚拟环境
echo.
echo 激活虚拟环境...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 激活虚拟环境失败。
    exit /b 1
)
echo 虚拟环境激活成功。

REM 安装依赖
echo.
echo 安装依赖...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 安装依赖失败。
    exit /b 1
)
echo 依赖安装成功。

REM 安装开发模式
echo.
echo 以开发模式安装CardioInsight AI...
pip install -e .
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 安装CardioInsight AI失败。
    exit /b 1
)
echo CardioInsight AI安装成功。

REM 创建必要的目录
echo.
echo 创建必要的目录...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\case_library mkdir data\case_library
if not exist models\ml_models mkdir models\ml_models
if not exist models\dl_models mkdir models\dl_models
if not exist models\optimized mkdir models\optimized
echo 目录创建成功。

REM 运行示例
echo.
set /p run_example=是否运行基本分析示例? (y/n): 
if /i "%run_example%"=="y" (
    echo.
    echo 运行基本分析示例...
    python examples\basic_analysis.py
    if %ERRORLEVEL% NEQ 0 (
        echo 错误: 运行示例失败。
    ) else (
        echo 示例运行成功。
    )
)

echo.
echo =======================================
echo CardioInsight AI 安装和设置完成!
echo =======================================
echo.
echo 使用说明:
echo 1. 激活虚拟环境: venv\Scripts\activate
echo 2. 运行基本分析示例: python examples\basic_analysis.py
echo 3. 运行多模态分析示例: python examples\multimodal_analysis.py
echo 4. 运行远程医疗示例: python examples\remote_healthcare.py
echo 5. 运行教学模块示例: python examples\teaching_module.py
echo.
echo 更多信息请参考 README.md 和 INSTALL.md 文件。

pause

