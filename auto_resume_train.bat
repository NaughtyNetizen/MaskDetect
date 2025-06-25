@echo off
REM Auto Resume Training Script for ZoomNet
REM 自动断点恢复训练脚本

setlocal EnableDelayedExpansion

if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="/?" goto :show_help

REM 设置默认参数
set "CONFIG=./configs/zoomnet/cod_zoomnet.py"
set "DATASETS_INFO=./configs/_base_/dataset/dataset_configs.json"
set "MODEL_NAME="
set "BATCH_SIZE="
set "INFO="
set "FORCE_RESTART=0"
set "DRY_RUN=0"

REM 解析命令行参数
:parse_args
if "%1"=="" goto :end_parse
if "%1"=="--config" (
    set "CONFIG=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--model-name" (
    set "MODEL_NAME=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--batch-size" (
    set "BATCH_SIZE=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--info" (
    set "INFO=%2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--force-restart" (
    set "FORCE_RESTART=1"
    shift
    goto :parse_args
)
if "%1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto :parse_args
)
shift
goto :parse_args

:end_parse

echo ===================================================================
echo                ZoomNet 自动断点恢复训练脚本
echo                Auto Resume Training Script
echo ===================================================================
echo.

REM 检查并激活虚拟环境
set "VENV_PATH="
set "PYTHON_CMD=python"

REM 检查常见的虚拟环境路径
if exist ".\venv\Scripts\activate.bat" (
    set "VENV_PATH=.\venv"
    echo 找到虚拟环境 ^(Found virtual environment^): venv
) else if exist ".\env\Scripts\activate.bat" (
    set "VENV_PATH=.\env"
    echo 找到虚拟环境 ^(Found virtual environment^): env
) else if exist ".\.venv\Scripts\activate.bat" (
    set "VENV_PATH=.\.venv"
    echo 找到虚拟环境 ^(Found virtual environment^): .venv
) else if exist ".\Scripts\activate.bat" (
    set "VENV_PATH=."
    echo 找到虚拟环境 ^(Found virtual environment^): current directory
) else (
    echo 警告: 未找到虚拟环境，将使用系统Python ^(Warning: No virtual environment found, using system Python^)
    echo 建议创建虚拟环境: python -m venv venv
    echo Recommend creating virtual environment: python -m venv venv
)

REM 如果找到虚拟环境，则激活它
if defined VENV_PATH (
    echo 激活虚拟环境 ^(Activating virtual environment^): %VENV_PATH%
    call "%VENV_PATH%\Scripts\activate.bat"
    if errorlevel 1 (
        echo 错误: 无法激活虚拟环境 ^(Error: Failed to activate virtual environment^)
        pause
        exit /b 1
    )
    set "PYTHON_CMD=%VENV_PATH%\Scripts\python.exe"
)

REM 检查Python环境
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境 ^(Error: Python not found^)
    if defined VENV_PATH (
        echo 虚拟环境可能损坏，请重新创建虚拟环境
        echo Virtual environment might be corrupted, please recreate it
    ) else (
        echo 请确保Python已安装并添加到PATH环境变量中，或创建虚拟环境
        echo Please ensure Python is installed and added to PATH, or create a virtual environment
    )
    pause
    exit /b 1
)

echo 使用Python环境 ^(Using Python environment^): 
%PYTHON_CMD% --version

REM 检查配置文件
if not exist "%CONFIG%" (
    echo 错误: 配置文件不存在 ^(Error: Config file not found^): %CONFIG%
    pause
    exit /b 1
)

echo 配置文件 ^(Config file^): %CONFIG%
if defined MODEL_NAME echo 模型名称 ^(Model name^): %MODEL_NAME%
if defined BATCH_SIZE echo 批次大小 ^(Batch size^): %BATCH_SIZE%
if defined INFO echo 实验标签 ^(Experiment tag^): %INFO%
echo.

REM 构造Python命令参数
set "PYTHON_ARGS=auto_resume_train.py --config %CONFIG% --datasets-info %DATASETS_INFO%"
if defined MODEL_NAME set "PYTHON_ARGS=%PYTHON_ARGS% --model-name %MODEL_NAME%"
if defined BATCH_SIZE set "PYTHON_ARGS=%PYTHON_ARGS% --batch-size %BATCH_SIZE%"
if defined INFO set "PYTHON_ARGS=%PYTHON_ARGS% --info %INFO%"
if "%FORCE_RESTART%"=="1" set "PYTHON_ARGS=%PYTHON_ARGS% --force-restart"
if "%DRY_RUN%"=="1" set "PYTHON_ARGS=%PYTHON_ARGS% --dry-run"

echo 执行命令 ^(Executing command^): %PYTHON_CMD% %PYTHON_ARGS%
echo.

REM 执行Python脚本
%PYTHON_CMD% %PYTHON_ARGS%
set "EXIT_CODE=%errorlevel%"

echo.
if %EXIT_CODE% equ 0 (
    echo 脚本执行完成 ^(Script completed successfully^)
) else (
    echo 脚本执行失败 ^(Script failed^), 错误代码 ^(Error code^): %EXIT_CODE%
)

if "%DRY_RUN%"=="0" pause
exit /b %EXIT_CODE%

:show_help
echo ===================================================================
echo                ZoomNet 自动断点恢复训练脚本
echo                Auto Resume Training Script
echo ===================================================================
echo.
echo 使用方法 ^(Usage^):
echo   auto_resume_train.bat [options]
echo.
echo 选项 ^(Options^):
echo   --config ^<file^>        配置文件路径 ^(Config file path^)
echo                          默认: ./configs/zoomnet/cod_zoomnet.py
echo   --model-name ^<name^>    模型名称 ^(Model name^)
echo   --batch-size ^<size^>    批次大小 ^(Batch size^)
echo   --info ^<tag^>           实验标签 ^(Experiment tag^)
echo   --force-restart        强制重新开始训练 ^(Force restart training^)
echo   --dry-run              只检查状态 ^(Check status only^)
echo   --help, -h, /?         显示此帮助 ^(Show this help^)
echo.
echo 示例 ^(Examples^):
echo   auto_resume_train.bat
echo   auto_resume_train.bat --config ./configs/zoomnet/sod_zoomnet.py
echo   auto_resume_train.bat --model-name ZoomNet --batch-size 16
echo   auto_resume_train.bat --dry-run
echo   auto_resume_train.bat --force-restart --info demo
echo.
pause
exit /b 0
