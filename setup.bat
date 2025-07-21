@echo off
:: Knowledge Extraction AI Agent - Setup Script for Windows
:: This script helps you set up the project quickly on Windows

echo 🤖 Knowledge Extraction AI Agent - Setup
echo =======================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3 is required but not installed.
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found

:: Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

:: Install requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt

:: Copy environment file
if not exist .env (
    echo ⚙️ Creating environment file...
    copy .env.example .env
    echo 📝 Please edit .env file with your API keys before running the application
) else (
    echo ✅ Environment file already exists
)

echo.
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys:
echo    - CLAUDE_API_KEY (from https://console.anthropic.com/)
echo    - OPENAI_API_KEY (from https://platform.openai.com/api-keys)
echo    - AZURE_API_KEY (optional, for secure processing)
echo.
echo 2. Launch the application:
echo    streamlit run ui_app.py
echo.
echo 3. Open your browser to: http://localhost:8501
echo.
pause