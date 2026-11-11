@echo off
REM Demarrage de l'agent Prematch AI (bot Telegram + scheduler).
REM Prerequis : .env rempli (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
cd /d "%~dp0.."
if exist .venv\Scripts\python.exe (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)
%PYTHON% -m prematch ingest-fixtures
%PYTHON% -m prematch bot
