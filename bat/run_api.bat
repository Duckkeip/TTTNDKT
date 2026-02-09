@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d D:\git\TTTNDKT
py -m uvicorn api_server:app --reload
pause