@echo off
cd /d D:\git\TTTNDKT
py -m uvicorn api_server:app --reload
pause