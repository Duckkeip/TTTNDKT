@echo off
cd /d D:\git\TTTNDKT

call venv\Scripts\activate
python -m streamlit run app.py

pause