@echo off
title Install Requirements ANPR System
color 0B
echo ========================================================
echo   MENGINSTAL DEPENDENSI ANPR SYSTEM...
echo ========================================================
echo.
cd /d "%~dp0"
pip install -r requirements.txt
echo.
echo ========================================================
echo   INSTALASI SELESAI!
echo   Sekarang Anda bisa menjalankan start_backend.bat
echo ========================================================
pause
