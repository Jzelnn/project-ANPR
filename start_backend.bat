@echo off
title ANPR Parking System Server - Port 5001
color 0A
echo ========================================================
echo    MENJALANKAN SERVER ANPR PARKIR (AI & CCTV)...
echo ========================================================
echo.
echo Server sedang memuat model AI (YOLO + Body Type + OCR)...
echo Browser akan otomatis terbuka ke http://localhost:5001
echo.
echo Tekan CTRL + C di jendela ini jika ingin mematikan server.
echo ========================================================
echo.

cd /d "%~dp0"

:: Buka browser otomatis setelah 3 detik
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5001"

if exist "C:\Users\Lenovo\AppData\Local\Programs\Python\Python39\python.exe" (
    "C:\Users\Lenovo\AppData\Local\Programs\Python\Python39\python.exe" app.py
) else (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        py -3.9 app.py 2>nul || py app.py
    ) else (
        python app.py
    )
)

pause
