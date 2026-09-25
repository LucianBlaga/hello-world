@echo off
setlocal
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 goto :nopython

rem 1) Our own virtual environment, if a previous install completed.
if exist .venv\installed.ok (
  call .venv\Scripts\activate
  goto :run
)

rem 2) Packages already installed for this Python? Just use them.
python -c "import cv2, yaml, flask, ultralytics, fast_alpr" >nul 2>&1
if not errorlevel 1 (
  echo Using the packages already installed for this Python.
  goto :run
)

rem 3) Fresh install into .venv
if exist .venv (
  echo Removing an incomplete previous install...
  rmdir /s /q .venv
)
echo Creating virtual environment in %CD%\.venv
python -m venv .venv
if errorlevel 1 goto :failvenv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :failpip
echo ok> .venv\installed.ok

:run
python -m porchwatch %*
pause
exit /b

:nopython
echo Python was not found. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
pause
exit /b 1

:failvenv
echo.
echo ============================================================
echo  Could not create the virtual environment - see the error above.
echo  Copy the text above and send it to Claude.
echo ============================================================
if exist .venv rmdir /s /q .venv
pause
exit /b 1

:failpip
echo.
echo ============================================================
echo  Package installation FAILED - see the error above.
echo  If it says "No such file or directory" or mentions Long Path,
echo  move this folder somewhere short such as C:\PorchWatch.
echo  Folder now: %~dp0
echo ============================================================
if exist .venv rmdir /s /q .venv
pause
exit /b 1
