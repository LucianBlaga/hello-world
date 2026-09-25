@echo off
cd /d "%~dp0"

rem Install into .venv the first time (or after a failed install).
if not exist .venv\installed.ok (
  if exist .venv (
    echo Removing an incomplete previous install...
    rmdir /s /q .venv
  )
  echo Creating virtual environment...
  python -m venv .venv || goto :fail
  call .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 goto :fail
  echo ok> .venv\installed.ok
) else (
  call .venv\Scripts\activate
)
python -m porchwatch %*
pause
exit /b

:fail
echo.
echo ============================================================
echo  Installation FAILED.
echo  If you saw "No such file or directory" or "Long Path" above,
echo  this folder path is too long for Windows:
echo    %~dp0
echo  Move the porchwatch folder somewhere short, e.g. C:\PorchWatch,
echo  and run start-windows.bat again.
echo ============================================================
if exist .venv rmdir /s /q .venv
pause
exit /b 1
