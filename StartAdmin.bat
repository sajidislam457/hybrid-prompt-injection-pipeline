@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title SecureAI Admin :3002
echo ============================================================
echo  SecureAI ADMIN (localhost only) — not the public chat
echo ============================================================
echo.

where node >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Node.js not found.
  pause
  exit /b 1
)

set PYTHONIOENCODING=utf-8
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

REM Stop any previous admin on 3002
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":3002 " ^| findstr LISTENING') do (
  echo [ADMIN] Stopping old process %%p on :3002
  taskkill /F /PID %%p >nul 2>&1
)

set "API_OK=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 3; if ($r.StatusCode -ge 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 set "API_OK=1"

if "!API_OK!"=="1" (
  echo [OK] Python API already healthy on :8000
  echo.
  goto start_admin
)

echo [API] Python API not running on :8000 — starting it...
> "%TEMP%\prompt_defense_api.cmd" (
  echo @echo off
  echo set PYTHONIOENCODING=utf-8
  echo cd /d "%~dp0"
  echo title Prompt Defense API :8000
  echo echo Starting API...
  echo "%PY%" run_api.py
  echo echo.
  echo echo API exited. Press any key to close.
  echo pause ^>nul
)
start "Prompt Defense API" "%TEMP%\prompt_defense_api.cmd"
echo [API] Waiting for http://127.0.0.1:8000/health ...

set /a "_i=0"
:wait_api
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
  echo [OK] API is healthy.
  echo.
  goto start_admin
)
set /a "_i+=1"
if !_i! geq 45 (
  echo [WARN] API health check timed out. Admin will still start.
  echo        Check the "Prompt Defense API" window for errors.
  echo.
  goto start_admin
)
timeout /t 1 /nobreak >nul
goto wait_api

:start_admin
if not exist "%~dp0admin\node_modules\" (
  echo [ADMIN] npm install...
  pushd "%~dp0admin"
  call npm install
  if errorlevel 1 (
    echo [ERROR] npm install failed.
    popd
    pause
    exit /b 1
  )
  popd
)

echo [ADMIN] Starting http://127.0.0.1:3002 ...
echo.
REM Open the browser after the server binds (opening first shows a failed tab).
start /b "" cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:3002/"
pushd "%~dp0admin"
call npm start
set "ADM_ERR=!errorlevel!"
popd
if not "!ADM_ERR!"=="0" (
  echo.
  echo [ERROR] Admin Node process exited ^(code !ADM_ERR!^).
  echo         Read the message above — often port 3002 already in use.
)
pause
endlocal
