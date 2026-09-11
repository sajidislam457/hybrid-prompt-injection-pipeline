@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

title Prompt Defense - Launcher
echo ============================================================
echo  Prompt Injection Defense System
echo ============================================================
echo.

REM ---- Python (prefer venv) ----
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
where "%PY%" >nul 2>&1
if errorlevel 1 (
  "%PY%" --version >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Python not found. Install Python or create .venv first.
    pause
    exit /b 1
  )
)

REM ---- Node / npm ----
where node >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Node.js not found. Install Node.js 18+ and reopen the terminal.
  pause
  exit /b 1
)
where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] npm not found. Reinstall Node.js with npm included.
  pause
  exit /b 1
)

REM ---- Free stuck ports if process is dead-listening (optional soft check) ----
call :port_listening 8000
if not errorlevel 1 (
  echo [API] Port 8000 already in use - will reuse it.
  set "START_API=0"
) else (
  set "START_API=1"
)

call :port_listening 3001
if not errorlevel 1 (
  echo [WEB] Port 3001 already in use - will reuse it.
  set "START_WEB=0"
) else (
  set "START_WEB=1"
)

REM ---- Write tiny launchers (avoids nested-quote bugs in start) ----
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

> "%TEMP%\prompt_defense_web.cmd" (
  echo @echo off
  echo cd /d "%~dp0web"
  echo title Prompt Defense Web :3001
  echo echo Starting Web UI...
  echo call npm start
  echo echo.
  echo echo Web exited. Press any key to close.
  echo pause ^>nul
)

if "%START_API%"=="1" (
  echo [API] Starting http://localhost:8000 ...
  start "Prompt Defense API" "%TEMP%\prompt_defense_api.cmd"
) else (
  echo [API] Already running.
)

if "%START_WEB%"=="1" (
  if not exist "%~dp0web\node_modules\" (
    echo [WEB] Installing npm packages ^(first time^)...
    pushd "%~dp0web"
    call npm install
    if errorlevel 1 (
      echo [ERROR] npm install failed.
      popd
      pause
      exit /b 1
    )
    popd
  )
  echo [WEB] Starting http://localhost:3001 ...
  start "Prompt Defense Web" "%TEMP%\prompt_defense_web.cmd"
) else (
  echo [WEB] Already running.
)

echo.
echo Waiting for API ^(port 8000^)...
call :wait_http http://127.0.0.1:8000/health 60
if errorlevel 1 (
  echo [WARN] API health check timed out.
  echo        Open the "Prompt Defense API" window and read the error.
) else (
  echo [OK] API is healthy.
)

echo Waiting for Web ^(port 3001^)...
call :wait_port 3001 45
if errorlevel 1 (
  echo [WARN] Web port 3001 not open yet.
  echo        Open the "Prompt Defense Web" window and read the error.
) else (
  echo [OK] Web is listening.
)

echo.
echo Opening browser: http://localhost:3001
start "" "http://localhost:3001/"

echo.
echo Keep the API and Web console windows open while you use the app.
echo To stop: close those two console windows.
echo.
pause
endlocal
exit /b 0

REM ========== helpers ==========
:port_listening
REM returns 0 if listening, 1 if not
netstat -ano | findstr ":%~1 " | findstr "LISTENING" >nul 2>&1
exit /b %ERRORLEVEL%

:wait_port
set "_port=%~1"
set "_max=%~2"
set /a "_i=0"
:wp_loop
call :port_listening %_port%
if not errorlevel 1 exit /b 0
set /a "_i+=1"
if %_i% geq %_max% exit /b 1
timeout /t 1 /nobreak >nul
goto wp_loop

:wait_http
set "_url=%~1"
set "_max=%~2"
set /a "_i=0"
:wh_loop
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -Uri '%_url%' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 exit /b 0
set /a "_i+=1"
if %_i% geq %_max% exit /b 1
timeout /t 1 /nobreak >nul
goto wh_loop
