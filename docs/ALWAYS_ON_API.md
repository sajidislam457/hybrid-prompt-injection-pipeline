# Always-on Python API (Windows)

The detector API (`run_api.py` on port **8000**) should stay up so neither the public chat nor the admin console needs a manual start every session.

## Recommended: Windows service (NSSM)

1. Download [NSSM](https://nssm.cc/download) and put `nssm.exe` on your PATH (or use a full path).
2. From an **Administrator** PowerShell in the project root:

```powershell
.\scripts\install_api_service.ps1
```

3. Confirm:

```powershell
Get-Service SecureAI-API
Invoke-WebRequest http://127.0.0.1:8000/health
```

4. Service options used by the script:
   - Start type: Automatic
   - Restart on failure
   - Working directory = repo root
   - Prefer `.venv\Scripts\python.exe` when present

### Uninstall

```powershell
.\scripts\install_api_service.ps1 -Uninstall
```

## Alternative: Task Scheduler

Create a task:
- Trigger: **At startup**
- Action: start `python run_api.py` (or `.venv\Scripts\python.exe run_api.py`)
- Start in: project root
- Settings: restart on failure every 1 minute

## Dev convenience

`Start.bat` still starts API + public web for interactive demos.  
`StartAdmin.bat` starts the **admin** console on `http://127.0.0.1:3002` (expects API already healthy).

## Security note

Keep the Python API on localhost when possible. Admin routes require `X-Admin-Token` even if the port is reachable.
