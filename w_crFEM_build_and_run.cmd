@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0w_crFEM_build_and_run.ps1"
exit /b %ERRORLEVEL%
