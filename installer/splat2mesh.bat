@echo off
rem Starts the splat2mesh app (local server + its window).
set ROOT=%~dp0..
start "" /D "%ROOT%" "%ROOT%\.venv\Scripts\pythonw.exe" -m splat2mesh.app
