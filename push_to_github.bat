@echo off
echo ==========================================
echo DNSAGuard GitHub Setup Script
echo ==========================================

REM Define path to GH CLI
set "GH_PATH=C:\Program Files\GitHub CLI\gh.exe"

REM Check if GH CLI exists at known path, otherwise try PATH
if not exist "%GH_PATH%" (
    where gh >nul 2>&1
    if %errorlevel% equ 0 (
        set "GH_PATH=gh"
    ) else (
        echo Error: GitHub CLI (gh) not found. Please restart your terminal or reinstall.
        pause
        exit /b
    )
)

echo Using GitHub CLI at: "%GH_PATH%"

REM Check Authentication
"%GH_PATH%" auth status >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo You are not logged in to GitHub.
    echo Logging in now...
    "%GH_PATH%" auth login -p https -w
)

REM Create Repo and Push
echo.
echo Creating GitHub Repository 'DNSAGuard'...
REM Try to create repo. If it fails (e.g. already exists), we continue.
"%GH_PATH%" repo create DNSAGuard --public --source=. --remote=origin --push

if %errorlevel% neq 0 (
    echo.
    echo Error creating/pushing repository. It might already exist.
    echo Trying to push to existing remote...
    git push -u origin main
)

echo.
echo Done!
pause
