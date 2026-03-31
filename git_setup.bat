@echo off
setlocal
cd /d "c:\Users\milan\PROJECTS\Project ATLAS"

echo ============================================
echo  PROJECT ATLAS — Git Push to GitHub
echo ============================================
echo.

REM Check if git repo already initialized
if not exist ".git" (
    echo [1/5]  Initializing git repo...
    git init
    git branch -M main
) else (
    echo [1/5]  Git repo already initialized. OK.
)

echo.
echo [2/5]  Setting remote origin...
git remote remove origin 2>nul
git remote add origin https://github.com/Iammilansoni/Project-Atlas.git
git remote -v

echo.
echo [3/5]  Staging all files...
git add -A
echo       Files staged:
git diff --cached --stat

echo.
echo [4/5]  Committing...
git -c user.email="deploy@project-atlas.local" -c user.name="Project ATLAS" ^
    commit -m "feat: production-ready deploy for Render (FAISS + data + config)"

echo.
echo [5/5]  Pushing to GitHub (browser login may pop up)...
git push -u origin main

echo.
echo ============================================
echo  DONE — Check https://github.com/Iammilansoni/Project-Atlas
echo ============================================
pause
