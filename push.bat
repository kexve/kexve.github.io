@echo off
git add --all
set /p message=message:
git commit -m %message%
git push origin master