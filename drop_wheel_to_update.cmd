ECHO Upgrading or Installing %~1
%~dp0\Python3114\python.exe -m pip install %1 --upgrade --no-dependencies
pause