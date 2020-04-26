cd %~dp0
call service_config.bat

call service_stop.bat
if %ERRORLEVEL% NEQ 0 goto :exit

nssm remove %SERVICE_NAME% 
@echo off
if %ERRORLEVEL% EQU 0 goto :exit
if %ERRORLEVEL% NEQ 3 goto :error
echo ---------------------------------
echo    Please Run as administrator
echo ---------------------------------
pause
goto :exit
:error
echo ERRORLEVEL %ERRORLEVEL%
echo .
pause
:exit
