cd %~dp0
call service_config.bat
nssm install %SERVICE_NAME% "%cd%\deepfaceengine.exe"
