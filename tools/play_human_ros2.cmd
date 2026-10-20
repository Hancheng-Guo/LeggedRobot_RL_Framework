@echo off
setlocal

set "PYTHON_EXE=%~1"
for %%I in ("%PYTHON_EXE%") do set "CONDA_PREFIX=%%~dpI"
if "%CONDA_PREFIX:~-1%"=="\" set "CONDA_PREFIX=%CONDA_PREFIX:~0,-1%"

set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library\mingw-w64\bin;%CONDA_PREFIX%\Library\usr\bin;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts;%PATH%"
if not exist "%CONDA_PREFIX%\etc\conda\activate.d\ros2-ros-workspace_activate.bat" (
    echo ROS 2 is not installed in %CONDA_PREFIX%. 1>&2
    exit /b 1
)
call "%CONDA_PREFIX%\etc\conda\activate.d\ros2-ros-workspace_activate.bat"
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m tools.play_human_ros2 --playback-config "%~2" "%~3"
exit /b %errorlevel%
