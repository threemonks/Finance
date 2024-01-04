@ECHO OFF
REM add relative python library path
setlocal
SET DIR=%~dp0

SET BASEDIR=%DIR:\bin\=%

SET PYLIBDIR=%BASEDIR%\sellertool
ECHO "Python Library Directory: %PYLIBDIR%"

REM activate virtual environment
CALL %BASEDIR%\venv\Scripts\activate.bat

SET PYTHONPATH=%PYLIBDIR%;%PYTHONPATH%
ECHO "Python Path: %PYTHONPATH%"

REM launch the python script via real python interpretter
ECHO "Executing Python: %BASEDIR%\venv\Scripts\python.exe %*"
%BASEDIR%\venv\Scripts\python.exe %*

if ERRORLEVEL 1 (
    ECHO "Python script failed!"
) else (
    ECHO "Python script succeeded!"
)

endlocal
