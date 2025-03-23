@echo off
setlocal

:: Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed. Please install Miniconda or Anaconda first.
    :: Set Anaconda download URL
    set "ANACONDA_URL=https://repo.anaconda.com/archive/Anaconda3-2023.09-1-Windows-x86_64.exe"
    set "INSTALLER=Anaconda3.exe"
    set "INSTALL_DIR=%USERPROFILE%\Anaconda3"

    :: Check if Conda is already installed
    where conda >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo Anaconda/Miniconda is already installed. Do you want to reinstall? (Y/N)
        set /p REINSTALL=
        if /I "%REINSTALL%" neq "Y" (
            echo Exiting installation.
            exit /b 0
        )
    )

    :: Ask user if they want to install Anaconda
    echo Do you want to install Anaconda? (Y/N)
    set /p INSTALL=
    if /I "%INSTALL%" neq "Y" (
        echo Exiting installation.
        exit /b 0
    )

    :: Download Anaconda
    echo Downloading Anaconda...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%ANACONDA_URL%', '%INSTALLER%')"

    :: Install Anaconda (silent install)
    echo Installing Anaconda...
    start /wait %INSTALLER% /InstallationType=JustMe /RegisterPython=1 /S /D=%INSTALL_DIR%

    :: Configure environment variables
    setx PATH "%INSTALL_DIR%;%INSTALL_DIR%\Scripts;%INSTALL_DIR%\Library\bin;%PATH%"

    :: Clean up installer
    del %INSTALLER%

    echo Anaconda installation complete. Please reopen the command prompt to apply changes.
    exit /b 1
)

:: Create conda environment
echo Creating conda environment...
conda env create -f environment.yaml

:: Prompt to activate environment
for /f "tokens=2" %%a in ('findstr /r "^name:" environment.yaml') do set ENV_NAME=%%a
echo To activate the environment, run: conda activate %ENV_NAME%

endlocal
