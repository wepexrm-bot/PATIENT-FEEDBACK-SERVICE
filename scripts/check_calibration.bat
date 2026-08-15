@echo off
setlocal
cd /d "%~dp0.."
echo ==========================================================
echo  MODEL CALIBRATION GATE
echo  Scores the served sentiment model against the UCI
echo  Drugs.com test split - 53,766 reviews.
echo  Run this AFTER every retrain / re-convert and BEFORE
echo  rebuilding the sentiment-model image.
echo ==========================================================
echo.
python scripts\evaluate_calibration.py --ece-threshold 0.05
if %ERRORLEVEL% NEQ 0 goto fail
echo.
echo  *** PASS: model is calibrated, ECE within threshold. Safe to rebuild. ***
echo.
exit /b 0

:fail
echo.
echo  *** FAIL: model is NOT calibrated enough. Do NOT rebuild or deploy. ***
echo.
exit /b 1