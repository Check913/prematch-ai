Set-Location "C:\Users\check\AppData\Local\HackerAI\prematch-ai"
python -m prematch backtest *>> "C:\Users\check\AppData\Local\HackerAI\prematch-ai\backtest3.log"
Add-Content -Path "C:\Users\check\AppData\Local\HackerAI\prematch-ai\backtest3.log" -Value "EXIT=$LASTEXITCODE"
