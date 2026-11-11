Set-Location "C:\Users\check\AppData\Local\HackerAI\prematch-ai"
python scripts\eval_vs_market.py *>> "C:\Users\check\AppData\Local\HackerAI\prematch-ai\evalm3.log"
Add-Content -Path "C:\Users\check\AppData\Local\HackerAI\prematch-ai\evalm3.log" -Value "EXIT=$LASTEXITCODE"
