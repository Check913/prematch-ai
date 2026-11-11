# Installe une tache planifiee Windows : l'agent demarre au boot de la session.
# Usage : powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$bat = Join-Path $root "scripts\start.bat"

$action = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "PrematchAI" -Action $action `
    -Trigger $trigger -Settings $settings -Force
Write-Host "Tache 'PrematchAI' installee : demarrage automatique a l'ouverture de session."
Write-Host "Pour supprimer : Unregister-ScheduledTask -TaskName PrematchAI -Confirm:`$false"
