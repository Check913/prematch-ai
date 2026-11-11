# Installe une tache planifiee Windows : l'agent demarre au boot de la session.
# Usage : powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$bat  = Join-Path $root "scripts\start.bat"

$action   = New-ScheduledTaskAction -Execute "cmd.exe" `
              -Argument "/c `"$bat`"" `
              -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
              -DontStopIfGoingOnBatteries -RestartCount 3 `
              -RestartInterval (New-TimeSpan -Minutes 5) `
              -ExecutionTimeLimit (New-TimeSpan -Days 3650)

Register-ScheduledTask -TaskName "PrematchAI" -Action $action `
    -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Tache 'PrematchAI' enregistree : demarrage auto a l'ouverture de session."
Write-Host "Pour supprimer : Unregister-ScheduledTask -TaskName PrematchAI -Confirm:`$false"
