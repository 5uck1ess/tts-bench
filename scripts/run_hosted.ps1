# Runs scripts/collect_hosted.py with the Fish API key decrypted for that one process.
#
# One-time setup, typed by Tym in his OWN PowerShell window (not in Claude chat, so the key never
# enters a transcript). Input is masked; the file is DPAPI-encrypted to his Windows account:
#   Read-Host "Fish API key" -AsSecureString | ConvertFrom-SecureString | Set-Content "$HOME\.fish_api_key.dpapi"
#
# Usage: .\scripts\run_hosted.ps1 --probe
$ErrorActionPreference = "Stop"
$keyFile = Join-Path $HOME ".fish_api_key.dpapi"
if (-not (Test-Path $keyFile)) { throw "No key file at $keyFile; see the setup line at the top of this script." }
$secure = Get-Content $keyFile | ConvertTo-SecureString
$env:FISH_API_KEY = [System.Net.NetworkCredential]::new("", $secure).Password
try {
    & "$PSScriptRoot\..\venvs\hosted\Scripts\python.exe" "$PSScriptRoot\collect_hosted.py" @args
} finally {
    Remove-Item Env:FISH_API_KEY -ErrorAction SilentlyContinue
}
