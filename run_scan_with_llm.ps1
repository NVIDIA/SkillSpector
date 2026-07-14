param(
    [Parameter(Mandatory = $true)]
    [string]$SkillPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputJson,

    [string]$Mailbox = "C:\temp\skillspector-mailbox"
)

$env:SKILLSPECTOR_PROVIDER       = "subprocess"
$env:SKILLSPECTOR_LLM_COMMAND    = "uv run --no-project python C:\zz\SkillSpector\skillspector_bridge.py"
$env:SKILLSPECTOR_MAILBOX        = $Mailbox
$env:SKILLSPECTOR_BRIDGE_TIMEOUT = "80"

New-Item -ItemType Directory -Force $Mailbox | Out-Null

$proc = Start-Process -FilePath "skillspector" `
    -ArgumentList @("scan", $SkillPath, "--format", "json", "--output", $OutputJson) `
    -NoNewWindow -PassThru `
    -Environment @{
        SKILLSPECTOR_PROVIDER       = "subprocess"
        SKILLSPECTOR_LLM_COMMAND    = "uv run --no-project python C:\zz\SkillSpector\skillspector_bridge.py"
        SKILLSPECTOR_MAILBOX        = $Mailbox
        SKILLSPECTOR_BRIDGE_TIMEOUT = "80"
        PATH                        = $env:PATH
    }

Write-Host "Scan started (PID $($proc.Id)). Output -> $OutputJson"
Write-Host "Monitoring mailbox: $Mailbox"
Write-Host "---"
Write-Host "When PENDING lines appear, read the .req file and write a .resp file within 80s."
Write-Host "---"

$reported = @{}

while (-not $proc.HasExited) {
    $reqs = Get-ChildItem $Mailbox -Filter "*.req" -ErrorAction SilentlyContinue
    foreach ($req in $reqs) {
        $respPath = $req.FullName -replace '\.req$', '.resp'
        if (-not (Test-Path $respPath) -and -not $reported.ContainsKey($req.Name)) {
            $reported[$req.Name] = $true
            Write-Host "PENDING: $($req.Name)  ($([math]::Round($req.Length / 1KB, 1)) KB)"
        }
    }
    Start-Sleep -Seconds 2
}

# Drain any final requests that arrived just before exit
Start-Sleep -Milliseconds 500
$remaining = Get-ChildItem $Mailbox -Filter "*.req" -ErrorAction SilentlyContinue |
    Where-Object { -not (Test-Path ($_.FullName -replace '\.req$', '.resp')) }
foreach ($req in $remaining) {
    if (-not $reported.ContainsKey($req.Name)) {
        Write-Host "PENDING (post-exit): $($req.Name)  ($([math]::Round($req.Length / 1KB, 1)) KB)"
    }
}

Write-Host "---"
Write-Host "Scan complete (exit code $($proc.ExitCode)). Results: $OutputJson"
