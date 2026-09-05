param(
    [string]$OmcPath = ""
)

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$createdDrive = $false

if ([string]::IsNullOrWhiteSpace($OmcPath)) {
    if ($env:MODELICA_OMC -and (Test-Path -LiteralPath $env:MODELICA_OMC)) { $OmcPath = $env:MODELICA_OMC }
    elseif ($env:OPENMODELICAHOME -and (Test-Path -LiteralPath (Join-Path $env:OPENMODELICAHOME "bin/omc.exe"))) { $OmcPath = Join-Path $env:OPENMODELICAHOME "bin/omc.exe" }
}

if (-not (Test-Path -LiteralPath $OmcPath)) {
    $command = Get-Command omc -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "OpenModelica compiler omc was not found. Pass -OmcPath explicitly."
    }
    $OmcPath = $command.Source
}

try {
    $existingDrive = Get-PSDrive -Name M -ErrorAction SilentlyContinue
    if ($null -ne $existingDrive) {
        throw "Drive M: is already in use. Remove that mapping or edit this script to use another drive letter."
    }
    subst M: $projectPath
    $createdDrive = $true

    $env:OPENMODELICALIBRARY = "M:/modelica_runtime/omlibrary"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:OMP_NUM_THREADS = "1"
    $resultPath = Join-Path $projectPath "outputs\modelica\PrecisionCabinetCooling_res.csv"
    $runStartedUtc = [DateTime]::UtcNow
    $omcOutput = (& $OmcPath --numProcs=1 -d=-parallelCodegen "M:/modelica/run_reference.mos" 2>&1 | Out-String)
    $omcOutput | Write-Host
    if ($LASTEXITCODE -ne 0) {
        throw "OpenModelica returned exit code $LASTEXITCODE."
    }
    if ($omcOutput -notmatch "The simulation finished successfully") {
        throw "OpenModelica did not report a successful simulation. Existing result files will not be reused."
    }
    if (-not (Test-Path -LiteralPath $resultPath)) {
        throw "OpenModelica did not create the expected result CSV."
    }
    if ((Get-Item -LiteralPath $resultPath).LastWriteTimeUtc -lt $runStartedUtc) {
        throw "The result CSV was not refreshed by this run; refusing to validate stale data."
    }

    Push-Location $projectPath
    try {
        python run_cross_validation.py outputs
        python render_report.py outputs
    } finally {
        Pop-Location
    }
} finally {
    if ($createdDrive) {
        subst M: /d
    }
}
