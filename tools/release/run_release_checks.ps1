param(
    [switch]$SkipPytest,
    [switch]$SkipDemo
)

$ErrorActionPreference = "Stop"

$python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

$argList = @("tools/release/run_release_checks.py")
if ($SkipPytest) {
    $argList += "--skip-pytest"
}
if ($SkipDemo) {
    $argList += "--skip-demo"
}
$argList += $args

& $python @argList
exit $LASTEXITCODE
