param(
    [Parameter(Mandatory = $true)]
    [string] $PythonPath,

    [Parameter(Mandatory = $true)]
    [string] $Marker
)

& $PythonPath -m pytest -m $Marker tests
exit $LASTEXITCODE
