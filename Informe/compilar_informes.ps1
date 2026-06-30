$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$auxDirName = "auxiliares_latex"
$auxDir = Join-Path $PSScriptRoot $auxDirName
New-Item -ItemType Directory -Path $auxDir -Force | Out-Null

$reports = @("Informe", "Informe_final")

foreach ($report in $reports) {
    $texFile = "$report.tex"
    if (-not (Test-Path -LiteralPath $texFile)) {
        Write-Host "Saltando ${texFile}: no existe."
        continue
    }

    Write-Host "Compilando $texFile..."
    & pdflatex -interaction=nonstopmode "-aux-directory=$auxDirName" $texFile
    & bibtex (Join-Path $auxDirName $report)
    & pdflatex -interaction=nonstopmode "-aux-directory=$auxDirName" $texFile
    & pdflatex -interaction=nonstopmode "-aux-directory=$auxDirName" $texFile
}

Write-Host "Listo. PDFs en Informe/ y auxiliares en Informe/$auxDirName/."
