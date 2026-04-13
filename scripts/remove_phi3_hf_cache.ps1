# 删除 Hugging Face 缓存中的 microsoft/Phi-3-mini（释放 C 盘）
# 用法: powershell -ExecutionPolicy Bypass -File scripts/remove_phi3_hf_cache.ps1

$ErrorActionPreference = "SilentlyContinue"
$roots = @(
    (Join-Path $env:USERPROFILE ".cache\huggingface\hub"),
    (Join-Path $env:USERPROFILE ".cache\huggingface\transformers"),
    $env:HF_HOME,
    $env:TRANSFORMERS_CACHE
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

foreach ($r in $roots) {
    Get-ChildItem -Path $r -Filter "models--microsoft--Phi-3*" -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            Write-Host "Removing $($_.FullName)"
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
}
Write-Host "Done."
