# Upload inference-only weights to a GitHub Release.
# Requires: GitHub personal access token with repo scope.
#
# Usage:
#   $env:GITHUB_TOKEN = "ghp_xxx"
#   powershell -ExecutionPolicy Bypass -File scripts/upload_pretrained_weights_release.ps1 -WeightsDir "D:\path\to\checkpoint\weights"

param(
    [string]$Repo = "123qiang06/BrepARG",
    [string]$Tag = "v1.0-pretrained-weights",
    [string]$ReleaseName = "Pretrained Weights v1.0",
    [string]$WeightsDir = "checkpoint/weights"
)

$ErrorActionPreference = "Stop"

if (-not $env:GITHUB_TOKEN) {
    throw "Set GITHUB_TOKEN environment variable first."
}

$headers = @{
    Authorization = "Bearer $env:GITHUB_TOKEN"
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

$releaseBody = @"
Inference-only pretrained weights for BrepARG.

Each checkpoint contains only ``model_state_dict`` (no optimizer / training metadata).

| File | Description |
|------|-------------|
| abc_ar.pt | AR model for ABC |
| abc_vqvae.pt | SE VQ-VAE for ABC (codebook=8192) |
| deepcad_ar.pt | AR model for DeepCAD |
| deepcad_vqvae.pt | SE VQ-VAE for DeepCAD (codebook=4096) |

See README for download and usage instructions.
"@

function Get-OrCreateRelease {
    $existing = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/tags/$Tag" -Headers $headers -Method Get -ErrorAction SilentlyContinue
    if ($existing.id) {
        Write-Host "Release already exists: $($existing.html_url)"
        return $existing
    }

    $payload = @{
        tag_name = $Tag
        name = $ReleaseName
        body = $releaseBody
        draft = $false
        prerelease = $false
    } | ConvertTo-Json

    $created = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases" -Headers $headers -Method Post -Body $payload -ContentType "application/json; charset=utf-8"
    Write-Host "Created release: $($created.html_url)"
    return $created
}

function Upload-Asset($ReleaseId, $FilePath) {
    $fileName = [IO.Path]::GetFileName($FilePath)
    $uploadHeaders = @{
        Authorization = "Bearer $env:GITHUB_TOKEN"
        Accept = "application/vnd.github+json"
        "Content-Type" = "application/octet-stream"
    }
    $uri = "https://uploads.github.com/repos/$Repo/releases/$ReleaseId/assets?name=$fileName"
    Invoke-RestMethod -Uri $uri -Headers $uploadHeaders -Method Post -InFile $FilePath | Out-Null
    Write-Host "Uploaded: $fileName"
}

$weightsPath = Resolve-Path $WeightsDir
$files = @("abc_ar.pt", "abc_vqvae.pt", "deepcad_ar.pt", "deepcad_vqvae.pt")
foreach ($name in $files) {
    $path = Join-Path $weightsPath $name
    if (-not (Test-Path $path)) {
        throw "Missing weight file: $path"
    }
}

$release = Get-OrCreateRelease
foreach ($name in $files) {
    Upload-Asset -ReleaseId $release.id -FilePath (Join-Path $weightsPath $name)
}

Write-Host "Done. Release URL: $($release.html_url)"
