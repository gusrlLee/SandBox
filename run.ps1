# 에러가 발생하면 즉시 중단하도록 설정
$ErrorActionPreference = "Stop"

# 1. 빌드 수행 (Debug 모드 명시)
Write-Host "🔨 Building Project..." -ForegroundColor Cyan
cmake --build build --config Debug --parallel

# 2. 빌드 성공 여부 확인 ($LASTEXITCODE가 0이 아니면 실패)
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build Failed! Aborting execution." -ForegroundColor Red
    exit 1
}

# 3. 실행
Write-Host "🚀 Running Sandbox..." -ForegroundColor Green

./bin/Debug/SandBox.exe