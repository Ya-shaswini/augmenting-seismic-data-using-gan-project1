# Test USGS Integration - PowerShell Script

Write-Host "=== Testing USGS Earthquake Integration ===" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check if backend is running
Write-Host "Test 1: Checking backend status..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/" -Method GET
    $result = $response.Content | ConvertFrom-Json
    Write-Host "✓ Backend is running: $($result.message)" -ForegroundColor Green
} catch {
    Write-Host "✗ Backend is not responding!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 2: Check monitor status
Write-Host "Test 2: Checking earthquake monitor status..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/data-sources/monitor/status" -Method GET
    $result = $response.Content | ConvertFrom-Json
    Write-Host "✓ Monitor Status:" -ForegroundColor Green
    Write-Host "  - Running: $($result.monitor.is_running)" -ForegroundColor White
    Write-Host "  - Check Interval: $($result.monitor.check_interval_seconds)s" -ForegroundColor White
    Write-Host "  - Min Magnitude: $($result.monitor.min_magnitude)" -ForegroundColor White
    Write-Host "  - Processed Events: $($result.monitor.processed_events_count)" -ForegroundColor White
} catch {
    Write-Host "✗ Failed to get monitor status" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host ""

# Test 3: Fetch recent earthquakes
Write-Host "Test 3: Fetching recent earthquakes (mag 4.0+, last 24h)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/data-sources/usgs/recent?min_magnitude=4.0&hours_back=24" -Method GET
    $result = $response.Content | ConvertFrom-Json
    Write-Host "✓ Found $($result.count) earthquake(s)" -ForegroundColor Green
    
    if ($result.count -gt 0) {
        Write-Host ""
        Write-Host "Recent Earthquakes:" -ForegroundColor Cyan
        foreach ($event in $result.events | Select-Object -First 5) {
            Write-Host "  • M$($event.magnitude) - $($event.location)" -ForegroundColor White
            Write-Host "    Time: $($event.time_formatted)" -ForegroundColor Gray
            Write-Host "    Depth: $($event.depth_km) km" -ForegroundColor Gray
            Write-Host ""
        }
    } else {
        Write-Host "  No earthquakes found in the last 24 hours (mag 4.0+)" -ForegroundColor Yellow
        Write-Host "  This is normal - try lowering the magnitude threshold" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Failed to fetch earthquakes" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host ""

# Test 4: Trigger manual check
Write-Host "Test 4: Triggering manual earthquake check..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/data-sources/monitor/check-now" -Method POST
    $result = $response.Content | ConvertFrom-Json
    Write-Host "✓ $($result.message)" -ForegroundColor Green
    Write-Host "  Check backend logs for results" -ForegroundColor Gray
} catch {
    Write-Host "✗ Failed to trigger manual check" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Testing Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Open http://localhost:3000 in your browser" -ForegroundColor White
Write-Host "2. Go to 'Real-Time Monitoring' tab" -ForegroundColor White
Write-Host "3. Watch for earthquake events (they appear automatically)" -ForegroundColor White
Write-Host "4. Check browser console (F12) for detailed event data" -ForegroundColor White
