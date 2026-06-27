// Global State
let selectedSourceType = 'image'; // 'image' or 'video'
let uploadedVideoPath = null;
let logPollingInterval = null;
let vehicleChartInstance = null;
let colorChartInstance = null;

// Tab Switcher
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');

    // Clean up video stream / polling when leaving detect tab
    if (tabId !== 'detect-tab') {
        stopVideoAndPolling();
    }
}

function stopVideoAndPolling() {
    const frame = document.getElementById('renderFrame');
    frame.src = '';
    frame.classList.add('hidden');
    document.getElementById('noFramePlaceholder').classList.remove('hidden');
    
    if (logPollingInterval) {
        clearInterval(logPollingInterval);
        logPollingInterval = null;
    }
}

// Source Input Switcher
function setSourceType(type) {
    selectedSourceType = type;
    stopVideoAndPolling();

    if (type === 'image') {
        document.getElementById('srcImgBtn').classList.add('active');
        document.getElementById('srcVidBtn').classList.remove('active');
        document.getElementById('imageSourceSection').classList.remove('hidden');
        document.getElementById('videoSourceSection').classList.add('hidden');
    } else {
        document.getElementById('srcImgBtn').classList.remove('active');
        document.getElementById('srcVidBtn').classList.add('active');
        document.getElementById('imageSourceSection').classList.add('hidden');
        document.getElementById('videoSourceSection').classList.remove('hidden');
    }
}

// File Upload Labels Listener
document.getElementById('imageInput').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        document.getElementById('imageUploadLabel').innerText = `已选择图片: ${e.target.files[0].name}`;
    }
});

document.getElementById('videoInput').addEventListener('change', async function(e) {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        document.getElementById('videoUploadLabel').innerText = `正在上传视频: ${file.name}...`;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const res = await fetch('/api/upload/video', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                uploadedVideoPath = data.file_path;
                document.getElementById('videoUploadLabel').innerText = `视频已上传并就绪: ${file.name}`;
            } else {
                alert('视频上传失败！');
                document.getElementById('videoUploadLabel').innerText = '点击或拖拽上传测试视频';
            }
        } catch (err) {
            console.error(err);
            alert('视频上传出错！');
            document.getElementById('videoUploadLabel').innerText = '点击或拖拽上传测试视频';
        }
    }
});

// Run Inference Engine
async function runDetection() {
    stopVideoAndPolling();
    
    const model = document.getElementById('modelSelect').value;
    const frame = document.getElementById('renderFrame');
    const placeholder = document.getElementById('noFramePlaceholder');
    const tableBody = document.querySelector('#sessionLogTable tbody');
    
    // Clear session table
    tableBody.innerHTML = '';

    if (selectedSourceType === 'image') {
        const imageInput = document.getElementById('imageInput');
        if (imageInput.files.length === 0) {
            alert('请先选择一张图片！');
            return;
        }

        placeholder.classList.add('hidden');
        frame.classList.remove('hidden');
        frame.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text y="50" font-size="20" fill="gray">正在处理图片...</text></svg>';

        const formData = new FormData();
        formData.append('file', imageInput.files[0]);
        formData.append('model', model);

        try {
            const res = await fetch('/api/detect/image', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (data.status === 'success') {
                frame.src = data.image;
                
                if (data.detections.length === 0) {
                    tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #64748b;">未检测到任何车辆或车牌</td></tr>';
                } else {
                    data.detections.forEach(det => {
                        appendSessionLogRow(det);
                        if (det.is_suspected) {
                            showAlarm(det.plate_str, '黑名单布控车辆');
                        }
                    });
                }
            } else {
                alert('推理处理失败！');
                placeholder.classList.remove('hidden');
                frame.classList.add('hidden');
            }
        } catch (err) {
            console.error(err);
            alert('请求出错！');
            placeholder.classList.remove('hidden');
            frame.classList.add('hidden');
        }

    } else {
        // Video processing
        if (!uploadedVideoPath) {
            alert('请先等待视频文件上传完成！');
            return;
        }

        placeholder.classList.add('hidden');
        frame.classList.remove('hidden');
        
        // Bind video MJPEG stream endpoint
        frame.src = `/api/detect/video?source_path=${encodeURIComponent(uploadedVideoPath)}&model=${model}`;
        
        // Start polling logs from SQLite database
        let lastLoggedId = 0;
        
        // Get initial max log ID
        try {
            const initialRes = await fetch('/api/logs?limit=1');
            const initialLogs = await initialRes.json();
            if (initialLogs.length > 0) {
                lastLoggedId = initialLogs[0].id;
            }
        } catch (err) {
            console.error(err);
        }

        logPollingInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/logs?limit=15');
                const logs = await res.json();
                
                // Filter new logs that occurred after start
                const newLogs = logs.filter(log => log.id > lastLoggedId).reverse();
                newLogs.forEach(log => {
                    lastLoggedId = Math.max(lastLoggedId, log.id);
                    appendSessionLogRow({
                        plate_str: log.plate,
                        vehicle_class: log.vehicle,
                        plate_color: log.color,
                        attribute: 'normal',
                        is_suspected: log.is_suspected
                    });
                    
                    if (log.is_suspected) {
                        showAlarm(log.plate, '在库黑名单布控');
                    }
                });
            } catch (err) {
                console.error(err);
            }
        }, 1500);
    }
}

function appendSessionLogRow(det) {
    const tableBody = document.querySelector('#sessionLogTable tbody');
    const emptyRow = document.getElementById('emptySessionRow');
    if (emptyRow) {
        emptyRow.remove();
    }

    const timeStr = new Date().toLocaleTimeString();
    const tr = document.createElement('tr');
    
    const statusBadge = det.is_suspected 
        ? '<span class="badge badge-alert">🚨 嫌疑车辆</span>' 
        : '<span class="badge badge-normal">🟢 正常通行</span>';

    tr.innerHTML = `
        <td>${timeStr}</td>
        <td>${det.vehicle_class}</td>
        <td style="font-weight: 700; color: #38bdf8;">${det.plate_str}</td>
        <td>${det.plate_color}</td>
        <td>${det.attribute || 'normal'}</td>
        <td>${statusBadge}</td>
    `;
    
    // Insert at beginning of table
    tableBody.insertBefore(tr, tableBody.firstChild);
}

// Alert banner handling
function showAlarm(plate, reason) {
    const banner = document.getElementById('alarmBanner');
    const text = document.getElementById('alarmText');
    text.innerHTML = `发现布控黑名单车辆：<strong>${plate}</strong> | 原因：${reason}`;
    banner.classList.remove('hidden');
    
    // Play alert sound if wanted, or print warning
    console.warn(`ALERT! Suspected vehicle detected: ${plate} - ${reason}`);
}

function closeAlarm() {
    document.getElementById('alarmBanner').classList.add('hidden');
}

// ----------------- TAB 2: HISTORY & ANALYTICS -----------------
async function loadHistory() {
    const search = document.getElementById('searchPlateInput').value;
    const vehicle = document.getElementById('filterVehicleSelect').value;
    
    try {
        const res = await fetch(`/api/logs?plate=${encodeURIComponent(search)}&vehicle=${encodeURIComponent(vehicle)}`);
        const logs = await res.json();
        
        // Render stats indicators
        updateStatsIndicators(logs);
        
        // Draw logs table
        const tbody = document.getElementById('historyTableBody');
        tbody.innerHTML = '';
        
        if (logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #64748b;">没有找到符合条件的记录</td></tr>';
            return;
        }
        
        logs.forEach(log => {
            const tr = document.createElement('tr');
            const suspectedText = log.is_suspected 
                ? '<span class="badge badge-alert">🚨 嫌疑布控</span>' 
                : '<span class="badge badge-normal">🟢 正常</span>';
                
            tr.innerHTML = `
                <td>${log.id}</td>
                <td>${log.date}</td>
                <td>${log.time}</td>
                <td>${log.vehicle}</td>
                <td style="font-weight: 700; color: #38bdf8;">${log.plate}</td>
                <td>${log.color}</td>
                <td>${suspectedText}</td>
            `;
            tbody.appendChild(tr);
        });

        // Redraw statistics charts
        drawCharts(logs);
        
    } catch (err) {
        console.error(err);
        alert('无法加载历史通行记录！');
    }
}

function updateStatsIndicators(logs) {
    const statTotal = document.getElementById('statTotal');
    const statAlerts = document.getElementById('statAlerts');
    const statUnique = document.getElementById('statUnique');
    const statActive = document.getElementById('statActive');
    
    statTotal.innerText = logs.length;
    statAlerts.innerText = logs.filter(l => l.is_suspected).length;
    
    const uniquePlatesSet = new Set(logs.map(l => l.plate));
    statUnique.innerText = uniquePlatesSet.size;
    
    // Calculate most active vehicle
    if (logs.length > 0) {
        const counts = {};
        logs.forEach(l => {
            counts[l.vehicle] = (counts[l.vehicle] || 0) + 1;
        });
        const active = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        statActive.innerText = active.toUpperCase();
    } else {
        statActive.innerText = 'N/A';
    }
}

function drawCharts(logs) {
    // 1. Vehicle chart calculations
    const vehCounts = {};
    logs.forEach(l => {
        vehCounts[l.vehicle] = (vehCounts[l.vehicle] || 0) + 1;
    });
    
    const vehLabels = Object.keys(vehCounts);
    const vehData = Object.values(vehCounts);
    
    // Destroy previous instance
    if (vehicleChartInstance) {
        vehicleChartInstance.destroy();
    }
    
    const ctx1 = document.getElementById('vehicleChart').getContext('2d');
    vehicleChartInstance = new Chart(ctx1, {
        type: 'pie',
        data: {
            labels: vehLabels.map(l => l.toUpperCase()),
            datasets: [{
                data: vehData,
                backgroundColor: [
                    '#38bdf8', '#10b981', '#fbbf24', '#f87171', '#a78bfa', '#ec4899'
                ],
                borderWidth: 1,
                borderColor: '#1e293b'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });

    // 2. Color chart calculations
    const colorCounts = {};
    logs.forEach(l => {
        colorCounts[l.color] = (colorCounts[l.color] || 0) + 1;
    });
    
    const colorLabels = Object.keys(colorCounts);
    const colorData = Object.values(colorCounts);
    
    if (colorChartInstance) {
        colorChartInstance.destroy();
    }
    
    const ctx2 = document.getElementById('colorChart').getContext('2d');
    colorChartInstance = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels: colorLabels.map(l => l.toUpperCase()),
            datasets: [{
                label: '车牌数',
                data: colorData,
                backgroundColor: '#38bdf8',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                y: {
                    ticks: { color: '#94a3b8', stepSize: 1 },
                    grid: { color: '#334155' }
                }
            }
        }
    });
}

// ----------------- TAB 3: BLACKLIST MANAGEMENT -----------------
async function loadBlacklist() {
    try {
        const res = await fetch('/api/blacklist');
        const list = await res.json();
        
        const tbody = document.getElementById('blacklistTableBody');
        tbody.innerHTML = '';
        
        if (list.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: #64748b;">当前布控库中无嫌疑车牌</td></tr>';
            return;
        }
        
        list.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: 700; color: #ef4444;">${row.plate}</td>
                <td>${row.reason || '无'}</td>
                <td>
                    <button class="delete-btn" onclick="deleteBlacklist('${row.plate}')">解除布控</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error(err);
        alert('加载嫌疑车牌库失败！');
    }
}

async function submitBlacklist(event) {
    event.preventDefault();
    const plate = document.getElementById('blackPlate').value;
    const reason = document.getElementById('blackReason').value;
    
    try {
        const res = await fetch('/api/blacklist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plate, reason })
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            alert('添加成功！');
            document.getElementById('addBlacklistForm').reset();
            loadBlacklist();
        } else {
            alert('添加失败: ' + data.detail);
        }
    } catch (err) {
        console.error(err);
        alert('提交出错！');
    }
}

async function deleteBlacklist(plate) {
    if (!confirm(`确定解除对车牌 [${plate}] 的布控吗？`)) {
        return;
    }
    
    try {
        const res = await fetch(`/api/blacklist/${encodeURIComponent(plate)}`, {
            method: 'DELETE'
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            alert('解除成功！');
            loadBlacklist();
        } else {
            alert('删除失败: ' + data.detail);
        }
    } catch (err) {
        console.error(err);
        alert('解控请求出错！');
    }
}
