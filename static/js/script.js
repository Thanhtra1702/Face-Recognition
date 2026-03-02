const appState = {
    status: "SCANNING",
    pollingInterval: null
};

// DOM Map
const ui = {
    cameraSection: document.querySelector('.camera-section'),
    viewScanning: document.getElementById('view-scanning'),
    viewSuccess: document.getElementById('view-success'),

    // Result Fields
    stName: document.getElementById('st-name'),
    stId: document.getElementById('st-id'),
    stClass: document.getElementById('st-class'),
    stTime: document.getElementById('st-time'),

    // Buttons
    btnConfirm: document.getElementById('btn-confirm'),
    btnRetry: document.getElementById('btn-retry'),

    // Notification
    distanceAlert: document.getElementById('distance-alert'),

    // Clock
    clock: document.getElementById('clock'),
    date: document.getElementById('date'),

    // Liveness
    livenessBadge: document.getElementById('liveness-badge'),
    livenessText: document.getElementById('liveness-text'),
    blinkInstruction: document.getElementById('blink-instruction')
};

// 1. Clock Logic (HH:mm:ss)
function updateClock() {
    const now = new Date();
    ui.clock.innerText = now.toLocaleTimeString('vi-VN', { hour12: false });
    const days = ['CHỦ NHẬT', 'THỨ HAI', 'THỨ BA', 'THỨ TƯ', 'THỨ NĂM', 'THỨ SÁU', 'THỨ BẢY'];
    const d = now.getDate().toString().padStart(2, '0');
    const m = (now.getMonth() + 1).toString().padStart(2, '0');
    ui.date.innerText = `${days[now.getDay()]}, ${d}/${m}/${now.getFullYear()}`;
}
setInterval(updateClock, 1000);
updateClock();

// 2. View Switcher
function switchView(viewName) {
    ui.viewScanning.classList.add('hidden');
    ui.viewSuccess.classList.add('hidden');

    if (viewName === 'SCANNING') ui.viewScanning.classList.remove('hidden');
    if (viewName === 'SUCCESS') ui.viewSuccess.classList.remove('hidden');
}

// 3. Actions
// Nút Xác Nhận: Gửi API confirm -> Reset UI
ui.btnConfirm.addEventListener('click', async () => {
    try {
        await fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'confirm' })
        });
        resetToScanning();
    } catch (e) { console.error("Confirm error:", e); }
});

// Nút Thử Lại: Thông báo server reset và Reset UI
ui.btnRetry.addEventListener('click', async () => {
    try {
        await fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'retry' }) // Gửi lệnh reset lên Server
        });
        resetToScanning();
    } catch (e) { console.error("Retry error:", e); }
});

function resetToScanning() {
    appState.status = "SCANNING";
    switchView('SCANNING');
    // Clear old data visual
    ui.stName.innerText = "--";
}

// 4. Polling Logic
function startPolling() {
    setInterval(async () => {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            updateUI(data);
        } catch (e) { console.error("Polling error:", e); }
    }, 100);
}

function updateUI(data) {
    const { status, data: studentData } = data;

    // Lock UI if in Confirm state (Success view is visible)
    if (appState.status === "CONFIRM" && status !== "CONFIRM") return;
    if (appState.status === status && status !== "SCANNING") return;

    appState.status = status;

    // Handle Distance Alert
    if (status === "SCANNING" && data.is_near === false) {
        ui.distanceAlert.classList.remove('hidden');
    } else {
        ui.distanceAlert.classList.add('hidden');
    }

    if (status === "SCANNING") {
        switchView('SCANNING');

        // Dual-Layer Feedback
        if (data.is_near) {
            ui.livenessBadge.style.display = 'flex';

            if (data.is_live) {
                ui.livenessBadge.classList.add('live');
                const method = data.fas_score > 0.99 ? "AI PASSIVE" : "BLINK OK";
                ui.livenessText.innerText = `LIVENESS: ${method}`;
                ui.blinkInstruction.classList.add('hidden');
            } else {
                ui.livenessBadge.classList.remove('live');
                ui.livenessText.innerText = `AI Confidence: ${(data.fas_score * 100).toFixed(1)}%`;

                // Chỉ hiện "Yêu cầu nháy mắt" nếu AI chưa tin tưởng
                if (data.fas_score <= 0.99) {
                    ui.blinkInstruction.classList.remove('hidden');
                } else {
                    ui.blinkInstruction.classList.add('hidden');
                }
            }
        } else {
            ui.livenessBadge.style.display = 'none';
            ui.blinkInstruction.classList.add('hidden');
        }
    }
    else if (status === "CONFIRM") {
        ui.livenessBadge.style.display = 'none';
        ui.blinkInstruction.classList.add('hidden');
        if (studentData) {
            ui.stName.innerText = (studentData.name || "NGƯỜI LẠ").toUpperCase();
            ui.stId.innerText = studentData.student_id || "N/A";

            // Map thêm thông tin Phòng/Lớp
            const schedule = studentData.schedule || "N/A";
            const room = studentData.room || "";
            ui.stClass.innerText = room ? `${schedule} - ${room}` : schedule;

            ui.stTime.innerText = studentData.checkin_time || "--";

            switchView('SUCCESS');
        }
    }
}

startPolling();
console.log("Kiosk Advanced UI v2.5 Initialized");
