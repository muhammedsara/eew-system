// Hybrid Mobile-IoT EEW Dashboard Logic v2.0
// Professional Academic Dashboard

const socket = io();

// Map Initialization
const map = L.map('map', {
    zoomControl: false,
    attributionControl: false
}).setView([40.7, 30.5], 9);

L.control.zoom({ position: 'bottomright' }).addTo(map);

// Dark Theme Map Tiles
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19,
    subdomains: 'abcd'
}).addTo(map);

// Layers
const mobileLayer = L.layerGroup().addTo(map);
const iotLayer = L.layerGroup().addTo(map);
const eventLayer = L.layerGroup().addTo(map);

// Icons
const createIcon = (color, size, glow = false) => L.divIcon({
    className: 'custom-div-icon',
    html: `<div style="background-color:${color}; width:${size}px; height:${size}px; border-radius:50%; ${glow ? `box-shadow:0 0 10px ${color};` : ''}"></div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2]
});

// UI Elements
const els = {
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    mobileSlider: document.getElementById('mobile-weight'),
    iotSlider: document.getElementById('iot-weight'),
    thresholdSlider: document.getElementById('threshold'),
    mobileVal: document.getElementById('mobile-weight-val'),
    iotVal: document.getElementById('iot-weight-val'),
    thresholdVal: document.getElementById('threshold-val'),
    demoMagSlider: document.getElementById('demo-magnitude'),
    demoMagVal: document.getElementById('demo-mag-val'),
    deviceCount: document.getElementById('device-count'),
    eventList: document.getElementById('events-list'),

    // Demo device counts
    demoMobileSlider: document.getElementById('demo-mobile-slider'),
    demoMobileCount: document.getElementById('demo-mobile-count'),
    demoIotSlider: document.getElementById('demo-iot-slider'),
    demoIotCount: document.getElementById('demo-iot-count'),

    // Demo Result Metrics
    metricScore: document.getElementById('metric-score'),
    metricThreshold: document.getElementById('metric-threshold'),
    metricMobile: document.getElementById('metric-mobile'),
    metricIot: document.getElementById('metric-iot'),
    demoDecision: document.getElementById('demo-decision'),

    // Buttons
    btnDemo: document.getElementById('btn-demo'),
    btnSim: document.getElementById('btn-simulate'),

    // Model info
    modelName: document.getElementById('model-name'),
    modelSize: document.getElementById('model-size'),
    sysModel: document.getElementById('sys-model')
};

// --- Load System Info ---
fetch('/api/system-info')
    .then(r => r.json())
    .then(info => {
        if (els.modelName) els.modelName.textContent = info.model.name;
        if (els.modelSize) els.modelSize.textContent = info.model.size_kb + ' KB TFLite';
        if (els.sysModel) els.sysModel.textContent = info.model.name.replace('Earthquake Detector ', '');
        console.log('System info loaded:', info);
    })
    .catch(e => console.error('Failed to load system info:', e));

// --- Socket Handlers ---

socket.on('connect', () => {
    console.log('Connected to Edge Gateway');
    updateConnectionStatus(true);
});

socket.on('disconnect', () => {
    updateConnectionStatus(false);
});

socket.on('status_update', (data) => {
    setStatus(data.mode, data.running);
});

socket.on('devices_update', (devices) => {
    renderDevices(devices);
});

socket.on('config_update', (config) => {
    els.mobileSlider.value = config.mobile_weight;
    els.iotSlider.value = config.iot_weight;
    updateWeightLabels();
});

socket.on('new_event', (event) => {
    addEventLog(event);
    if (event.type === 'earthquake' && event.detected) {
        showEarthquakeOnMap(event);
        setStatus('alert', true);
        setTimeout(() => setStatus('monitoring', true), 5000);
    }
});

socket.on('simulation_complete', (summary) => {
    updateMetrics(summary.metrics);
    if (els.btnSim) {
        els.btnSim.disabled = false;
        els.btnSim.innerHTML = '🚀 Run Batch Simulation';
    }
    setStatus('monitoring', true);

    // Show results
    const resultsContent = document.querySelector('.results-content');
    const placeholder = document.querySelector('.results-placeholder');
    if (resultsContent && placeholder) {
        placeholder.style.display = 'none';
        resultsContent.style.display = 'block';

        document.getElementById('eval-tpr').textContent = (summary.metrics.recall_tpr * 100).toFixed(0) + '%';
        document.getElementById('eval-fpr').textContent = (summary.metrics.fpr * 100).toFixed(1) + '%';
        document.getElementById('eval-precision').textContent = (summary.metrics.precision * 100).toFixed(0) + '%';
        document.getElementById('eval-f1').textContent = summary.metrics.f1_score.toFixed(2);

        // Confusion matrix
        if (summary.metrics.confusion) {
            document.getElementById('cm-tp').textContent = summary.metrics.confusion.tp;
            document.getElementById('cm-fn').textContent = summary.metrics.confusion.fn;
            document.getElementById('cm-fp').textContent = summary.metrics.confusion.fp;
            document.getElementById('cm-tn').textContent = summary.metrics.confusion.tn;
        }
    }
});

// --- UI Actions ---

// Weight Sliders
if (els.mobileSlider) {
    els.mobileSlider.addEventListener('input', (e) => {
        const mobileW = parseFloat(e.target.value);
        const iotW = (1.0 - mobileW).toFixed(1);

        els.iotSlider.value = iotW;
        updateWeightLabels();
        emitConfigUpdate(mobileW, parseFloat(iotW));
    });
}

if (els.iotSlider) {
    els.iotSlider.addEventListener('input', (e) => {
        const iotW = parseFloat(e.target.value);
        const mobileW = (1.0 - iotW).toFixed(1);

        els.mobileSlider.value = mobileW;
        updateWeightLabels();
        emitConfigUpdate(parseFloat(mobileW), iotW);
    });
}

if (els.thresholdSlider) {
    els.thresholdSlider.addEventListener('input', (e) => {
        els.thresholdVal.textContent = e.target.value;
    });
}

if (els.demoMagSlider) {
    els.demoMagSlider.addEventListener('input', (e) => {
        els.demoMagVal.textContent = e.target.value;
    });
}

// Demo device count sliders
if (els.demoMobileSlider) {
    els.demoMobileSlider.addEventListener('input', (e) => {
        els.demoMobileCount.textContent = e.target.value;
    });
}

if (els.demoIotSlider) {
    els.demoIotSlider.addEventListener('input', (e) => {
        els.demoIotCount.textContent = e.target.value;
    });
}

function updateWeightLabels() {
    if (els.mobileVal) els.mobileVal.textContent = els.mobileSlider.value;
    if (els.iotVal) els.iotVal.textContent = els.iotSlider.value;
}

function emitConfigUpdate(m, i) {
    socket.emit('update_config', { mobile_weight: m, iot_weight: i });
}

// Update demo result display
function updateDemoResult(decision, event) {
    // Use slider threshold, not backend (for display consistency)
    const displayThreshold = els.thresholdSlider ? parseFloat(els.thresholdSlider.value) : decision.threshold;

    if (els.metricScore) els.metricScore.textContent = decision.score.toFixed(2);
    if (els.metricThreshold) els.metricThreshold.textContent = displayThreshold.toFixed(2);
    if (els.metricMobile) els.metricMobile.textContent = decision.mobile_count;
    if (els.metricIot) els.metricIot.textContent = decision.iot_count;

    if (els.demoDecision) {
        if (decision.is_earthquake) {
            els.demoDecision.innerHTML = `<span style="color: var(--accent-danger); font-weight: 600;">[!] EARTHQUAKE DETECTED (M${event.magnitude})</span>`;
            els.demoDecision.style.background = 'rgba(244, 63, 94, 0.1)';
        } else {
            els.demoDecision.innerHTML = `<span style="color: var(--accent-success);">[OK] No Earthquake (Score below threshold)</span>`;
            els.demoDecision.style.background = 'rgba(52, 211, 153, 0.1)';
        }
    }
}

// Trigger Demo with Step-by-Step Animation
if (els.btnDemo) {
    els.btnDemo.addEventListener('click', async () => {
        console.log("Demo button clicked - Starting animated simulation");
        const mag = els.demoMagSlider ? parseFloat(els.demoMagSlider.value) : 5.8;
        const mobileCount = els.demoMobileSlider ? parseInt(els.demoMobileSlider.value) : 200;
        const iotCount = els.demoIotSlider ? parseInt(els.demoIotSlider.value) : 10;
        const threshold = els.thresholdSlider ? parseFloat(els.thresholdSlider.value) : 0.85;

        els.btnDemo.disabled = true;
        els.btnDemo.innerHTML = '⏳ Simulating...';

        // Reset simulation panels
        resetSimulationPanels();
        updateSimStatus('RUNNING', '#fbbf24');

        try {
            // Generate epicenter for this simulation
            const epicenterLat = 40.7 + (Math.random() - 0.5) * 0.5;
            const epicenterLon = 30.5 + (Math.random() - 0.5) * 0.5;

            // Step 1: Earthquake triggered (t=0)
            await animateStep(1, 500);

            // Step 2: P-wave propagating (t=0.5s) - Show wave on map!
            await animateStep(2, 300);
            showWaveAnimation(epicenterLat, epicenterLon, mag);
            await new Promise(r => setTimeout(r, 200));

            // Show simulated devices around epicenter
            renderSimulatedDevices(epicenterLat, epicenterLon, mobileCount, iotCount);

            // Step 3: Devices detecting (t=1.0s)
            await animateStep(3, 500);
            updateDeviceStates(mobileCount, iotCount, 'detecting');
            highlightTriggeredDevices();

            // Step 4: Edge AI inference (t=1.5s)
            await animateStep(4, 400);
            updateDeviceStates(mobileCount, iotCount, 'processing');

            // Step 5: Consensus voting (t=2.0s)
            await animateStep(5, 400);

            // Make actual API call
            const res = await fetch('/api/demo', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    magnitude: mag,
                    lat: epicenterLat,
                    lon: epicenterLon,
                    num_mobile: mobileCount,
                    num_iot: iotCount,
                    threshold: threshold
                })
            });
            const json = await res.json();
            console.log("Demo response:", json);

            // Update consensus engine visualization
            if (json.success && json.decision) {
                await updateConsensusSteps(json.decision, mobileCount, iotCount, threshold);

                // Step 6: Decision
                await animateStep(6, 300);
                const step6Text = document.getElementById('step-6-text');
                if (step6Text) {
                    step6Text.textContent = json.decision.is_earthquake
                        ? '[+] EARTHQUAKE DETECTED'
                        : '[-] No earthquake (below threshold)';
                }

                // Update demo result display
                updateDemoResult(json.decision, json.event);
                updateDeviceStates(mobileCount, iotCount, 'complete', json.decision);
                updateSimStatus(json.decision.is_earthquake ? 'DETECTED' : 'NO EARTHQUAKE',
                    json.decision.is_earthquake ? '#f43f5e' : '#34d399');
            }
        } catch (e) {
            console.error("Demo failed:", e);
            updateSimStatus('ERROR', '#f43f5e');
        } finally {
            setTimeout(() => {
                els.btnDemo.disabled = false;
                els.btnDemo.innerHTML = '⚡ Trigger Earthquake';
            }, 1000);
        }
    });
}

// Simulation Animation Functions
function resetSimulationPanels() {
    // Reset timeline steps
    for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`step-${i}`);
        if (step) {
            step.style.opacity = '0.4';
            const icon = step.querySelector('.step-icon');
            if (icon) icon.textContent = '○';
        }
    }
    const step6Text = document.getElementById('step-6-text');
    if (step6Text) step6Text.textContent = 'Decision pending...';

    // Reset consensus steps
    for (let i = 1; i <= 4; i++) {
        const cons = document.getElementById(`cons-${i}`);
        if (cons) cons.style.borderLeftColor = '#3b4a6b';
        const result = document.getElementById(`cons-${i}-result`);
        if (result) result.textContent = 'Waiting...';
    }

    // Reset device states
    const deviceStates = document.getElementById('device-states');
    if (deviceStates) {
        deviceStates.innerHTML = '<div style="color: var(--text-muted);">Initializing devices...</div>';
    }
    const triggeredCount = document.getElementById('triggered-count');
    if (triggeredCount) triggeredCount.textContent = '0 triggered';
}

function updateSimStatus(status, color) {
    const simStatus = document.getElementById('sim-status');
    if (simStatus) {
        simStatus.textContent = status;
        simStatus.style.background = `${color}22`;
        simStatus.style.color = color;
    }
}

async function animateStep(stepNum, delayMs) {
    return new Promise(resolve => {
        setTimeout(() => {
            const step = document.getElementById(`step-${stepNum}`);
            if (step) {
                step.style.opacity = '1';
                const icon = step.querySelector('.step-icon');
                if (icon) {
                    icon.textContent = '●';
                    icon.style.color = '#4ade80';
                }
            }
            resolve();
        }, delayMs);
    });
}

function updateDeviceStates(mobileCount, iotCount, phase, decision = null) {
    const deviceStates = document.getElementById('device-states');
    const triggeredCount = document.getElementById('triggered-count');
    if (!deviceStates) return;

    if (phase === 'detecting') {
        const trigMobile = Math.floor(mobileCount * 0.7);
        const trigIot = Math.floor(iotCount * 0.9);
        deviceStates.innerHTML = `
            <div style="color: #fbbf24;">[*] Detecting signals...</div>
            <div>Mobile: ${trigMobile}/${mobileCount} responding</div>
            <div>IoT: ${trigIot}/${iotCount} responding</div>
        `;
        if (triggeredCount) triggeredCount.textContent = `${trigMobile + trigIot} responding`;
    } else if (phase === 'processing') {
        deviceStates.innerHTML = `
            <div style="color: #3b82f6;">[AI] Running Edge AI...</div>
            <div>Model: 1D-CNN v1.0 (343KB)</div>
            <div>Inference: ~45ms per device</div>
        `;
    } else if (phase === 'complete' && decision) {
        const trigMobile = decision.mobile_count || Math.floor(mobileCount * 0.7);
        const trigIot = decision.iot_count || Math.floor(iotCount * 0.9);
        deviceStates.innerHTML = `
            <div style="color: ${decision.is_earthquake ? '#f43f5e' : '#34d399'}; font-weight: 600;">
                ${decision.is_earthquake ? '[!] Alert sent' : '[OK] Normal status'}
            </div>
            <div>[M] Mobile triggered: ${trigMobile}</div>
            <div>[I] IoT triggered: ${trigIot}</div>
            <div>Score: ${decision.score.toFixed(3)}</div>
        `;
        if (triggeredCount) triggeredCount.textContent = `${trigMobile + trigIot} triggered`;
    }
}

async function updateConsensusSteps(decision, mobileCount, iotCount, threshold) {
    const mobileW = els.mobileSlider ? parseFloat(els.mobileSlider.value) : 0.3;
    const iotW = els.iotSlider ? parseFloat(els.iotSlider.value) : 0.7;

    // Step 1: DBSCAN Clustering
    await new Promise(r => setTimeout(r, 200));
    updateConsensusStep(1, `→ 1 cluster found (${decision.mobile_count + decision.iot_count} devices)`, '#4ade80');

    // Step 2: Temporal Window
    await new Promise(r => setTimeout(r, 200));
    updateConsensusStep(2, `→ ${decision.mobile_count + decision.iot_count} events in 2s window`, '#4ade80');

    // Step 3: Weighted Voting
    await new Promise(r => setTimeout(r, 200));
    const mobileScore = (decision.mobile_count * mobileW).toFixed(2);
    const iotScore = (decision.iot_count * iotW).toFixed(2);
    const totalWeight = (decision.mobile_count * mobileW + decision.iot_count * iotW).toFixed(2);
    updateConsensusStep(3, `→ Mobile: ${mobileScore} + IoT: ${iotScore} = ${decision.score.toFixed(3)}`, '#4ade80');

    // Step 4: Threshold Check
    await new Promise(r => setTimeout(r, 200));
    const passed = decision.score >= threshold;
    updateConsensusStep(4, `→ ${decision.score.toFixed(2)} ${passed ? '≥' : '<'} ${threshold} → ${passed ? 'ALERT' : 'OK'}`,
        passed ? '#f43f5e' : '#34d399');
}

function updateConsensusStep(stepNum, text, color) {
    const cons = document.getElementById(`cons-${stepNum}`);
    const result = document.getElementById(`cons-${stepNum}-result`);
    if (cons) cons.style.borderLeftColor = color;
    if (result) {
        result.textContent = text;
        result.style.color = color;
    }
}

// --- Map Animation Functions ---

// Store simulated device markers for animation
let simulatedMobileMarkers = [];
let simulatedIotMarkers = [];

function showWaveAnimation(lat, lon, magnitude) {
    eventLayer.clearLayers();

    // Epicenter marker with magnitude
    const epicenter = L.circleMarker([lat, lon], {
        radius: 15 + magnitude * 2,
        color: '#f43f5e',
        fillColor: '#f43f5e',
        fillOpacity: 0.7,
        weight: 2
    }).addTo(eventLayer);

    epicenter.bindTooltip(`M${magnitude} Epicenter`, { permanent: true, direction: 'top', className: 'epicenter-label' });

    // P-wave circle animation
    const wave = L.circle([lat, lon], {
        radius: 1000,
        color: '#f43f5e',
        weight: 2,
        fillOpacity: 0.1,
        dashArray: '5, 5'
    }).addTo(eventLayer);

    // Animate wave expansion
    let radius = 1000;
    const maxRadius = 60000; // 60km
    const animateWave = () => {
        radius += 800;
        wave.setRadius(radius);
        wave.setStyle({ opacity: Math.max(0.1, 1 - radius / maxRadius) });
        if (radius < maxRadius) {
            requestAnimationFrame(animateWave);
        } else {
            wave.remove();
        }
    };
    animateWave();

    // Fly to epicenter
    map.flyTo([lat, lon], 10, { duration: 0.5 });
}

function renderSimulatedDevices(epicenterLat, epicenterLon, mobileCount, iotCount) {
    // Clear previous simulated markers
    simulatedMobileMarkers.forEach(m => m.remove());
    simulatedIotMarkers.forEach(m => m.remove());
    simulatedMobileMarkers = [];
    simulatedIotMarkers = [];

    // Generate mobile devices in random positions around epicenter (within 30km)
    for (let i = 0; i < Math.min(mobileCount, 100); i++) { // Limit to 100 for performance
        const angle = Math.random() * 2 * Math.PI;
        const distance = Math.random() * 0.3; // ~30km in degrees
        const lat = epicenterLat + distance * Math.cos(angle);
        const lon = epicenterLon + distance * Math.sin(angle) / Math.cos(epicenterLat * Math.PI / 180);

        const marker = L.circleMarker([lat, lon], {
            radius: 4,
            color: '#38bdf8',
            fillColor: '#38bdf8',
            fillOpacity: 0.6,
            weight: 1
        }).addTo(mobileLayer);

        simulatedMobileMarkers.push(marker);
    }

    // Generate IoT anchors (fixed positions, evenly distributed)
    for (let i = 0; i < iotCount; i++) {
        const angle = (i / iotCount) * 2 * Math.PI;
        const distance = 0.1 + Math.random() * 0.15; // 10-25km from epicenter
        const lat = epicenterLat + distance * Math.cos(angle);
        const lon = epicenterLon + distance * Math.sin(angle) / Math.cos(epicenterLat * Math.PI / 180);

        const marker = L.circleMarker([lat, lon], {
            radius: 8,
            color: '#818cf8',
            fillColor: '#818cf8',
            fillOpacity: 0.8,
            weight: 2
        }).addTo(iotLayer);

        simulatedIotMarkers.push(marker);
    }

    // Update device count display
    if (els.deviceCount) {
        els.deviceCount.textContent = mobileCount + iotCount;
    }
}

function highlightTriggeredDevices() {
    // Animate devices being triggered (turn red briefly then green)
    const triggerDelay = 50;

    // Trigger IoT devices first (they're more reliable)
    simulatedIotMarkers.forEach((marker, i) => {
        setTimeout(() => {
            marker.setStyle({ color: '#fbbf24', fillColor: '#fbbf24' }); // Yellow = detecting
            setTimeout(() => {
                marker.setStyle({ color: '#4ade80', fillColor: '#4ade80' }); // Green = triggered
            }, 200);
        }, i * triggerDelay);
    });

    // Trigger mobile devices (some random ones won't trigger)
    simulatedMobileMarkers.forEach((marker, i) => {
        const willTrigger = Math.random() > 0.3; // 70% trigger rate
        setTimeout(() => {
            if (willTrigger) {
                marker.setStyle({ color: '#fbbf24', fillColor: '#fbbf24' });
                setTimeout(() => {
                    marker.setStyle({ color: '#4ade80', fillColor: '#4ade80' });
                }, 200);
            } else {
                marker.setStyle({ opacity: 0.3, fillOpacity: 0.3 }); // Dim non-triggered
            }
        }, i * (triggerDelay / 2) + 100);
    });
}


// Trigger Simulation
if (els.btnSim) {
    els.btnSim.addEventListener('click', async () => {
        els.btnSim.disabled = true;
        els.btnSim.innerHTML = '⏳ Simulating...';
        setStatus('simulation', true);

        const eqCount = document.getElementById('eval-eq-count')?.value || 20;
        const fpCount = document.getElementById('eval-fp-count')?.value || 100;

        try {
            await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    n_earthquakes: parseInt(eqCount),
                    n_false_positives: parseInt(fpCount)
                })
            });
        } catch (e) {
            els.btnSim.disabled = false;
            els.btnSim.innerHTML = '🚀 Run Batch Simulation';
        }
    });
}


// --- Render Functions ---

function setStatus(mode, running) {
    els.statusDot.className = 'status-dot';

    if (!running) {
        els.statusDot.classList.add('standby');
        els.statusText.textContent = 'Standby';
        return;
    }

    if (mode === 'simulation') {
        els.statusDot.classList.add('simulation');
        els.statusText.textContent = 'Running Simulation';
    } else if (mode === 'alert') {
        els.statusDot.classList.add('alert');
        els.statusText.textContent = 'EARTHQUAKE ALERT';
        document.body.style.boxShadow = 'inset 0 0 50px rgba(244, 63, 94, 0.5)';
        setTimeout(() => document.body.style.boxShadow = 'none', 3000);
    } else {
        els.statusDot.classList.add('active');
        els.statusText.textContent = 'Monitoring Active';
    }
}

function updateConnectionStatus(connected) {
    if (connected) {
        els.statusDot.style.backgroundColor = 'var(--accent-success)';
        els.statusText.textContent = 'Connected';
    } else {
        els.statusDot.style.backgroundColor = 'var(--text-muted)';
        els.statusText.textContent = 'Disconnected';
    }
}

function renderDevices(devices) {
    mobileLayer.clearLayers();
    iotLayer.clearLayers();

    devices.mobile.forEach(d => {
        L.marker([d.lat, d.lon], {
            icon: createIcon('#38bdf8', 4)
        }).addTo(mobileLayer);
    });

    devices.iot.forEach(d => {
        L.marker([d.lat, d.lon], {
            icon: createIcon('#818cf8', 10, true)
        }).addTo(iotLayer);
    });

    els.deviceCount.textContent = devices.mobile.length + devices.iot.length;
}

function showEarthquakeOnMap(event) {
    eventLayer.clearLayers();

    // Epicenter
    L.circleMarker([event.lat, event.lon], {
        radius: 20,
        color: '#f43f5e',
        fillColor: '#f43f5e',
        fillOpacity: 0.5
    }).addTo(eventLayer);

    // Wave
    const circle = L.circle([event.lat, event.lon], {
        radius: 1000,
        color: '#f43f5e',
        weight: 1,
        fillOpacity: 0.1
    }).addTo(eventLayer);

    // Animation
    let r = 1000;
    const animate = () => {
        r += 1000;
        circle.setRadius(r);
        if (r < 50000) requestAnimationFrame(animate);
    };
    animate();

    map.flyTo([event.lat, event.lon], 10);
}

function addEventLog(event) {
    // Remove empty state if present
    const emptyState = els.eventList.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const item = document.createElement('div');
    item.className = 'event-item';

    const isQuake = event.type === 'earthquake' && event.detected;
    const color = isQuake ? 'var(--accent-danger)' : 'var(--accent-success)';
    const icon = isQuake
        ? '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>'
        : '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';

    item.innerHTML = `
        <div class="event-icon" style="color: ${color}; background: ${isQuake ? 'rgba(244,63,94,0.1)' : 'rgba(52,211,153,0.1)'}">
            ${icon}
        </div>
        <div class="event-details">
            <h4>${isQuake ? `Earthquake M${event.magnitude}` : (event.detected ? 'False Alarm' : 'Noise Ignored')}</h4>
            <p>${new Date(event.timestamp * 1000).toLocaleTimeString()} • Score: ${event.score.toFixed(2)}</p>
        </div>
    `;

    els.eventList.prepend(item);
    if (els.eventList.children.length > 20) {
        els.eventList.lastChild.remove();
    }
}

function updateMetrics(m) {
    if (els.metricTpr) els.metricTpr.textContent = (m.recall_tpr * 100).toFixed(0) + '%';
    if (els.metricFpr) els.metricFpr.textContent = (m.fpr * 100).toFixed(1) + '%';
    if (els.metricF1) els.metricF1.textContent = m.f1_score.toFixed(2);
    if (els.metricPrecision) els.metricPrecision.textContent = (m.precision * 100).toFixed(0) + '%';
}

// Initial fetch
fetch('/api/status').then(r => r.json()).then(data => {
    setStatus(data.mode, data.running);
    if (data.devices) renderDevices(data.devices);
});


// --- TABS LOGIC ---
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        // Update Sidebar
        document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');

        // Show Content
        const tabId = item.getAttribute('data-tab');
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(`tab-${tabId}`).classList.add('active');

        // Special Actions
        if (tabId === 'home') {
            map.invalidateSize();
        }
        if (tabId === 'analysis') {
            loadAnalysisData();
        }
        if (tabId === 'models') {
            loadModels();
        }
    });
});

// --- CHARTS LOGIC ---
let charts = {};

async function loadAnalysisData() {
    try {
        const res = await fetch('/api/analysis/sample');
        const data = await res.json();

        updateChart('chart-raw', 'Raw Signal', data.labels, data.raw, '#94a3b8');
        updateChart('chart-filtered', 'Filtered', data.labels, data.filtered, '#38bdf8');
        updateChart('chart-normalized', 'Normalized', data.labels, data.normalized, '#34d399');

    } catch (e) {
        console.error("Analysis load failed", e);
    }
}

function updateChart(canvasId, label, labels, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderColor: color,
                borderWidth: 1.5,
                backgroundColor: color + '20',
                fill: true,
                pointRadius: 0,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: {
                    display: false,
                    grid: { display: false }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#64748b', font: { size: 9 } }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

document.getElementById('btn-refresh-analysis')?.addEventListener('click', loadAnalysisData);

// ==================================================
// MODELS TAB - Dynamic Model Loading
// ==================================================

let modelsData = [];
let modelsChart = null;

async function loadModels() {
    console.log('🔍 loadModels() called - Starting model fetch...');
    const loading = document.getElementById('models-loading');
    const table = document.getElementById('models-table');
    const tbody = document.getElementById('models-tbody');

    if (loading) loading.style.display = 'block';
    if (table) table.style.display = 'none';

    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        modelsData = data.models || [];

        if (tbody) {
            tbody.innerHTML = modelsData.map((model, index) => {
                const recall = model.recall ? (model.recall * 100).toFixed(1) + '%' : '-';
                const precision = model.precision ? (model.precision * 100).toFixed(1) + '%' : '-';
                const f1 = model.f1_score ? model.f1_score.toFixed(2) : '-';
                const acc = model.accuracy ? (model.accuracy * 100).toFixed(0) + '%' : '-';
                const statusClass = model.status === 'current' ? 'status-active' : 'status-archived';
                const statusLabel = model.status === 'current' ? 'CURRENT' : 'Available';
                const rowStyle = model.status === 'current' ? 'background: rgba(56, 189, 248, 0.15);' : '';
                const recallClass = model.recall && model.recall >= 0.90 ? 'good' : '';
                const precClass = model.precision && model.precision >= 0.70 ? 'good' : (model.precision && model.precision < 0.6 ? 'bad' : '');

                return `
                    <tr style="${rowStyle}" data-index="${index}">
                        <td><strong>${model.version}</strong></td>
                        <td>${model.type.toUpperCase()}</td>
                        <td>${model.size_kb} KB</td>
                        <td class="${recallClass}">${recall}</td>
                        <td class="${precClass}">${precision}</td>
                        <td>${f1}</td>
                        <td>${acc}</td>
                        <td><span class="${statusClass}">${statusLabel}</span></td>
                        <td>
                            <button class="btn-select-model" data-index="${index}" 
                                style="padding: 4px 8px; font-size: 10px; background: var(--accent-primary); border: none; border-radius: 4px; color: white; cursor: pointer;">
                                Select
                            </button>
                        </td>
                    </tr>
                `;
            }).join('');

            // Add click handlers
            document.querySelectorAll('.btn-select-model').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const idx = parseInt(e.target.dataset.index);
                    selectModel(modelsData[idx]);
                });
            });
        }

        if (loading) loading.style.display = 'none';
        if (table) table.style.display = 'table';

        // Update chart
        updateModelsChart();

    } catch (error) {
        console.error('Failed to load models:', error);
        if (loading) loading.innerHTML = 'Failed to load models. Check if server is running.';
    }
}

function selectModel(model) {
    const detailsSection = document.getElementById('selected-model-details');
    const content = document.getElementById('model-details-content');

    if (detailsSection && content) {
        detailsSection.style.display = 'block';
        content.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
                <div class="info-card">
                    <h4>${model.name}</h4>
                    <ul style="list-style: none; padding: 0; margin: 10px 0; font-size: 12px;">
                        <li><strong>Version:</strong> ${model.version}</li>
                        <li><strong>Type:</strong> ${model.type}</li>
                        <li><strong>Size:</strong> ${model.size_kb} KB</li>
                        <li><strong>Status:</strong> ${model.status}</li>
                    </ul>
                </div>
                <div class="info-card">
                    <h4>Performance</h4>
                    <ul style="list-style: none; padding: 0; margin: 10px 0; font-size: 12px;">
                        <li><strong>Recall:</strong> ${model.recall ? (model.recall * 100).toFixed(1) + '%' : 'N/A'}</li>
                        <li><strong>Precision:</strong> ${model.precision ? (model.precision * 100).toFixed(1) + '%' : 'N/A'}</li>
                        <li><strong>F1 Score:</strong> ${model.f1_score ? model.f1_score.toFixed(3) : 'N/A'}</li>
                        <li><strong>Accuracy:</strong> ${model.accuracy ? (model.accuracy * 100).toFixed(0) + '%' : 'N/A'}</li>
                    </ul>
                </div>
                <div class="info-card">
                    <h4>File Path</h4>
                    <div style="font-size: 10px; word-break: break-all; color: var(--text-muted); padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">
                        ${model.path || 'Unknown'}
                    </div>
                </div>
            </div>
        `;

        // Highlight selected row
        document.querySelectorAll('#models-tbody tr').forEach(row => {
            row.style.border = 'none';
        });
        const selectedRow = document.querySelector(`#models-tbody tr[data-index="${modelsData.indexOf(model)}"]`);
        if (selectedRow) {
            selectedRow.style.border = '2px solid var(--accent-primary)';
        }
    }
}

function updateModelsChart() {
    const ctx = document.getElementById('models-chart');
    if (!ctx) return;

    // Prepare data with only models that have metrics
    const validModels = modelsData.filter(m => m.recall || m.precision || m.f1_score);

    if (validModels.length === 0) return;

    const labels = validModels.map(m => m.version);
    const recalls = validModels.map(m => (m.recall || 0) * 100);
    const precisions = validModels.map(m => (m.precision || 0) * 100);
    const f1s = validModels.map(m => (m.f1_score || 0) * 100);

    if (modelsChart) {
        modelsChart.destroy();
    }

    modelsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Recall (%)',
                    data: recalls,
                    backgroundColor: 'rgba(52, 211, 153, 0.7)',
                    borderColor: '#34d399',
                    borderWidth: 1
                },
                {
                    label: 'Precision (%)',
                    data: precisions,
                    backgroundColor: 'rgba(56, 189, 248, 0.7)',
                    borderColor: '#38bdf8',
                    borderWidth: 1
                },
                {
                    label: 'F1 Score (%)',
                    data: f1s,
                    backgroundColor: 'rgba(251, 191, 36, 0.7)',
                    borderColor: '#fbbf24',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: { color: '#94a3b8', font: { size: 11 } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                }
            }
        }
    });
}

// Load models when Models tab is clicked
document.querySelector('[data-tab="models"]')?.addEventListener('click', () => {
    setTimeout(loadModels, 100);
});

// Refresh button
document.getElementById('btn-refresh-models')?.addEventListener('click', loadModels);

// Auto-load if Models tab is already active
if (document.getElementById('tab-models')?.classList.contains('active')) {
    loadModels();
}

// ==================================================
// MODEL SELECTOR - Sidebar dropdown functionality
// ==================================================

const modelSelect = document.getElementById('model-select');
let allModelsData = [];

// Load all models for sidebar dropdown
async function loadModelSelector() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        allModelsData = data.models || [];
        const currentModel = data.current_model || 'v1.0';

        if (modelSelect) {
            modelSelect.innerHTML = allModelsData.map(m => {
                const label = m.version === currentModel ? `${m.version} (Current)` : m.version;
                return `<option value="${m.version}" ${m.version === currentModel ? 'selected' : ''}>${label}</option>`;
            }).join('');
        }

        // Update sidebar model info
        const currentModelData = allModelsData.find(m => m.version === currentModel);
        if (currentModelData) {
            updateSidebarModelInfo(currentModelData);
        }
    } catch (error) {
        console.error('Failed to load model selector:', error);
    }
}

// Handle model selection change
if (modelSelect) {
    modelSelect.addEventListener('change', async (e) => {
        const version = e.target.value;
        try {
            const response = await fetch('/api/models/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ version })
            });
            const data = await response.json();

            if (data.success) {
                console.log('Model switched to:', version);

                // Update sidebar info
                const modelData = allModelsData.find(m => m.version === version);
                if (modelData) {
                    updateSidebarModelInfo(modelData);
                    updateRightPanelMetrics(modelData);
                }

                // Update dropdown labels
                loadModelSelector();
            }
        } catch (error) {
            console.error('Failed to switch model:', error);
        }
    });
}

// Update sidebar model info display
function updateSidebarModelInfo(model) {
    const modelName = document.getElementById('model-name');
    const modelSize = document.getElementById('model-size');
    const sysModel = document.getElementById('sys-model');

    if (modelName) modelName.textContent = model.name || `KURTAR ${model.version}`;
    if (modelSize) modelSize.textContent = `${model.size_kb} KB ${model.type.toUpperCase()}`;
    if (sysModel) sysModel.textContent = model.version;
}

// Update right panel metrics with selected model data
function updateRightPanelMetrics(model) {
    const v1Recall = document.getElementById('v1-recall');
    const v1Precision = document.getElementById('v1-precision');
    const v1F1 = document.getElementById('v1-f1');
    const v1Samples = document.getElementById('v1-samples');

    if (v1Recall && model.recall) {
        v1Recall.textContent = (model.recall * 100).toFixed(0) + '%';
    }
    if (v1Precision && model.precision) {
        v1Precision.textContent = (model.precision * 100).toFixed(0) + '%';
    }
    if (v1F1 && model.f1_score) {
        v1F1.textContent = model.f1_score.toFixed(2);
    }
    if (v1Samples && model.dataset) {
        const total = model.dataset.total || (model.dataset.eq_windows + model.dataset.har_windows);
        v1Samples.textContent = total ? (total / 1000).toFixed(0) + 'K' : '18K';
    }

    // Update panel title to reflect selected model
    const panelTitle = document.querySelector('#tab-home .right-column .panel:first-child .panel-title');
    if (panelTitle) {
        panelTitle.innerHTML = `Model ${model.version} Results <span style="font-size: 9px; background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 6px; border-radius: 10px; margin-left: 8px;">${model.recall > 0.99 ? 'EXCELLENT' : 'REAL DATA'}</span>`;
    }
}

// Listen for model changes from server
socket.on('model_changed', (data) => {
    console.log('Model changed event:', data);
    loadModelSelector();
});

// Load model selector on page load
loadModelSelector();

