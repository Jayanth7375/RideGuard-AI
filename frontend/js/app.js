/**
 * RideGuard AI - Enterprise Single-Page Application Controller
 * Handles dynamic CSV ingestion, column mapping, validation, ML training,
 * and live dashboard visualizations with strict empty-state management.
 */

// Plotly White & Yellow Light Theme Configuration
const PLOTLY_THEME = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(248, 250, 252, 0.7)',
    font: {
        family: 'Inter, sans-serif',
        color: '#475569',
        size: 11
    },
    margin: { l: 30, r: 20, t: 30, b: 30 },
    xaxis: {
        gridcolor: '#E2E8F0',
        linecolor: '#CBD5E1',
        zerolinecolor: '#E2E8F0',
        tickfont: { color: '#475569', size: 10 }
    },
    yaxis: {
        gridcolor: '#E2E8F0',
        linecolor: '#CBD5E1',
        zerolinecolor: '#E2E8F0',
        tickfont: { color: '#475569', size: 10 }
    }
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false
};

// Global Application State
let g_activeDataset = null;
let g_modelsData = null;
let g_recommendations = [];

document.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) {
        lucide.createIcons();
    }

    initSidebarCollapse();
    initNavigation();
    initGotoUploadButtons();
    loadTelemetry();
    loadOverview();
    initPredictionForm();
    initBatchPrediction();
    initModelRetrain();
    initDatasetManager();
});

// -------------------------------------------------------------
// COLLAPSIBLE & COMPRESSIBLE SIDEBAR HANDLER
// -------------------------------------------------------------
function initSidebarCollapse() {
    const toggleBtn = document.getElementById('btn-top-sidebar-toggle');
    const collapseBtn = document.getElementById('btn-sidebar-collapse');
    const icon = document.getElementById('sidebar-toggle-icon');

    const setCollapsed = (collapsed) => {
        if (collapsed) {
            document.body.classList.add('sidebar-compressed');
            if (icon) icon.setAttribute('data-lucide', 'panel-left-open');
            localStorage.setItem('rg-sidebar-compressed', 'true');
        } else {
            document.body.classList.remove('sidebar-compressed');
            if (icon) icon.setAttribute('data-lucide', 'panel-left-close');
            localStorage.setItem('rg-sidebar-compressed', 'false');
        }
        if (window.lucide) lucide.createIcons();
        setTimeout(() => {
            document.querySelectorAll('.js-plotly-plot').forEach(c => {
                try { Plotly.Plots.resize(c); } catch (e) {}
            });
        }, 300);
    };

    // Restore user preference from localStorage
    if (localStorage.getItem('rg-sidebar-compressed') === 'true') {
        setCollapsed(true);
    }

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const isCurrently = document.body.classList.contains('sidebar-compressed');
            setCollapsed(!isCurrently);
        });
    }

    if (collapseBtn) {
        collapseBtn.addEventListener('click', () => {
            setCollapsed(true);
        });
    }
}

// -------------------------------------------------------------
// SPA NAVIGATION CONTROLLER
// -------------------------------------------------------------
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const views = document.querySelectorAll('.view-panel');
    const breadcrumb = document.getElementById('top-breadcrumb');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('data-target');
            const viewTitle = link.querySelector('span').textContent;

            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            views.forEach(v => v.classList.add('hidden'));
            const targetView = document.getElementById(targetId);
            if (targetView) {
                targetView.classList.remove('hidden');
                breadcrumb.textContent = viewTitle;
                handleViewActivation(targetId);
            }
        });
    });
}

function switchToTab(targetViewId, tabTitle) {
    const navLinks = document.querySelectorAll('.nav-link');
    const views = document.querySelectorAll('.view-panel');
    const breadcrumb = document.getElementById('top-breadcrumb');

    navLinks.forEach(l => {
        if (l.getAttribute('data-target') === targetViewId) {
            l.classList.add('active');
        } else {
            l.classList.remove('active');
        }
    });

    views.forEach(v => v.classList.add('hidden'));
    const targetView = document.getElementById(targetViewId);
    if (targetView) {
        targetView.classList.remove('hidden');
        if (breadcrumb && tabTitle) breadcrumb.textContent = tabTitle;
        handleViewActivation(targetViewId);
    }
}

function initGotoUploadButtons() {
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.btn-goto-upload');
        if (btn) {
            e.preventDefault();
            switchToTab('view-dataset', 'Dataset Management');
        }
    });
}

function handleViewActivation(viewId) {
    setTimeout(() => {
        const charts = document.querySelectorAll('.js-plotly-plot');
        charts.forEach(c => Plotly.Plots.resize(c));
    }, 120);

    if (viewId === 'view-overview') {
        loadOverview();
    } else if (viewId === 'view-analytics') {
        loadAnalytics();
    } else if (viewId === 'view-prediction') {
        loadPredictionState();
    } else if (viewId === 'view-models') {
        loadModelPerformance();
    } else if (viewId === 'view-explainable') {
        loadExplainability();
    } else if (viewId === 'view-recommendations') {
        loadRecommendations();
    } else if (viewId === 'view-dataset') {
        loadDatasetInfo();
    }
}

// -------------------------------------------------------------
// TELEMETRY & SYSTEM STATUS
// -------------------------------------------------------------
async function loadTelemetry() {
    try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();

        const sidebarLabel = document.getElementById('sidebar-dataset-label');
        const sidebarRows = document.getElementById('sidebar-dataset-rows');
        const sidebarChamp = document.getElementById('sidebar-champion-model');
        const topText = document.getElementById('top-dataset-text');
        const topDot = document.getElementById('top-dataset-dot');
        const topModelText = document.getElementById('top-model-text');
        const topModelBadge = document.getElementById('top-model-badge');

        if (data.dataset_connected) {
            sidebarLabel.textContent = data.dataset_source;
            sidebarRows.textContent = `${(data.total_rows ?? 0).toLocaleString()} rides`;
            topText.textContent = `${data.dataset_source}: ${(data.total_rows ?? 0).toLocaleString()} rides`;
            topDot.className = 'w-1.5 h-1.5 rounded-full bg-rg-success animate-pulse';

            if (data.champion_model) {
                sidebarChamp.textContent = data.champion_model;
                sidebarChamp.className = 'font-semibold text-rg-success';
                topModelText.textContent = `${data.champion_model} Online`;
                topModelBadge.className = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono bg-rg-elevated border border-blue-500/30 text-blue-300';
            } else {
                sidebarChamp.textContent = 'Awaiting Training';
                sidebarChamp.className = 'font-semibold text-rg-warning';
                topModelText.textContent = 'Models Not Trained';
                topModelBadge.className = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono bg-rg-elevated border border-rg-border text-rg-muted';
            }
        } else {
            sidebarLabel.textContent = 'No Dataset';
            sidebarRows.textContent = '0 rides';
            sidebarChamp.textContent = '--';
            sidebarChamp.className = 'font-semibold text-rg-muted';
            topText.textContent = 'No Dataset Connected';
            topDot.className = 'w-1.5 h-1.5 rounded-full bg-rg-dim';
            topModelText.textContent = 'Models Not Trained';
            topModelBadge.className = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono bg-rg-elevated border border-rg-border text-rg-muted';
        }
    } catch (err) {
        console.error('Failed to load telemetry', err);
    }
}

// -------------------------------------------------------------
// VIEW 1: EXECUTIVE OVERVIEW
// -------------------------------------------------------------
async function loadOverview() {
    try {
        const res = await fetch('/api/overview');
        const data = await res.json();

        const emptyState = document.getElementById('overview-empty-state');
        const contentArea = document.getElementById('overview-content');

        if (!data.dataset_connected || data.status === 'no_dataset') {
            emptyState.classList.remove('hidden');
            contentArea.classList.add('hidden');
            return;
        }

        emptyState.classList.add('hidden');
        contentArea.classList.remove('hidden');

        // 1. KPI Values
        const metrics = data.metrics || data.kpis || {};
        document.getElementById('kpi-total-bookings').textContent = (metrics.total_bookings ?? 0).toLocaleString();
        document.getElementById('kpi-completed-rides').textContent = (metrics.completed_rides ?? 0).toLocaleString();
        document.getElementById('kpi-completed-pct').textContent = `${(100 - (metrics.cancellation_rate ?? 0)).toFixed(1)}% fulfillment rate`;
        document.getElementById('kpi-cancelled-rides').textContent = (metrics.cancelled_rides ?? 0).toLocaleString();
        document.getElementById('kpi-cancelled-pct').textContent = `${(metrics.cancellation_rate ?? 0).toFixed(1)}% dropout rate`;
        document.getElementById('kpi-cancellation-rate').textContent = `${(metrics.cancellation_rate ?? 0).toFixed(1)}%`;
        
        const avgWait = metrics.average_wait_time ?? metrics.avg_waiting_time;
        document.getElementById('kpi-avg-wait').textContent = (avgWait !== null && avgWait !== undefined)
            ? `${Number(avgWait).toFixed(1)} min`
            : 'N/A';

        // 2. Booking Outcome Donut Chart
        const donut = data.booking_distribution || data.donut || {};
        const completedCount = donut.completed ?? metrics.completed_rides ?? 0;
        const cancelledCount = donut.cancelled ?? metrics.cancelled_rides ?? 0;

        if (completedCount > 0 || cancelledCount > 0) {
            const donutData = [{
                values: [completedCount, cancelledCount],
                labels: ['Completed', 'Cancelled'],
                type: 'pie',
                hole: 0.65,
                marker: {
                    colors: ['#22C55E', '#EF4444'],
                    line: { color: '#080D18', width: 2 }
                },
                textinfo: 'label+percent',
                hoverinfo: 'label+value+percent'
            }];

            const donutLayout = {
                ...PLOTLY_THEME,
                showlegend: true,
                legend: { x: 0.1, y: -0.1, orientation: 'h' }
            };
            Plotly.newPlot('chart-outcome-donut', donutData, donutLayout, PLOTLY_CONFIG);
        }

        // 3. Hourly Trend Dual-Axis Chart
        const hourlyTrend = data.hourly_analytics || data.hourly_trend || [];
        if (hourlyTrend && hourlyTrend.length > 0) {
            const hours = hourlyTrend.map(d => d.hour);
            const volumes = hourlyTrend.map(d => d.total);
            const rates = hourlyTrend.map(d => d.cancellation_rate ?? d.rate ?? 0);

            const trendTraces = [
                {
                    x: hours,
                    y: volumes,
                    name: 'Total Bookings',
                    type: 'bar',
                    marker: { color: 'rgba(59, 130, 246, 0.45)' },
                    yaxis: 'y'
                },
                {
                    x: hours,
                    y: rates,
                    name: 'Cancellation Rate (%)',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#EF4444', width: 2.5 },
                    marker: { size: 5, color: '#F87171' },
                    yaxis: 'y2'
                }
            ];

            const trendLayout = {
                ...PLOTLY_THEME,
                xaxis: { ...PLOTLY_THEME.xaxis, title: 'Hour of Day (0-23)' },
                yaxis: { ...PLOTLY_THEME.yaxis, title: 'Volume', side: 'left' },
                yaxis2: {
                    title: 'Cancellation Rate (%)',
                    overlaying: 'y',
                    side: 'right',
                    showgrid: false,
                    linecolor: '#243247'
                },
                legend: { x: 0.02, y: 0.98 }
            };
            Plotly.newPlot('chart-hourly-trend', trendTraces, trendLayout, PLOTLY_CONFIG);
        }

        // 4. Insights List
        const insights = data.cancellation_insights || data.insights || [];
        const insightsList = document.getElementById('overview-insights-list');
        if (insightsList) {
            insightsList.innerHTML = '';
            if (insights.length > 0) {
                insights.forEach(item => {
                    const li = document.createElement('li');
                    li.className = 'flex items-start gap-2';
                    li.innerHTML = `<span class="text-rg-primary mt-0.5">&bull;</span><span>${item}</span>`;
                    insightsList.appendChild(li);
                });
            } else {
                insightsList.innerHTML = '<li class="text-rg-dim italic">No statistically significant cancellation clusters detected in dataset.</li>';
            }
        }

        // 5. AI Summary
        const modelInfo = data.model || data.ai_summary || {};
        const modelName = modelInfo.name || modelInfo.model_name || 'Awaiting Training';
        const rocAuc = (modelInfo.roc_auc !== null && modelInfo.roc_auc !== undefined) ? Number(modelInfo.roc_auc).toFixed(3) : '--';
        const f1Score = (modelInfo.f1_score !== null && modelInfo.f1_score !== undefined) ? Number(modelInfo.f1_score).toFixed(3) : '--';
        const recordsAnalyzed = modelInfo.records_analyzed ?? metrics.total_bookings ?? 0;
        const topFactor = modelInfo.top_factor || 'Driver Arrival Time / Waiting Delay';

        document.getElementById('summary-model-name').textContent = modelName;
        document.getElementById('summary-metrics').textContent = `${rocAuc} / ${f1Score}`;
        document.getElementById('summary-records').textContent = `${recordsAnalyzed.toLocaleString()} trips`;
        document.getElementById('summary-top-factor').textContent = topFactor;

    } catch (err) {
        console.error('Failed to load overview data', err);
    }
}

// -------------------------------------------------------------
// VIEW 2: CANCELLATION ANALYTICS
// -------------------------------------------------------------
async function loadAnalytics() {
    try {
        const res = await fetch('/api/analytics');
        const data = await res.json();

        const emptyState = document.getElementById('analytics-empty-state');
        const contentArea = document.getElementById('analytics-content');

        if (data.status === 'no_dataset' || !data.vehicle_breakdown || data.vehicle_breakdown.length === 0) {
            emptyState.classList.remove('hidden');
            contentArea.classList.add('hidden');
            return;
        }

        emptyState.classList.add('hidden');
        contentArea.classList.remove('hidden');

        // 1. Vehicle Breakdown
        if (data.vehicle_breakdown && data.vehicle_breakdown.length > 0) {
            const vVehicles = data.vehicle_breakdown.map(v => v.vehicle);
            const vCompleted = data.vehicle_breakdown.map(v => v.completed);
            const vCancelled = data.vehicle_breakdown.map(v => v.cancelled);

            const vTraces = [
                { x: vVehicles, y: vCompleted, name: 'Completed', type: 'bar', marker: { color: '#22C55E' } },
                { x: vVehicles, y: vCancelled, name: 'Cancelled', type: 'bar', marker: { color: '#EF4444' } }
            ];
            const vLayout = { ...PLOTLY_THEME, barmode: 'group', legend: { x: 0.02, y: 0.98 } };
            Plotly.newPlot('chart-vehicle-breakdown', vTraces, vLayout, PLOTLY_CONFIG);
        }

        // 2. Location Breakdown
        if (data.location_breakdown && data.location_breakdown.length > 0) {
            const lSorted = [...data.location_breakdown].reverse();
            const lLocations = lSorted.map(l => l.location);
            const lRates = lSorted.map(l => Number(l.rate));

            const barColors = lRates.map(r => r > 40 ? '#EF4444' : (r > 25 ? '#EAB308' : '#FACC15'));

            const lTrace = [{
                x: lRates,
                y: lLocations,
                type: 'bar',
                orientation: 'h',
                marker: { color: barColors },
                text: lRates.map(r => `${r.toFixed(1)}%`),
                textposition: 'auto',
                textfont: { color: '#0F172A', size: 10, weight: 600 }
            }];
            const lLayout = {
                ...PLOTLY_THEME,
                height: 280,
                xaxis: { ...PLOTLY_THEME.xaxis, type: 'linear', title: 'Cancellation Rate (%)' },
                yaxis: { ...PLOTLY_THEME.yaxis, type: 'category', automargin: true },
                margin: { l: 150, r: 30, t: 20, b: 40 }
            };
            Plotly.newPlot('chart-location-breakdown', lTrace, lLayout, PLOTLY_CONFIG);
        }

        // 3. Waiting Delay Cohort
        if (data.wait_distribution && data.wait_distribution.length > 0) {
            const wBins = data.wait_distribution.map(w => w.bin);
            const wCompleted = data.wait_distribution.map(w => w.completed);
            const wCancelled = data.wait_distribution.map(w => w.cancelled);

            const wTraces = [
                { x: wBins, y: wCompleted, name: 'Completed', type: 'bar', marker: { color: '#22C55E' } },
                { x: wBins, y: wCancelled, name: 'Cancelled', type: 'bar', marker: { color: '#EF4444' } }
            ];
            const wLayout = { ...PLOTLY_THEME, barmode: 'stack', legend: { x: 0.8, y: 0.98 } };
            Plotly.newPlot('chart-wait-distribution', wTraces, wLayout, PLOTLY_CONFIG);
        }

        // 4. Correlation Matrix
        if (data.correlation && data.correlation.columns) {
            const cTrace = [{
                z: data.correlation.values,
                x: data.correlation.columns,
                y: data.correlation.columns,
                type: 'heatmap',
                colorscale: [[0, '#FEFCE8'], [0.5, '#FEF08A'], [1, '#EAB308']],
                zmin: -1,
                zmax: 1
            }];
            const cLayout = {
                ...PLOTLY_THEME,
                margin: { l: 90, r: 20, t: 20, b: 70 }
            };
            Plotly.newPlot('chart-correlation-matrix', cTrace, cLayout, PLOTLY_CONFIG);
        }

    } catch (err) {
        console.error('Failed to load analytics', err);
    }
}

// -------------------------------------------------------------
// VIEW 3: AI PREDICTION
// -------------------------------------------------------------
async function loadPredictionState() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();

        const emptyState = document.getElementById('prediction-empty-state');
        const contentArea = document.getElementById('prediction-content');

        if (!data.models_status || data.models_status !== 'ready') {
            if (emptyState) emptyState.classList.remove('hidden');
            if (contentArea) contentArea.classList.add('hidden');
        } else {
            if (emptyState) emptyState.classList.add('hidden');
            if (contentArea) contentArea.classList.remove('hidden');
            await loadPredictionSchema();
        }
    } catch (err) {
        console.error('Failed to check prediction state', err);
    }
}

async function loadPredictionSchema() {
    const container = document.getElementById('prediction-dynamic-fields');
    if (!container) return;

    try {
        const res = await fetch('/api/predict/schema');
        if (!res.ok) return;
        const schema = await res.json();
        const features = schema.features || [];

        const countBadge = document.getElementById('schema-feature-count');
        if (countBadge) {
            countBadge.textContent = `${features.length} Features Active`;
        }

        if (features.length === 0) {
            container.innerHTML = `
                <div class="col-span-full py-6 text-center text-slate-500">
                    <p class="text-xs">No model features detected. Please upload a dataset and train models.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = '';
        features.forEach(feat => {
            const fieldWrapper = document.createElement('div');
            fieldWrapper.className = 'col-span-1 space-y-1';

            if (feat.type === 'categorical') {
                const optionsHtml = (feat.options || []).map(opt =>
                    `<option value="${opt}" ${opt === feat.default ? 'selected' : ''}>${opt}</option>`
                ).join('');

                fieldWrapper.innerHTML = `
                    <label class="block text-slate-700 font-semibold text-xs tracking-tight">${feat.label}</label>
                    <select name="${feat.name}" data-type="categorical" class="dynamic-pred-input w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-slate-900 outline-none focus:border-yellow-500 focus:ring-1 focus:ring-yellow-400 text-xs font-medium shadow-sm transition-all hover:border-slate-400">
                        ${optionsHtml}
                    </select>
                `;
            } else {
                fieldWrapper.innerHTML = `
                    <div class="flex items-center justify-between">
                        <label class="block text-slate-700 font-semibold text-xs tracking-tight">${feat.label}</label>
                        <span class="text-[10px] text-slate-600 font-mono font-medium">Range: ${feat.min} - ${feat.max}</span>
                    </div>
                    <input type="number" name="${feat.name}" data-type="numerical" min="${feat.min}" max="${feat.max}" step="${feat.step}" value="${feat.default}" class="dynamic-pred-input w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-slate-900 outline-none focus:border-yellow-500 focus:ring-1 focus:ring-yellow-400 text-xs font-medium shadow-sm transition-all hover:border-slate-400">
                `;
            }
            container.appendChild(fieldWrapper);
        });

        if (window.lucide) window.lucide.createIcons();
    } catch (e) {
        console.warn('Could not populate dynamic prediction schema', e);
    }
}

async function populatePredictionDropdowns() {
    await loadPredictionSchema();
}

function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    const singleBtn = document.getElementById('pred-tab-single-btn');
    const batchBtn = document.getElementById('pred-tab-batch-btn');
    const singleTab = document.getElementById('pred-tab-single');
    const batchTab = document.getElementById('pred-tab-batch');

    if (singleBtn && batchBtn) {
        singleBtn.addEventListener('click', () => {
            singleBtn.className = 'border-b-2 border-rg-primary text-rg-primary pb-2 font-semibold';
            batchBtn.className = 'border-b-2 border-transparent text-rg-muted hover:text-white pb-2';
            singleTab.classList.remove('hidden');
            batchTab.classList.add('hidden');
        });

        batchBtn.addEventListener('click', () => {
            batchBtn.className = 'border-b-2 border-rg-primary text-rg-primary pb-2 font-semibold';
            singleBtn.className = 'border-b-2 border-transparent text-rg-muted hover:text-white pb-2';
            batchTab.classList.remove('hidden');
            singleTab.classList.add('hidden');
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const submitBtn = form.querySelector('button[type="submit"]');
        const origBtnHtml = submitBtn ? submitBtn.innerHTML : '';
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i data-lucide="loader" class="w-4 h-4 animate-spin"></i><span>Analyzing Dispatch Risk...</span>';
            if (window.lucide) window.lucide.createIcons();
        }

        const payload = {};
        const inputs = form.querySelectorAll('.dynamic-pred-input');
        if (inputs.length > 0) {
            inputs.forEach(input => {
                const name = input.getAttribute('name');
                if (!name) return;
                const dataType = input.getAttribute('data-type');
                if (dataType === 'numerical' || input.type === 'number') {
                    const val = parseFloat(input.value);
                    payload[name] = isNaN(val) ? 0 : val;
                } else {
                    payload[name] = input.value;
                }
            });
        } else {
            // Fallback for legacy static elements if present
            const vDist = parseFloat(document.getElementById('pred-distance')?.value);
            const vFare = parseFloat(document.getElementById('pred-fare')?.value);
            const vHour = parseInt(document.getElementById('pred-hour')?.value);
            const vEta = parseInt(document.getElementById('pred-eta')?.value);
            const vWait = parseInt(document.getElementById('pred-wait')?.value);

            payload['Vehicle_Type'] = document.getElementById('pred-vehicle')?.value || 'Prime Sedan';
            payload['Pickup_Location'] = document.getElementById('pred-pickup')?.value || 'Airport Terminal 1';
            payload['Drop_Location'] = document.getElementById('pred-drop')?.value || 'Downtown Arts District';
            payload['Ride_Distance'] = isNaN(vDist) ? 12.5 : vDist;
            payload['Booking_Value'] = isNaN(vFare) ? 280.0 : vFare;
            payload['Hour'] = isNaN(vHour) ? 17 : vHour;
            payload['Driver_Arrival_Time'] = isNaN(vEta) ? 8 : vEta;
            payload['Waiting_Time'] = isNaN(vWait) ? 4 : vWait;
            payload['Payment_Method'] = document.getElementById('pred-payment')?.value || 'UPI / Digital Wallet';
            payload['Customer_Rating'] = 4.6;
        }

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                let errMsg = 'Prediction failed.';
                try {
                    const errData = await res.json();
                    errMsg = errData.detail || errMsg;
                } catch (_) {
                    errMsg = await res.text().catch(() => 'Server error.');
                }
                alert(errMsg);
                return;
            }

            const data = await res.json();
            renderPredictionResults(data);

            // Subtle highlight animation on results card to give instant visual feedback
            const resCard = document.getElementById('pred-results-card');
            if (resCard) {
                resCard.classList.remove('ring-4', 'ring-yellow-400');
                void resCard.offsetWidth; // trigger reflow
                resCard.classList.add('ring-4', 'ring-yellow-400', 'transition-all', 'duration-300');
                setTimeout(() => {
                    resCard.classList.remove('ring-4', 'ring-yellow-400');
                }, 600);
            }
        } catch (err) {
            console.error('Prediction failed', err);
            alert('Prediction request failed.');
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = origBtnHtml;
                if (window.lucide) window.lucide.createIcons();
            }
        }
    });
}

function renderPredictionResults(data) {
    document.getElementById('pred-empty-state').classList.add('hidden');
    document.getElementById('pred-results-card').classList.remove('hidden');

    const probPct = data.cancellation_probability * 100;
    const willCancel = (data.will_cancel === 'YES' || data.predicted_status === 'Cancelled');
    const confidence = data.confidence_score ?? Math.round(Math.max(probPct, 100 - probPct) * 10) / 10;

    // 1. Verdict Box: YES / NO
    const verdictBox = document.getElementById('verdict-card-box');
    const verdictEl = document.getElementById('res-will-cancel');
    const verdictIcon = document.getElementById('res-verdict-icon');
    const verdictSub = document.getElementById('res-verdict-subtext');

    verdictEl.textContent = willCancel ? 'YES' : 'NO';
    if (willCancel) {
        verdictEl.className = 'text-3xl font-black tracking-tight text-rose-400';
        verdictIcon.textContent = '🚨';
        verdictSub.textContent = 'High Dropout & Cancellation Risk';
        verdictSub.className = 'text-[10px] font-semibold text-rose-300/90';
        verdictBox.className = 'border-2 border-rose-500/50 rounded-lg p-3 bg-rose-500/10 flex flex-col justify-between';
    } else {
        verdictEl.className = 'text-3xl font-black tracking-tight text-emerald-400';
        verdictIcon.textContent = '✅';
        verdictSub.textContent = 'High Confidence of Completion';
        verdictSub.className = 'text-[10px] font-semibold text-emerald-300/90';
        verdictBox.className = 'border-2 border-emerald-500/50 rounded-lg p-3 bg-emerald-500/10 flex flex-col justify-between';
    }

    // 2. Confidence Score Display
    const confScoreEl = document.getElementById('res-confidence-score');
    confScoreEl.textContent = `${confidence.toFixed(1)}%`;

    const confBar = document.getElementById('res-confidence-bar');
    if (confBar) {
        confBar.style.width = `${confidence}%`;
        confBar.className = `h-full rounded-full transition-all duration-500 ${willCancel ? 'bg-rose-500' : 'bg-emerald-500'}`;
    }

    // 3. Recommended Operational Action (MAIN FOCUS)
    const actionEl = document.getElementById('res-recommended-action');
    const priorityEl = document.getElementById('res-action-priority');
    const actionContainer = document.getElementById('res-action-container');

    const actionText = data.recommended_action || (willCancel
        ? '🚨 Re-dispatch Closer Driver & Credit Waiting Token: High ETA delay detected. Reassign a closer driver (< 3 km radius) and send an automated waiting credit to retain the passenger.'
        : '✅ Proceed With Standard Dispatch: Trip health metrics are optimal. Standard automated dispatch and tracking active.');

    const priorityText = data.action_priority || (willCancel ? 'Critical Intervention' : 'Routine Dispatch');
    actionEl.textContent = actionText;
    priorityEl.textContent = priorityText;

    if (willCancel) {
        actionContainer.className = 'bg-rose-500/10 border-2 border-rose-500/50 rounded-xl p-4 space-y-2.5 shadow-lg shadow-rose-950/20';
        priorityEl.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-rose-500/20 border border-rose-500/40 text-rose-400';
    } else if (data.risk_category === 'Medium') {
        actionContainer.className = 'bg-amber-500/10 border-2 border-amber-500/50 rounded-xl p-4 space-y-2.5 shadow-lg shadow-amber-950/20';
        priorityEl.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-500/20 border border-amber-500/40 text-amber-400';
    } else {
        actionContainer.className = 'bg-emerald-500/10 border-2 border-emerald-500/50 rounded-xl p-4 space-y-2.5 shadow-lg shadow-emerald-950/20';
        priorityEl.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-500/20 border border-emerald-500/40 text-emerald-400';
    }

    // 4. Metadata
    let riskBadgeClass = 'badge-low';
    if (data.risk_category === 'Medium') riskBadgeClass = 'badge-medium';
    else if (data.risk_category === 'High') riskBadgeClass = 'badge-high';

    const badge = document.getElementById('res-risk-badge');
    badge.textContent = `${data.risk_category} Risk`;
    badge.className = `px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${riskBadgeClass}`;

    document.getElementById('res-raw-prob').textContent = `${probPct.toFixed(1)}%`;
    document.getElementById('res-model-used').textContent = data.model_used;

    // 5. Local SHAP Attribution
    if (data.shap_explanation && data.shap_explanation.top_factors) {
        const top = [...data.shap_explanation.top_factors].reverse();
        const features = top.map(t => t.feature);
        const shaps = top.map(t => t.shap_value);
        const colors = shaps.map(s => (s > 0 ? '#EF4444' : '#22C55E'));

        const shapTrace = [{
            x: shaps,
            y: features,
            type: 'bar',
            orientation: 'h',
            marker: { color: colors }
        }];
        const shapLayout = {
            ...PLOTLY_THEME,
            height: 160,
            margin: { l: 140, r: 25, t: 10, b: 35 },
            xaxis: { ...PLOTLY_THEME.xaxis, type: 'linear', title: 'Impact (SHAP Value)' },
            yaxis: { ...PLOTLY_THEME.yaxis, type: 'category', automargin: true }
        };
        Plotly.newPlot('chart-local-shap', shapTrace, shapLayout, PLOTLY_CONFIG);

        const narrative = document.getElementById('res-shap-narrative');
        narrative.textContent = data.shap_explanation.summary_text || 'Local game-theoretic factors computed.';
    }
}

function initBatchPrediction() {
    const fileInput = document.getElementById('batch-file-input');
    if (!fileInput) return;

    fileInput.addEventListener('change', async (e) => {
        if (!e.target.files.length) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/predict/batch', { method: 'POST', body: formData });
            if (!res.ok) {
                const errData = await res.json();
                alert(errData.detail || 'Batch prediction failed.');
                return;
            }
            const data = await res.json();

            document.getElementById('batch-results-area').classList.remove('hidden');
            document.getElementById('batch-summary-text').textContent =
                `Scored ${data.total_records.toLocaleString()} records: ${data.predicted_cancelled} Cancelled, ${data.predicted_completed} Completed (Avg Cancellation Prob: ${data.avg_cancellation_probability}%).`;

            const tbody = document.querySelector('#batch-results-table tbody');
            tbody.innerHTML = '';
            data.sample_preview.forEach(row => {
                const tr = document.createElement('tr');
                const willCanc = (row.Will_Cancel === 'YES' || row.Predicted_Status === 'Cancelled');
                const conf = row.Confidence_Score ? `${row.Confidence_Score}%` : `${(Math.max(row.Cancellation_Probability, 1 - row.Cancellation_Probability) * 100).toFixed(1)}%`;
                const act = row.Recommended_Action || (willCanc ? 'Reassign driver to reduce wait time' : 'Standard dispatch monitoring');

                let riskClass = 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/30';
                if (row.Risk_Category === 'Medium') riskClass = 'text-amber-400 bg-amber-500/10 border border-amber-500/30';
                else if (row.Risk_Category === 'High') riskClass = 'text-rose-400 bg-rose-500/10 border border-rose-500/30';

                tr.innerHTML = `
                    <td class="p-2.5 font-medium text-white">${row.Vehicle_Type || '--'}</td>
                    <td class="p-2.5 text-rg-muted truncate max-w-[120px]">${row.Pickup_Location || '--'}</td>
                    <td class="p-2.5">
                        <span class="px-2 py-0.5 rounded font-black text-[11px] ${willCanc ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'}">
                            ${willCanc ? 'YES' : 'NO'}
                        </span>
                    </td>
                    <td class="p-2.5 font-mono font-bold text-white">${conf}</td>
                    <td class="p-2.5">
                        <span class="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${riskClass}">
                            ${row.Risk_Category}
                        </span>
                    </td>
                    <td class="p-2.5 text-[11px] text-rg-cyan font-medium leading-snug">
                        ${act}
                    </td>
                `;
                tbody.appendChild(tr);
            });
        } catch (err) {
            console.error('Batch prediction error', err);
        }
    });
}

// -------------------------------------------------------------
// VIEW 4: MODEL PERFORMANCE BENCHMARK
// -------------------------------------------------------------
async function loadModelPerformance() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();

        const emptyState = document.getElementById('models-empty-state');
        const contentArea = document.getElementById('models-content');

        if (data.status === 'not_trained' || !data.leaderboard || !data.leaderboard.length) {
            emptyState.classList.remove('hidden');
            contentArea.classList.add('hidden');
            return;
        }

        emptyState.classList.add('hidden');
        contentArea.classList.remove('hidden');
        g_modelsData = data;

        // 1. Model Cards Grid
        const grid = document.getElementById('model-cards-grid');
        grid.innerHTML = '';
        data.leaderboard.forEach(m => {
            const card = document.createElement('div');
            card.className = `model-card ${m.is_champion ? 'champion' : ''}`;
            card.innerHTML = `
                <div class="flex justify-between items-center mb-2">
                    <span class="font-bold text-xs text-white">${m.model}</span>
                    ${m.is_champion ? '<span class="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-rg-success/20 text-rg-success border border-rg-success/30">Champion</span>' : ''}
                </div>
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <div>
                        <div class="text-[10px] text-rg-dim">ROC-AUC</div>
                        <div class="font-mono text-rg-cyan font-bold">${m.roc_auc.toFixed(3)}</div>
                    </div>
                    <div>
                        <div class="text-[10px] text-rg-dim">F1 Score</div>
                        <div class="font-mono text-rg-primary font-bold">${m.f1_score.toFixed(3)}</div>
                    </div>
                    <div>
                        <div class="text-[10px] text-rg-dim">Accuracy</div>
                        <div class="font-mono text-rg-muted">${(m.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                        <div class="text-[10px] text-rg-dim">Recall</div>
                        <div class="font-mono text-rg-muted">${(m.recall * 100).toFixed(1)}%</div>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });

        // 2. Comparison Bar Chart
        const models = data.leaderboard.map(d => d.model);
        const f1s = data.leaderboard.map(d => d.f1_score);
        const aucs = data.leaderboard.map(d => d.roc_auc);

        const bTraces = [
            { x: models, y: f1s, name: 'F1 Score', type: 'bar', marker: { color: '#EAB308' } },
            { x: models, y: aucs, name: 'ROC-AUC', type: 'bar', marker: { color: '#FACC15' } }
        ];
        const bLayout = { ...PLOTLY_THEME, barmode: 'group', legend: { x: 0.02, y: 0.98 } };
        Plotly.newPlot('chart-models-bar', bTraces, bLayout, PLOTLY_CONFIG);

        // 3. Multi-Model ROC Curves
        const rocTraces = [];
        const colors = ['#EAB308', '#16A34A', '#F97316', '#6366F1'];
        let idx = 0;
        for (const [name, curve] of Object.entries(data.roc_curves)) {
            rocTraces.push({
                x: curve.fpr,
                y: curve.tpr,
                mode: 'lines',
                name: `${name} (${curve.auc.toFixed(2)})`,
                line: { color: colors[idx % colors.length], width: 2 }
            });
            idx++;
        }
        rocTraces.push({
            x: [0, 1],
            y: [0, 1],
            mode: 'lines',
            name: 'Random Chance',
            line: { dash: 'dash', color: '#64748B', width: 1.5 }
        });

        const rocLayout = {
            ...PLOTLY_THEME,
            xaxis: { ...PLOTLY_THEME.xaxis, title: 'False Positive Rate' },
            yaxis: { ...PLOTLY_THEME.yaxis, title: 'True Positive Rate' },
            legend: { x: 0.45, y: 0.15 }
        };
        Plotly.newPlot('chart-roc-curves', rocTraces, rocLayout, PLOTLY_CONFIG);

        // 4. Confusion Matrix Initial Render
        const cmSelect = document.getElementById('cm-model-select');
        if (cmSelect) {
            cmSelect.value = data.champion;
            renderConfusionMatrix(data.champion);
            cmSelect.onchange = (e) => renderConfusionMatrix(e.target.value);
        }

        setTimeout(() => {
            ['chart-models-bar', 'chart-roc-curves', 'chart-confusion-matrix'].forEach(id => {
                const el = document.getElementById(id);
                if (el && el.classList.contains('js-plotly-plot')) {
                    Plotly.Plots.resize(el);
                }
            });
        }, 150);

    } catch (err) {
        console.error('Failed to load model performance', err);
    }
}

function renderConfusionMatrix(modelName) {
    if (!g_modelsData || !g_modelsData.confusion_matrices || !g_modelsData.confusion_matrices[modelName]) return;
    const cm = g_modelsData.confusion_matrices[modelName];

    // Scikit-learn confusion matrix orientation:
    // [[TN, FP],
    //  [FN, TP]]
    const tn = Number(cm[0][0]);
    const fp = Number(cm[0][1]);
    const fn = Number(cm[1][0]);
    const tp = Number(cm[1][1]);

    // Update 4-quadrant metric cards
    const setTxt = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val.toLocaleString(); };
    setTxt('cm-stat-tn', tn);
    setTxt('cm-stat-fp', fp);
    setTxt('cm-stat-fn', fn);
    setTxt('cm-stat-tp', tp);

    // Heatmap labels & values
    const xLabels = ['Pred: Completed (0)', 'Pred: Cancelled (1)'];
    const yLabels = ['True: Completed (0)', 'True: Cancelled (1)'];
    const zValues = [
        [tn, fp],
        [fn, tp]
    ];

    const annotations = [
        {
            x: 'Pred: Completed (0)',
            y: 'True: Completed (0)',
            text: `<b>${tn}</b><br><span style="font-size:10px; color:#15803D">True Negative</span>`,
            font: { color: '#0F172A', size: 16, family: 'Inter, sans-serif' },
            showarrow: false
        },
        {
            x: 'Pred: Cancelled (1)',
            y: 'True: Completed (0)',
            text: `<b>${fp}</b><br><span style="font-size:10px; color:#A16207">False Alarm</span>`,
            font: { color: '#0F172A', size: 16, family: 'Inter, sans-serif' },
            showarrow: false
        },
        {
            x: 'Pred: Completed (0)',
            y: 'True: Cancelled (1)',
            text: `<b>${fn}</b><br><span style="font-size:10px; color:#B91C1C">Missed Cancel</span>`,
            font: { color: '#0F172A', size: 16, family: 'Inter, sans-serif' },
            showarrow: false
        },
        {
            x: 'Pred: Cancelled (1)',
            y: 'True: Cancelled (1)',
            text: `<b>${tp}</b><br><span style="font-size:10px; color:#854D0E">True Positive</span>`,
            font: { color: '#0F172A', size: 16, family: 'Inter, sans-serif' },
            showarrow: false
        }
    ];

    const cmTrace = [{
        z: zValues,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        hoverongaps: false,
        colorscale: [
            [0, '#FEFCE8'],
            [0.35, '#FEF08A'],
            [0.7, '#FACC15'],
            [1, '#EAB308']
        ],
        showscale: true,
        colorbar: {
            title: { text: 'Trips', font: { color: '#475569', size: 11 } },
            tickfont: { color: '#475569', size: 10 },
            len: 0.9,
            thickness: 14
        }
    }];

    const cmLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(248, 250, 252, 0.7)',
        font: { family: 'Inter, sans-serif', color: '#475569', size: 11 },
        height: 290,
        margin: { l: 145, r: 40, t: 25, b: 60 },
        xaxis: {
            title: { text: 'Predicted Class', font: { size: 12, color: '#0F172A' }, standoff: 12 },
            type: 'category',
            tickfont: { color: '#0F172A', size: 11, family: 'Inter, sans-serif' },
            gridcolor: '#E2E8F0',
            linecolor: '#CBD5E1'
        },
        yaxis: {
            title: { text: 'Actual Ground Truth', font: { size: 12, color: '#0F172A' }, standoff: 15 },
            type: 'category',
            autorange: 'reversed',
            tickfont: { color: '#0F172A', size: 11, family: 'Inter, sans-serif' },
            gridcolor: '#E2E8F0',
            linecolor: '#CBD5E1'
        },
        annotations: annotations
    };

    Plotly.newPlot('chart-confusion-matrix', cmTrace, cmLayout, PLOTLY_CONFIG).then(() => {
        setTimeout(() => {
            const el = document.getElementById('chart-confusion-matrix');
            if (el && el.classList.contains('js-plotly-plot')) {
                Plotly.Plots.resize(el);
            }
        }, 50);
    });
}

function initModelRetrain() {
    const btn = document.getElementById('btn-retrain-models');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        btn.textContent = 'Retraining...';
        btn.disabled = true;
        try {
            await fetch('/api/train', { method: 'POST' });
            await loadTelemetry();
            await loadModelPerformance();
            await loadPredictionSchema();
            btn.innerHTML = '<i data-lucide="rotate-cw" class="w-3.5 h-3.5"></i><span>Retrain Models</span>';
            btn.disabled = false;
            if (window.lucide) lucide.createIcons();
        } catch (err) {
            console.error('Retrain failed', err);
            btn.textContent = 'Retrain Failed';
            btn.disabled = false;
        }
    });
}

// -------------------------------------------------------------
// VIEW 5: EXPLAINABLE AI (SHAP)
// -------------------------------------------------------------
async function loadExplainability() {
    try {
        const res = await fetch('/api/explainability');
        const data = await res.json();

        const emptyState = document.getElementById('explainable-empty-state');
        const contentArea = document.getElementById('explainable-content');

        if (data.status === 'not_trained' || !data.global_importance || !data.global_importance.length) {
            emptyState.classList.remove('hidden');
            contentArea.classList.add('hidden');
            return;
        }

        emptyState.classList.add('hidden');
        contentArea.classList.remove('hidden');

        // 1. Global Bar Chart
        const sorted = [...data.global_importance].reverse();
        const features = sorted.map(d => String(d.feature));
        const imps = sorted.map(d => Number(d.importance));

        const gTrace = [{
            x: imps,
            y: features,
            type: 'bar',
            orientation: 'h',
            marker: {
                color: '#EAB308',
                line: { color: '#CA8A04', width: 1 }
            },
            text: imps.map(v => v.toFixed(4)),
            textposition: 'auto',
            textfont: { color: '#0F172A', size: 10, family: 'Inter, monospace', weight: 600 }
        }];

        const gLayout = {
            ...PLOTLY_THEME,
            height: Math.max(380, features.length * 28),
            margin: { l: 160, r: 40, t: 15, b: 50 },
            xaxis: {
                ...PLOTLY_THEME.xaxis,
                title: { text: 'Mean |SHAP Value| (Impact on Cancellation Probability)', font: { size: 11, color: '#475569' }, standoff: 10 },
                type: 'linear'
            },
            yaxis: {
                ...PLOTLY_THEME.yaxis,
                type: 'category',
                automargin: true,
                tickfont: { color: '#0F172A', size: 11, family: 'Inter, sans-serif', weight: 500 }
            }
        };

        Plotly.newPlot('chart-global-shap', gTrace, gLayout, PLOTLY_CONFIG).then(() => {
            setTimeout(() => {
                const el = document.getElementById('chart-global-shap');
                if (el && el.classList.contains('js-plotly-plot')) {
                    Plotly.Plots.resize(el);
                }
            }, 60);
        });

        // 2. Table
        const tbody = document.querySelector('#table-shap-drivers tbody');
        tbody.innerHTML = '';
        data.global_importance.forEach((item, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="py-2 text-rg-dim">#${idx + 1}</td>
                <td class="py-2 font-medium text-white">${item.feature}</td>
                <td class="py-2 font-mono text-rg-cyan">${item.importance.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Failed to load SHAP explainability', err);
    }
}

// -------------------------------------------------------------
// VIEW 6: POLICY RECOMMENDATIONS
// -------------------------------------------------------------
async function loadRecommendations() {
    try {
        const res = await fetch('/api/recommendations');
        const data = await res.json();

        const emptyState = document.getElementById('recommendations-empty-state');
        const contentArea = document.getElementById('recommendations-content');

        if (data.status === 'no_dataset' || !data.recommendations || !data.recommendations.length) {
            emptyState.classList.remove('hidden');
            contentArea.classList.add('hidden');
            return;
        }

        emptyState.classList.add('hidden');
        contentArea.classList.remove('hidden');
        g_recommendations = data.recommendations;

        renderRecommendationsList('all');

        const buttons = document.querySelectorAll('.rec-filter-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                buttons.forEach(b => {
                    b.className = 'rec-filter-btn px-3 py-1 rounded text-rg-muted hover:text-white';
                });
                btn.className = 'rec-filter-btn px-3 py-1 rounded bg-rg-primary text-white';
                const filter = btn.getAttribute('data-filter');
                renderRecommendationsList(filter);
            });
        });
    } catch (err) {
        console.error('Failed to load recommendations', err);
    }
}

function renderRecommendationsList(filter) {
    const container = document.getElementById('recommendations-container');
    container.innerHTML = '';

    const list = g_recommendations.filter(r => {
        if (filter === 'all') return true;
        return r.priority.toLowerCase() === filter.toLowerCase();
    });

    if (!list.length) {
        container.innerHTML = `<div class="text-xs text-rg-muted p-4 bg-rg-card rounded border border-rg-border">No recommendations matching priority tier "${filter}".</div>`;
        return;
    }

    list.forEach(rec => {
        const card = document.createElement('div');
        card.className = `policy-card priority-${rec.priority}`;
        card.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <span class="text-[11px] font-bold text-yellow-700 uppercase tracking-wider">${rec.category} &bull; ${rec.id}</span>
                <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${rec.priority === 'Critical' || rec.priority === 'High' ? 'badge-high' : 'badge-medium'}">${rec.priority} Priority</span>
            </div>
            <h3 class="text-sm font-bold text-white mb-2">${rec.title}</h3>
            <div class="bg-rg-elevated border border-rg-border rounded p-3 text-xs text-rg-muted space-y-1 mb-3">
                <div><strong class="text-white">Identified Pattern:</strong> ${rec.pattern}</div>
                <div><strong class="text-white">Supporting Evidence:</strong> ${rec.evidence}</div>
            </div>
            <div class="text-xs text-rg-text mb-3 leading-relaxed">
                <strong class="text-rg-primary">Recommended Action:</strong> ${rec.recommended_action}
            </div>
            <div class="flex justify-between items-center border-t border-rg-border pt-2 text-[11px]">
                <span class="text-rg-success font-medium"><strong>Potential Benefit:</strong> ${rec.expected_benefit}</span>
                <span class="text-rg-dim italic">${rec.causal_disclaimer}</span>
            </div>
        `;
        container.appendChild(card);
    });
}

// -------------------------------------------------------------
// VIEW 7: DATASET INGESTION, VALIDATION & SCHEMA CENTER
// -------------------------------------------------------------
async function loadDatasetInfo() {
    try {
        const res = await fetch('/api/dataset');
        if (!res.ok) {
            console.warn('/api/dataset returned status', res.status);
            return;
        }
        const data = await res.json();
        g_activeDataset = data;

        const activeCard = document.getElementById('active-dataset-card');

        if (!data || !data.has_dataset) {
            if (activeCard) activeCard.classList.add('hidden');
            return;
        }

        if (activeCard) activeCard.classList.remove('hidden');

        // Top info
        const setTxt = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
        setTxt('active-dataset-name', data.filename || 'Uploaded Dataset');
        setTxt('active-file-size', data.filesize_bytes ? `${(data.filesize_bytes / 1024).toFixed(1)} KB` : '-- KB');
        const totalRecs = data.total_records ?? 0;
        const totalCols = data.total_columns ?? 0;
        setTxt('active-row-count', `${totalRecs.toLocaleString()} rows`);
        setTxt('active-col-count', `${totalCols} columns`);
        setTxt('active-target-display', `Target: ${data.target_column || 'Not Mapped'}`);

        setTxt('ds-total-rows', totalRecs.toLocaleString());
        setTxt('ds-total-cols', totalCols);

        // Fetch validation report
        const valRes = await fetch('/api/dataset/validation');
        if (!valRes.ok) return;
        const valReport = await valRes.json();

        // Validation badge
        const badge = document.getElementById('active-validation-badge');
        badge.textContent = valReport.status;
        if (valReport.status === 'Valid') {
            badge.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-500/10 border border-emerald-500/30 text-emerald-400';
        } else if (valReport.status === 'Valid with warnings') {
            badge.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-500/10 border border-amber-500/30 text-amber-400';
        } else {
            badge.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-rose-500/10 border border-rose-500/30 text-rose-400';
        }

        document.getElementById('ds-duplicates').textContent = (valReport.duplicate_records ?? 0).toLocaleString();
        document.getElementById('ds-missing-cells').textContent = (valReport.missing_cells ?? 0).toLocaleString();

        // Warnings list
        const warnBox = document.getElementById('validation-warnings-container');
        const warnList = document.getElementById('validation-warnings-list');
        if (valReport.warnings && valReport.warnings.length > 0) {
            warnBox.classList.remove('hidden');
            warnList.innerHTML = '';
            valReport.warnings.forEach(w => {
                const li = document.createElement('li');
                li.textContent = w;
                warnList.appendChild(li);
            });
        } else {
            warnBox.classList.add('hidden');
        }

        // Schema breakdown table
        const tbodySchema = document.getElementById('schema-table-body');
        tbodySchema.innerHTML = '';
        if (valReport.schema) {
            valReport.schema.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="p-2.5 font-medium text-white">${row.column}</td>
                    <td class="p-2.5 text-rg-dim">${row.type}</td>
                    <td class="p-2.5">${row.missing_count}</td>
                    <td class="p-2.5">${row.missing_pct}%</td>
                    <td class="p-2.5">${row.unique_values}</td>
                    <td class="p-2.5 text-rg-cyan truncate max-w-[140px]">${row.sample}</td>
                `;
                tbodySchema.appendChild(tr);
            });
        }

        // Target Distribution Tags
        const tagsContainer = document.getElementById('target-distribution-tags');
        tagsContainer.innerHTML = '';
        if (valReport.target_distribution) {
            for (const [cls, cnt] of Object.entries(valReport.target_distribution)) {
                const pill = document.createElement('span');
                pill.className = 'px-2 py-0.5 rounded text-[11px] font-mono bg-rg-card border border-rg-border text-white';
                pill.textContent = `${cls}: ${cnt.toLocaleString()}`;
                tagsContainer.appendChild(pill);
            }
        }

        // Training Status Pill
        const trainPill = document.getElementById('training-status-pill');
        if (data.models_trained) {
            trainPill.textContent = `Trained (${data.champion_model})`;
            trainPill.className = 'font-semibold text-rg-success font-mono';
        } else {
            trainPill.textContent = 'Awaiting Training';
            trainPill.className = 'font-semibold text-rg-warning font-mono';
        }

        // Render Schema Mapping Dropdowns
        renderSchemaMapping(data.column_mapping, valReport.schema ? valReport.schema.map(s => s.column) : []);

        // Load preview
        loadRawPreview();

    } catch (err) {
        console.error('Failed to load dataset details', err);
    }
}

function renderSchemaMapping(currentMapping, availableColumns) {
    const container = document.getElementById('mapping-dropdowns-container');
    container.innerHTML = '';

    const standardFields = [
        { id: 'booking_status', label: 'Booking Status (Target Variable)', highlight: true },
        { id: 'vehicle_type', label: 'Vehicle Category' },
        { id: 'pickup_location', label: 'Pickup Location / Hub' },
        { id: 'drop_location', label: 'Drop Destination Hub' },
        { id: 'ride_distance', label: 'Ride Distance (km)' },
        { id: 'booking_value', label: 'Booking Fare / Amount' },
        { id: 'waiting_time', label: 'Waiting Time (mins)' },
        { id: 'driver_arrival_time', label: 'Driver Arrival Time (ETA)' },
        { id: 'booking_date', label: 'Booking Date' },
        { id: 'booking_time', label: 'Booking Time' },
        { id: 'cancellation_reason', label: 'Cancellation Reason (Leakage Shield)' }
    ];

    standardFields.forEach(f => {
        const div = document.createElement('div');
        div.className = f.highlight ? 'bg-blue-500/10 border border-blue-500/30 rounded p-2.5' : 'bg-rg-elevated border border-rg-border rounded p-2.5';
        
        let selectHtml = `<select id="map-${f.id}" class="w-full mt-1 bg-rg-card border border-rg-border rounded px-2 py-1 text-white text-xs outline-none focus:border-rg-primary">`;
        selectHtml += `<option value="">-- Not in Dataset / None --</option>`;
        
        const mappedCol = currentMapping ? currentMapping[f.id] : null;

        availableColumns.forEach(col => {
            const isSelected = (mappedCol === col) ? 'selected' : '';
            selectHtml += `<option value="${col}" ${isSelected}>${col}</option>`;
        });
        selectHtml += `</select>`;

        div.innerHTML = `
            <label class="block text-[11px] font-medium ${f.highlight ? 'text-blue-300 font-bold' : 'text-rg-muted'}">${f.label}</label>
            ${selectHtml}
        `;
        container.appendChild(div);
    });
}

async function loadRawPreview() {
    try {
        const res = await fetch('/api/dataset/preview');
        if (!res.ok) return;
        const data = await res.json();

        const thead = document.getElementById('preview-table-head');
        const tbody = document.getElementById('preview-table-body');
        thead.innerHTML = '';
        tbody.innerHTML = '';

        if (!data.columns || !data.preview || !data.preview.length) return;

        // Head row
        const trHead = document.createElement('tr');
        data.columns.forEach(col => {
            const th = document.createElement('th');
            th.className = 'p-2.5 border-b border-rg-border';
            th.textContent = col;
            trHead.appendChild(th);
        });
        thead.appendChild(trHead);

        // Body rows
        data.preview.forEach(row => {
            const tr = document.createElement('tr');
            data.columns.forEach(col => {
                const td = document.createElement('td');
                td.className = 'p-2.5';
                const val = row[col];
                td.textContent = (val !== null && val !== undefined) ? String(val) : '';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error('Failed to load raw preview', err);
    }
}

function initDatasetManager() {
    const dropzone = document.getElementById('csv-dropzone');
    const fileInput = document.getElementById('dataset-file-input');
    const btnSample = document.getElementById('btn-load-sample');
    const btnRemove = document.getElementById('btn-remove-dataset');
    const btnSaveMap = document.getElementById('btn-save-mapping');
    const btnTrain = document.getElementById('btn-train-active-dataset');

    // Click on dropzone opens file input
    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('border-rg-primary', 'bg-blue-500/5');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('border-rg-primary', 'bg-blue-500/5');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('border-rg-primary', 'bg-blue-500/5');
            if (e.dataTransfer.files.length) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    // Load sample button
    if (btnSample) {
        btnSample.addEventListener('click', async () => {
            btnSample.disabled = true;
            btnSample.textContent = 'Loading Sample...';
            try {
                const res = await fetch('/api/dataset/sample', { method: 'POST' });
                if (!res.ok) {
                    const err = await res.json();
                    alert(err.detail || 'Failed to load sample dataset.');
                    return;
                }
                await loadTelemetry();
                await loadDatasetInfo();
                await loadPredictionSchema();
                alert('Historical sample dataset (1,000 rides) successfully connected! Please click "Train Machine Learning Models" to train.');
            } catch (err) {
                console.error('Failed to load sample', err);
            } finally {
                btnSample.disabled = false;
                btnSample.innerHTML = '<i data-lucide="file-text" class="w-3.5 h-3.5 text-rg-cyan"></i><span>Load Repository Sample (ride_bookings.csv)</span>';
                if (window.lucide) lucide.createIcons();
            }
        });
    }

    // Remove dataset button
    if (btnRemove) {
        btnRemove.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to remove the active dataset? This will clear all trained models and statistics.')) return;
            try {
                await fetch('/api/dataset/remove', { method: 'POST' });
                await loadTelemetry();
                await loadDatasetInfo();
                await loadPredictionSchema();
                alert('Dataset removed. System returned to clean empty state.');
            } catch (err) {
                console.error('Failed to remove dataset', err);
            }
        });
    }

    // Save Mapping button
    if (btnSaveMap) {
        btnSaveMap.addEventListener('click', async () => {
            const mapping = {};
            const standardFields = [
                'booking_status', 'vehicle_type', 'pickup_location', 'drop_location',
                'ride_distance', 'booking_value', 'waiting_time', 'driver_arrival_time',
                'booking_date', 'booking_time', 'cancellation_reason'
            ];

            standardFields.forEach(f => {
                const sel = document.getElementById(`map-${f}`);
                if (sel && sel.value) {
                    mapping[f] = sel.value;
                }
            });

            try {
                const res = await fetch('/api/dataset/mapping', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mapping: mapping, target_col: mapping['booking_status'] })
                });
                if (!res.ok) {
                    const err = await res.json();
                    alert(err.detail || 'Failed to save mapping.');
                    return;
                }
                await loadDatasetInfo();
                alert('Column mapping saved and dataset successfully re-validated!');
            } catch (err) {
                console.error('Failed to save mapping', err);
            }
        });
    }

    // Train button
    if (btnTrain) {
        btnTrain.addEventListener('click', async () => {
            btnTrain.disabled = true;
            btnTrain.innerHTML = '<span class="inline-block animate-spin mr-2">&#9696;</span><span>Training 4 ML Models (Logistic Regression, Decision Tree, Random Forest, XGBoost)...</span>';

            try {
                const res = await fetch('/api/train', { method: 'POST' });
                if (!res.ok) {
                    const err = await res.json();
                    alert(err.detail || 'Model training failed.');
                    return;
                }
                const data = await res.json();
                await loadTelemetry();
                await loadDatasetInfo();
                await loadPredictionSchema();
                alert(`Machine learning training complete! Champion Model: ${data.champion}. All 4 models trained and ready.`);
                switchToTab('view-overview', 'Executive Overview');
            } catch (err) {
                console.error('Training failed', err);
                alert('Training failed. Please check column mapping and target variable.');
            } finally {
                btnTrain.disabled = false;
                btnTrain.innerHTML = '<i data-lucide="play" class="w-4 h-4 fill-white"></i><span>Train Machine Learning Models</span>';
                if (window.lucide) lucide.createIcons();
            }
        });
    }
}

async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        alert('Please select a valid CSV (.csv) file.');
        return;
    }

    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const progressPct = document.getElementById('upload-progress-pct');

    progressContainer.classList.remove('hidden');
    progressBar.style.width = '30%';
    progressPct.textContent = '30%';

    const formData = new FormData();
    formData.append('file', file);

    try {
        progressBar.style.width = '60%';
        progressPct.textContent = '60%';

        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || 'Upload failed.');
            return;
        }

        progressBar.style.width = '100%';
        progressPct.textContent = '100%';

        const data = await res.json().catch(() => ({}));
        await loadTelemetry();
        await loadDatasetInfo();

        setTimeout(() => {
            progressContainer.classList.add('hidden');
            progressBar.style.width = '0%';
            progressPct.textContent = '0%';
        }, 800);

        const rows = data.total_records ?? (data.shape ? data.shape[0] : 0);
        const cols = data.total_columns ?? (data.shape ? data.shape[1] : 0);
        alert(`Successfully uploaded "${file.name}" (${rows.toLocaleString()} rows, ${cols} columns). Please review the Schema Mapping and click "Train Machine Learning Models" to activate predictions and analytics.`);
    } catch (err) {
        console.error('Upload failed', err);
        alert('Error uploading dataset.');
        progressContainer.classList.add('hidden');
    }
}

// Global responsive resize listener for Plotly charts
window.addEventListener('resize', () => {
    document.querySelectorAll('.js-plotly-plot').forEach(c => {
        try { Plotly.Plots.resize(c); } catch (e) {}
    });
});
