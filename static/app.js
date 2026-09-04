// ---------- Tabs ----------
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

const GOLD = "#d6aa4d";
const NAVY = "#4f789e";
const GREEN = "#45d28a";
const RED = "#ff6b65";
const PALETTE = ["#d6aa4d","#4f789e","#45d28a","#ff6b65","#7c6bb1","#8a9bae","#c07b55","#5b9a9a"];
if (typeof Chart !== "undefined") {
  Chart.defaults.color = "#8fa2b8";
  Chart.defaults.font.family = "Cairo, sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = "rgba(255,255,255,.07)";
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
}


async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

// ---------- Boot ----------
(async function init() {
  const statusEl = document.getElementById("apiStatus");
  try {
    await apiGet("/health");
    statusEl.textContent = "متصل بالـ API ✓";
    statusEl.classList.add("ok"); document.querySelector(".live-dot")?.style.setProperty("background", "#45d28a"); document.querySelector(".live-dot")?.style.setProperty("boxShadow", "0 0 0 5px rgba(69,210,138,.10), 0 0 14px rgba(69,210,138,.55)");
  } catch (e) {
    statusEl.textContent = "مش قادر أوصل للـ API — تأكد إنه شغال على " + API_BASE;
    statusEl.classList.add("err"); document.querySelector(".live-dot")?.style.setProperty("background", "#ff6b65");
    return;
  }

  // Load in order: insights needs the stat cards created by loadModels().
  // Running these three requests in parallel can cause a race condition
  // where /insights finishes before /models and the charts never render.
  await loadModels();
  await loadFeatures();
  await loadInsights();
})();

// ---------- Chart helpers ----------
const chartInstances = {};

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

const GRID = "rgba(255,255,255,.055)";
const TICK = "#8fa2b8";
const TOOLTIP = {
  backgroundColor: "#07111f",
  titleColor: "#f4f7fb",
  bodyColor: "#c7d3df",
  borderColor: "rgba(214,170,77,.30)",
  borderWidth: 1,
  padding: 12,
  cornerRadius: 10,
  displayColors: true,
};

const AXIS = {
  x: {
    beginAtZero: true,
    grid: { color: GRID, drawBorder: false },
    ticks: { color: TICK, padding: 8 },
    border: { display: false }
  },
  y: {
    beginAtZero: true,
    grid: { color: GRID, drawBorder: false },
    ticks: { color: TICK, padding: 8 },
    border: { display: false }
  }
};

function makeBarDataset(label, data, color, radius = 7) {
  return {
    label,
    data,
    backgroundColor: color,
    borderColor: color,
    borderWidth: 0,
    borderRadius: radius,
    borderSkipped: false,
    maxBarThickness: 34,
    hoverBackgroundColor: GOLD
  };
}

function chartOptions(extra = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 650, easing: "easeOutQuart" },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        position: "top",
        align: "end",
        labels: { color: "#aebdcb", padding: 16, usePointStyle: true, pointStyle: "circle" }
      },
      tooltip: TOOLTIP
    },
    ...extra
  };
}

// ---------- Models tab ----------
async function loadModels() {
  try {
    const models = await apiGet("/models");
    const tbody = document.querySelector("#modelsTable tbody");
    tbody.innerHTML = "";

    models.forEach((m, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>${m.Model}</td>
        <td>${m.F1}</td>
        <td>${m.Accuracy}</td>
        <td>${m.Precision}</td>
        <td>${m.Recall}</td>
        <td>${m.ROC_AUC ?? "—"}</td>`;
      tbody.appendChild(tr);
    });

    destroyChart("modelsChart");
    chartInstances.modelsChart = new Chart(document.getElementById("modelsChart"), {
      type: "bar",
      data: {
        labels: models.map(m => m.Model),
        datasets: [
          makeBarDataset("F1 Score", models.map(m => m.F1), GOLD),
          makeBarDataset("Accuracy", models.map(m => m.Accuracy), NAVY)
        ]
      },
      options: chartOptions({
        plugins: {
          legend: {
            position: "top",
            align: "end",
            labels: { color: "#aebdcb", padding: 18, usePointStyle: true }
          },
          tooltip: {
            ...TOOLTIP,
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`
            }
          }
        },
        scales: {
          x: { ...AXIS.x, grid: { display: false }, ticks: { ...AXIS.x.ticks, maxRotation: 35, minRotation: 0 } },
          y: { ...AXIS.y, max: 1, ticks: { ...AXIS.y.ticks, callback: v => `${Math.round(v * 100)}%` } }
        }
      })
    });

    const sel = document.getElementById("modelSelect");
    sel.innerHTML = models.map(m => `<option value="${m.file.replace('.pkl','')}">${m.Model}</option>`).join("");

    document.getElementById("statGrid").innerHTML = `
      <div class="stat-card"><div class="stat-value">${models.length}</div><div class="stat-label">عدد الموديلات المتاحة</div></div>
      <div class="stat-card"><div class="stat-value">${models[0].Model}</div><div class="stat-label">أفضل موديل (F1 = ${models[0].F1})</div></div>
      <div class="stat-card" id="statTotalBookings"><div class="stat-value">—</div><div class="stat-label">إجمالي الحجوزات</div></div>
      <div class="stat-card" id="statCancelRate"><div class="stat-value">—</div><div class="stat-label">نسبة الإلغاء الكلية</div></div>`;
  } catch (e) {
    document.querySelector("#modelsTable tbody").innerHTML =
      `<tr><td colspan="7">مقدرتش أجيب /models — تأكد إن train.py اشتغل وartifacts/ موجودة</td></tr>`;
  }
}

// ---------- Features tab ----------
async function loadFeatures() {
  try {
    const data = await apiGet("/features");
    document.getElementById("featureSource").textContent =
      `مبني على أوزان الموديل: ${data.source_model}`;

    const sorted = data.features;
    destroyChart("featuresChart");
    chartInstances.featuresChart = new Chart(document.getElementById("featuresChart"), {
      type: "bar",
      data: {
        labels: sorted.map(f => f.feature),
        datasets: [{
          label: "Feature Importance",
          data: sorted.map(f => f.importance),
          backgroundColor: sorted.map((_, i) => i === 0 ? GOLD : "rgba(214,170,77,.55)"),
          borderRadius: 6,
          borderSkipped: false,
          maxBarThickness: 24
        }]
      },
      options: chartOptions({
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          tooltip: {
            ...TOOLTIP,
            callbacks: {
              label: ctx => ` Importance: ${Number(ctx.parsed.x).toFixed(4)}`
            }
          }
        },
        scales: {
          x: { ...AXIS.x, ticks: { ...AXIS.x.ticks, callback: v => Number(v).toFixed(3) } },
          y: { ...AXIS.y, grid: { display: false } }
        }
      })
    });
  } catch (e) {
    document.getElementById("featureSource").textContent = "مقدرتش أجيب /features";
  }
}

// ---------- Insights tab + overview charts ----------
async function loadInsights() {
  try {
    const data = await apiGet("/insights");

    document.getElementById("statTotalBookings").querySelector(".stat-value").textContent =
      data.summary_stats.total_bookings.toLocaleString();
    document.getElementById("statCancelRate").querySelector(".stat-value").textContent =
      data.summary_stats.overall_cancellation_rate + "%";

    // Season — elegant doughnut
    const seasonLabels = Object.keys(data.cancel_rate_by_season);
    destroyChart("seasonChart");
    chartInstances.seasonChart = new Chart(document.getElementById("seasonChart"), {
      type: "doughnut",
      data: {
        labels: seasonLabels,
        datasets: [{
          data: Object.values(data.cancel_rate_by_season),
          backgroundColor: PALETTE,
          borderColor: "#0e1c2f",
          borderWidth: 5,
          hoverOffset: 10
        }]
      },
      options: chartOptions({
        cutout: "68%",
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#aebdcb", padding: 16, usePointStyle: true, pointStyle: "circle" }
          },
          tooltip: {
            ...TOOLTIP,
            callbacks: {
              label: ctx => ` ${ctx.label}: ${Number(ctx.raw).toFixed(1)}%`
            }
          }
        }
      })
    });

    // Deposit — sorted descending for easier reading
    const depositEntries = Object.entries(data.cancel_rate_by_deposit_type)
      .sort((a, b) => Number(b[1]) - Number(a[1]));
    destroyChart("depositChart");
    chartInstances.depositChart = new Chart(document.getElementById("depositChart"), {
      type: "bar",
      data: {
        labels: depositEntries.map(x => x[0]),
        datasets: [makeBarDataset("Cancellation Rate", depositEntries.map(x => x[1]), RED, 8)]
      },
      options: chartOptions({
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          tooltip: {
            ...TOOLTIP,
            callbacks: { label: ctx => ` ${Number(ctx.parsed.x).toFixed(1)}%` }
          }
        },
        scales: {
          x: { ...AXIS.x, ticks: { ...AXIS.x.ticks, callback: v => `${v}%` } },
          y: { ...AXIS.y, grid: { display: false } }
        }
      })
    });

    // Market segment — horizontal, sorted
    const segEntries = Object.entries(data.cancel_rate_by_market_segment)
      .sort((a, b) => Number(b[1]) - Number(a[1]));
    destroyChart("segmentChart");
    chartInstances.segmentChart = new Chart(document.getElementById("segmentChart"), {
      type: "bar",
      data: {
        labels: segEntries.map(x => x[0]),
        datasets: [makeBarDataset("Cancellation Rate", segEntries.map(x => x[1]), NAVY, 7)]
      },
      options: chartOptions({
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          tooltip: {
            ...TOOLTIP,
            callbacks: { label: ctx => ` ${Number(ctx.parsed.x).toFixed(1)}%` }
          }
        },
        scales: {
          x: { ...AXIS.x, ticks: { ...AXIS.x.ticks, callback: v => `${v}%` } },
          y: { ...AXIS.y, grid: { display: false } }
        }
      })
    });

    // Lead time — clean trend line with highlighted points
    const leadLabels = Object.keys(data.cancel_rate_by_lead_time_bucket);
    destroyChart("leadTimeChart");
    chartInstances.leadTimeChart = new Chart(document.getElementById("leadTimeChart"), {
      type: "line",
      data: {
        labels: leadLabels,
        datasets: [{
          label: "Cancellation Rate",
          data: Object.values(data.cancel_rate_by_lead_time_bucket),
          borderColor: GOLD,
          backgroundColor: "rgba(214,170,77,.10)",
          fill: true,
          tension: .35,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointBackgroundColor: GOLD,
          pointBorderColor: "#07111f",
          pointBorderWidth: 2
        }]
      },
      options: chartOptions({
        plugins: {
          legend: { display: false },
          tooltip: {
            ...TOOLTIP,
            callbacks: { label: ctx => ` ${Number(ctx.parsed.y).toFixed(1)}%` }
          }
        },
        scales: {
          x: { ...AXIS.x, grid: { display: false } },
          y: { ...AXIS.y, ticks: { ...AXIS.y.ticks, callback: v => `${v}%` } }
        }
      })
    });

    // Chi-square table
    const tbody = document.querySelector("#chiTable tbody");
    tbody.innerHTML = "";
    data.chi_square_results.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.feature}</td>
        <td>${r.chi2}</td>
        <td>${r.p_value < 0.0001 ? "<0.0001" : r.p_value.toFixed(4)}</td>
        <td><span class="badge ${r.significant ? "yes" : "no"}">${r.significant ? "معنوي" : "غير معنوي"}</span></td>`;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error(e);
  }
}

// ---------- Predict tab ----------
const PREDICT_FIELDS = [
  { name: "hotel", label: "نوع الفندق", type: "select", options: ["Resort Hotel", "City Hotel"] },
  { name: "lead_time", label: "عدد أيام الحجز المسبق", type: "number", value: 30 },
  { name: "arrival_date_month", label: "شهر الوصول", type: "select",
    options: ["January","February","March","April","May","June","July","August","September","October","November","December"] },
  { name: "arrival_date_year", label: "سنة الوصول", type: "number", value: 2017 },
  { name: "arrival_date_day_of_month", label: "يوم الوصول", type: "number", value: 15 },
  { name: "arrival_date_week_number", label: "رقم أسبوع الوصول", type: "number", value: 24 },
  { name: "stays_in_weekend_nights", label: "ليالي عطلة نهاية الأسبوع", type: "number", value: 1 },
  { name: "stays_in_week_nights", label: "ليالي أيام الأسبوع", type: "number", value: 2 },
  { name: "adults", label: "عدد البالغين", type: "number", value: 2 },
  { name: "children", label: "عدد الأطفال", type: "number", value: 0 },
  { name: "babies", label: "عدد الرضّع", type: "number", value: 0 },
  { name: "meal", label: "نوع الوجبة", type: "select", options: ["BB","HB","FB","SC"] },
  { name: "country", label: "كود الدولة", type: "text", value: "PRT" },
  { name: "market_segment", label: "قناة الحجز", type: "select",
    options: ["Direct","Corporate","Online TA","Offline TA/TO","Complementary","Groups","Aviation"] },
  { name: "distribution_channel", label: "قناة التوزيع", type: "select",
    options: ["Direct","Corporate","TA/TO","GDS"] },
  { name: "is_repeated_guest", label: "ضيف متكرر؟", type: "select", options: ["0","1"] },
  { name: "previous_cancellations", label: "عدد الإلغاءات السابقة", type: "number", value: 0 },
  { name: "previous_bookings_not_canceled", label: "حجوزات سابقة ناجحة", type: "number", value: 0 },
  { name: "reserved_room_type", label: "نوع الغرفة المحجوزة", type: "text", value: "A" },
  { name: "assigned_room_type", label: "نوع الغرفة المخصصة", type: "text", value: "A" },
  { name: "booking_changes", label: "عدد تعديلات الحجز", type: "number", value: 0 },
  { name: "deposit_type", label: "نوع العربون", type: "select", options: ["No Deposit","Refundable","Non Refund"] },
  { name: "agent", label: "كود الوكيل", type: "number", value: 9 },
  { name: "company", label: "كود الشركة (اسيبه فاضي لو مفيش)", type: "text", value: "" },
  { name: "days_in_waiting_list", label: "أيام قائمة الانتظار", type: "number", value: 0 },
  { name: "customer_type", label: "نوع العميل", type: "select",
    options: ["Transient","Transient-Party","Contract","Group"] },
  { name: "adr", label: "متوسط سعر الليلة (ADR)", type: "number", value: 100 },
  { name: "required_car_parking_spaces", label: "أماكن انتظار سيارات مطلوبة", type: "number", value: 0 },
  { name: "total_of_special_requests", label: "عدد الطلبات الخاصة", type: "number", value: 0 },
];

function buildPredictForm() {
  const form = document.getElementById("predictForm");
  form.innerHTML = PREDICT_FIELDS.map(f => {
    if (f.type === "select") {
      return `<label class="field"><span>${f.label}</span>
        <select name="${f.name}">${f.options.map(o => `<option value="${o}">${o}</option>`).join("")}</select>
      </label>`;
    }
    return `<label class="field"><span>${f.label}</span>
      <input name="${f.name}" type="${f.type}" value="${f.value ?? ""}">
    </label>`;
  }).join("");
}
buildPredictForm();

document.getElementById("predictBtn").addEventListener("click", async () => {
  const btn = document.getElementById("predictBtn");
  const resultBox = document.getElementById("predictResult");
  const modelKey = document.getElementById("modelSelect").value;

  const formData = new FormData(document.getElementById("predictForm"));
  const booking = {};
  PREDICT_FIELDS.forEach(f => {
    let val = formData.get(f.name);
    if (f.type === "number") val = val === "" ? 0 : Number(val);
    if (f.name === "is_repeated_guest") val = Number(val);
    if (f.name === "company" && val === "") val = null;
    booking[f.name] = val;
  });

  btn.disabled = true;
  btn.textContent = "بيحسب...";
  resultBox.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_key: modelKey, booking })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "خطأ غير متوقع");
    }
    const data = await res.json();
    resultBox.classList.remove("hidden", "canceled", "notcanceled");
    resultBox.classList.add(data.prediction === 1 ? "canceled" : "notcanceled");
    resultBox.innerHTML = `
      <div class="result-title">${data.prediction === 1 ? "❌ الحجز متوقع يتلغي" : "✅ الحجز متوقع يكمّل"}</div>
      <div class="result-sub">الموديل: ${data.model_used}${data.cancellation_probability !== null ? ` — احتمال الإلغاء: ${(data.cancellation_probability * 100).toFixed(1)}%` : ""}</div>`;
  } catch (e) {
    resultBox.classList.remove("hidden", "canceled", "notcanceled");
    resultBox.classList.add("canceled");
    resultBox.innerHTML = `<div class="result-title">حصل خطأ</div><div class="result-sub">${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "توقع الآن";
  }
});
