// ---------- Tabs ----------
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

const GOLD = "#c9a24b";
const NAVY = "#13293d";
const GREEN = "#2e7d55";
const RED = "#b3453a";
const PALETTE = ["#13293d","#c9a24b","#2e7d55","#b3453a","#5b7f97","#8a6d3b","#3d6b8a","#a3874f"];

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
    statusEl.classList.add("ok");
  } catch (e) {
    statusEl.textContent = "مش قادر أوصل للـ API — تأكد إنه شغال على " + API_BASE;
    statusEl.classList.add("err");
    return;
  }

  loadModels();
  loadFeatures();
  loadInsights();
})();

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

    new Chart(document.getElementById("modelsChart"), {
      type: "bar",
      data: {
        labels: models.map(m => m.Model),
        datasets: [
          { label: "F1", data: models.map(m => m.F1), backgroundColor: GOLD },
          { label: "Accuracy", data: models.map(m => m.Accuracy), backgroundColor: NAVY },
        ]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
    });

    // populate predict dropdown
    const sel = document.getElementById("modelSelect");
    sel.innerHTML = models.map(m => `<option value="${m.file.replace('.pkl','')}">${m.Model}</option>`).join("");

    // stats grid (overview)
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
    new Chart(document.getElementById("featuresChart"), {
      type: "bar",
      data: {
        labels: sorted.map(f => f.feature),
        datasets: [{ label: "الأهمية", data: sorted.map(f => f.importance), backgroundColor: GOLD }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true } }
      }
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

    // season chart
    const seasonLabels = Object.keys(data.cancel_rate_by_season);
    new Chart(document.getElementById("seasonChart"), {
      type: "doughnut",
      data: {
        labels: seasonLabels,
        datasets: [{ data: Object.values(data.cancel_rate_by_season), backgroundColor: PALETTE }]
      }
    });

    // deposit chart
    const depositLabels = Object.keys(data.cancel_rate_by_deposit_type);
    new Chart(document.getElementById("depositChart"), {
      type: "bar",
      data: {
        labels: depositLabels,
        datasets: [{ label: "نسبة الإلغاء %", data: Object.values(data.cancel_rate_by_deposit_type), backgroundColor: RED }]
      },
      options: { plugins: { legend: { display: false } } }
    });

    // market segment chart
    const segLabels = Object.keys(data.cancel_rate_by_market_segment);
    new Chart(document.getElementById("segmentChart"), {
      type: "bar",
      data: {
        labels: segLabels,
        datasets: [{ label: "نسبة الإلغاء %", data: Object.values(data.cancel_rate_by_market_segment), backgroundColor: NAVY }]
      },
      options: { indexAxis: "y", plugins: { legend: { display: false } } }
    });

    // lead time chart
    const leadLabels = Object.keys(data.cancel_rate_by_lead_time_bucket);
    new Chart(document.getElementById("leadTimeChart"), {
      type: "line",
      data: {
        labels: leadLabels,
        datasets: [{ label: "نسبة الإلغاء %", data: Object.values(data.cancel_rate_by_lead_time_bucket), borderColor: GOLD, backgroundColor: "#c9a24b33", fill: true, tension: .3 }]
      }
    });

    // chi square table
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
