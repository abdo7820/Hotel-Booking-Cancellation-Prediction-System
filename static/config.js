// غيّر القيمة دي لو الـ API شغال على domain/port مختلف
const API_BASE = window.location.origin.includes("5500") || window.location.protocol === "file:"
  ? "http://localhost:8000"
  : window.location.origin;
