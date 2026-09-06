/* dash.js — shared JS for the Action Clip Bot dashboard */

function toast(message, kind = "info", timeoutMs = 3500) {
  const region = document.getElementById("toast-region");
  if (!region) {
    alert(message);
    return;
  }
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  region.appendChild(el);
  setTimeout(() => el.remove(), timeoutMs);
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  }[m]));
}

async function toggleAccount(id, btn) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const res = await fetch(`/accounts/${id}/toggle`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    btn.textContent = data.enabled ? "Disable" : "Enable";
    const row = btn.closest("tr");
    if (row) row.classList.toggle("disabled-row", !data.enabled);
    const badge = row ? row.querySelector(".status-badge") : null;
    if (badge) {
      badge.textContent = data.enabled ? "Active" : "Inactive";
      badge.className = `status-badge ${data.enabled ? "enabled" : "disabled"}`;
    }
    toast(data.enabled ? "Account enabled." : "Account disabled.", "success");
  } catch (e) {
    btn.textContent = original;
    toast("Failed to toggle account: " + e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

async function deleteAccount(id, btn) {
  if (!confirm("Delete this account? This cannot be undone.")) return;
  btn.disabled = true;
  try {
    const res = await fetch(`/accounts/${id}/delete`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const row = btn.closest("tr");
    if (row) row.remove();
    toast("Account deleted.", "success");
  } catch (e) {
    btn.disabled = false;
    toast("Failed to delete account: " + e.message, "error");
  }
}

async function copyText(text, fallbackEl) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (e) {
    // Fallback for non-HTTPS / older browsers.
    try {
      const el = fallbackEl || document.createElement("textarea");
      const needsAppend = !fallbackEl;
      el.value = text;
      if (needsAppend) document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      if (needsAppend) el.remove();
      return true;
    } catch (err) {
      return false;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Server renders .active already; JS is only a fallback for edge cases.
  const path = window.location.pathname;
  const links = document.querySelectorAll(".side-nav a");
  if (!document.querySelector(".side-nav a.active")) {
    links.forEach((link) => {
      const href = link.getAttribute("href");
      if (href === path || (href !== "/" && path.startsWith(href))) {
        link.classList.add("active");
      }
    });
  }
  // Sidebar toggle (mobile off-canvas + desktop collapse).
  const toggle = document.getElementById("nav-toggle");
  const scrim = document.getElementById("scrim");
  const closeNav = () => {
    document.body.classList.remove("nav-open");
    toggle?.setAttribute("aria-expanded", "false");
    if (scrim) scrim.hidden = true;
  };
  if (toggle) {
    toggle.addEventListener("click", () => {
      const open = document.body.classList.toggle("nav-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
      if (scrim) scrim.hidden = !open;
    });
  }
  scrim?.addEventListener("click", closeNav);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeNav();
  });
  document.querySelectorAll(".side-link").forEach((a) =>
    a.addEventListener("click", () => {
      if (window.innerWidth <= 960) closeNav();
    })
  );
  initStatusPill();
  initTableFilters();
});

// Live pipeline status in the topbar (quiet 10s poll; stops after 3 failures).
function initStatusPill() {
  const pill = document.getElementById("status-pill");
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  const healthDot = document.getElementById("health-dot");
  if (!pill || !dot || !text) return;
  let failures = 0;
  async function poll() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      failures = 0;
      if (healthDot) healthDot.classList.remove("down");
      if (data.busy) {
        dot.classList.add("busy");
        text.textContent = "Running…";
        pill.href = data.current_run_id ? `/runs/${data.current_run_id}` : "/runs";
      } else {
        dot.classList.remove("busy");
        text.textContent = "Idle";
        pill.href = "/";
      }
    } catch (e) {
      failures += 1;
      if (healthDot) healthDot.classList.add("down");
      if (failures >= 3) return; // stop hammering a dead server
    }
    setTimeout(poll, 10000);
  }
  poll();
}

// Generic client-side table filter: <input data-table-filter="table-id"
// filters rows of #table-id by row text. Case-insensitive, no deps.
function initTableFilters() {
  document.querySelectorAll("[data-table-filter]").forEach((input) => {
    const table = document.getElementById(input.getAttribute("data-table-filter"));
    if (!table) return;
    const tbody = table.querySelector("tbody");
    if (!tbody) return;
    const rows = Array.from(tbody.rows);
    let emptyRow = null;
    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      let visible = 0;
      rows.forEach((tr) => {
        const hit = !q || tr.textContent.toLowerCase().includes(q);
        tr.style.display = hit ? "" : "none";
        if (hit) visible += 1;
      });
      if (visible === 0) {
        if (!emptyRow) {
          emptyRow = document.createElement("tr");
          emptyRow.className = "filter-empty";
          const td = document.createElement("td");
          td.colSpan = 99;
          td.className = "muted";
          td.style.textAlign = "center";
          td.textContent = "No rows match this filter.";
          emptyRow.appendChild(td);
          tbody.appendChild(emptyRow);
        }
        emptyRow.style.display = "";
      } else if (emptyRow) {
        emptyRow.style.display = "none";
      }
    });
  });
}
