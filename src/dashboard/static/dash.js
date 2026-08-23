/* dash.js — shared JS for the Action Clip Bot dashboard */

async function toggleAccount(id, btn) {
  try {
    const res = await fetch(`/accounts/${id}/toggle`, { method: 'POST' });
    const data = await res.json();
    btn.textContent = data.enabled ? 'Disable' : 'Enable';
    const row = btn.closest('tr');
    row.classList.toggle('disabled-row', !data.enabled);
    const badge = row.querySelector('.status-badge');
    if (badge) {
      badge.textContent = data.enabled ? 'Active' : 'Inactive';
      badge.className = `status-badge ${data.enabled ? 'enabled' : 'disabled'}`;
    }
  } catch (e) {
    alert('Failed to toggle account: ' + e);
  }
}

async function deleteAccount(id, btn) {
  if (!confirm('Delete this account? This cannot be undone.')) return;
  try {
    const res = await fetch(`/accounts/${id}/delete`, { method: 'POST' });
    if (res.ok) {
      btn.closest('tr').remove();
    } else {
      alert('Failed to delete');
    }
  } catch (e) {
    alert('Failed to delete account: ' + e);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const path = window.location.pathname;
  const links = document.querySelectorAll(".nav-links a");
  links.forEach(link => {
    if (link.getAttribute("href") === path) {
      link.classList.add("active");
    }
  });
});
