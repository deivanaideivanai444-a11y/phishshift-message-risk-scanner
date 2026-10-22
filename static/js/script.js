const form = document.getElementById("scan-form");
const message = document.getElementById("body_text");
const count = document.getElementById("char-count");
const loading = document.getElementById("loading-note");
const button = document.getElementById("analyse-button");

function updateCount() {
  if (message && count) count.textContent = `${message.value.length} / 5000`;
}
if (message) {
  updateCount();
  message.addEventListener("input", updateCount);
}
if (form) {
  form.addEventListener("submit", (event) => {
    if (!message.value.trim()) {
      event.preventDefault();
      message.focus();
      return;
    }
    loading.hidden = false;
    button.disabled = true;
  });
}
