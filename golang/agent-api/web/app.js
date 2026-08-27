const thread = document.getElementById("thread");
const form = document.getElementById("chat-form");
const input = document.getElementById("message");
const send = document.getElementById("send");
const statusEl = document.getElementById("status");
const sidebar = document.getElementById("sidebar");
const backdrop = document.getElementById("backdrop");
const menuBtn = document.getElementById("menu-btn");

function setView(name) {
  document.querySelectorAll(".view").forEach((el) => {
    el.hidden = el.id !== `view-${name}`;
  });
  document.querySelectorAll(".nav-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.view === name);
  });
  sidebar.classList.remove("open");
  backdrop.hidden = true;
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => setView(btn.dataset.view));
});

menuBtn.addEventListener("click", () => {
  const open = !sidebar.classList.contains("open");
  sidebar.classList.toggle("open", open);
  backdrop.hidden = !open;
});
backdrop.addEventListener("click", () => {
  sidebar.classList.remove("open");
  backdrop.hidden = true;
});

function addBubble(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `bubble ${role}`;
  wrap.innerHTML = `<span class="who">${role === "user" ? "You" : "Mini Hack Assistant"}</span>`;
  const body = document.createElement("div");
  body.textContent = text;
  wrap.appendChild(body);
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
  return wrap;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addBubble("user", message);
  input.value = "";
  send.disabled = true;
  statusEl.textContent = "Thinking";

  const pending = addBubble("agent", "Writing a reply...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      pending.classList.add("error");
      pending.lastChild.textContent = data.error || "The assistant could not reply.";
      statusEl.textContent = "Offline";
      return;
    }
    pending.lastChild.textContent = data.text;
    statusEl.textContent = "Online";
  } catch (err) {
    pending.classList.add("error");
    pending.lastChild.textContent = "Something went wrong. Please try again.";
    statusEl.textContent = "Offline";
  } finally {
    send.disabled = false;
    input.focus();
  }
});

addBubble("agent", "Hi — I am Mini Hack Assistant. Ask me anything about Avalanche or this session.");
setView("overview");
