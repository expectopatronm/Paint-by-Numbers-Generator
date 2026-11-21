"use strict";

const frame = new URLSearchParams(location.search).get("frame");
if (frame === "inset" || frame === "panel") document.documentElement.dataset.frame = frame;

const byId = id => document.getElementById(id);
const file = byId("file");
const save = byId("save");
const tone = byId("tone");
const softness = byId("softness");
const grain = byId("grain");
const noise = byId("noise");
const outline = byId("outline");
const zoomOut = byId("zoom-out");
const zoomIn = byId("zoom-in");
const fit = byId("fit");
const pan = byId("pan");
const status = byId("status");
const original = byId("original");
const result = byId("result");
const views = document.querySelector(".views");
const viewports = [...document.querySelectorAll(".viewport")];
const originalContext = original.getContext("2d");
const resultContext = result.getContext("2d");
const descriptions = {
  edges: "Trace changes in brightness as black lines.",
  adaptive: "Separate light and dark areas using the brightness around each pixel.",
};
const settings = {
  edges: { mode: "edges", tone: 128, softness: 1, grain: 0, noise: 0, outline: false },
  adaptive: { mode: "adaptive", tone: 32, softness: 1, grain: 0, noise: 0, outline: false },
};
const maxPixels = 4_000_000;
const zoomLevels = [1, 1.25, 1.5, 2, 3, 4, 6, 8];
let resizer;
let mode = "edges";
let zoomIndex = 0;
let panEnabled = true;
let source;
let target;
let fileName;
let pending;
let loadId = 0;

function render() {
  clearTimeout(pending);
  if (!source) return;
  Stencil.render(source, settings[mode], target);
  resultContext.putImageData(target, 0, 0);
}

function scheduleRender() {
  clearTimeout(pending);
  if (source) pending = setTimeout(render, 55);
}

function updateControls() {
  const current = settings[mode];
  tone.value = current.tone;
  softness.value = current.softness;
  outline.checked = current.outline;
  byId("tone-value").value = current.tone;
  byId("softness-value").value = current.softness.toFixed(1);
  for (const control of [grain, noise]) {
    const value = current[control.id];
    const label = value ? String(value) : "Off";
    control.value = value;
    byId(`${control.id}-value`).value = label;
    control.setAttribute("aria-valuetext", label);
  }
  byId("mode-description").textContent = descriptions[mode];
  byId("outline-field").hidden = mode !== "adaptive";
}

function updatePan() {
  const canPan = zoomIndex > 0 && panEnabled;
  views.classList.toggle("can-pan", canPan);
  pan.disabled = zoomIndex === 0;
  pan.setAttribute("aria-pressed", String(canPan));
}

function setZoom(index) {
  const viewport = viewports[0];
  const before = original.getBoundingClientRect();
  const centerX = (viewport.scrollLeft + viewport.clientWidth / 2) / before.width;
  const centerY = (viewport.scrollTop + viewport.clientHeight / 2) / before.height;
  zoomIndex = Math.max(0, Math.min(zoomLevels.length - 1, index));
  const scale = zoomLevels[zoomIndex];
  byId("zoom-value").value = `${Math.round(scale * 100)}%`;
  views.style.setProperty("--preview-zoom", scale);
  views.classList.toggle("is-zoomed", scale > 1);
  zoomOut.disabled = scale === 1;
  fit.disabled = scale === 1;
  zoomIn.disabled = zoomIndex === zoomLevels.length - 1;
  updatePan();
  for (const pane of viewports) {
    const image = pane.firstElementChild.getBoundingClientRect();
    pane.scrollTo(
      scale === 1 ? 0 : centerX * image.width - pane.clientWidth / 2,
      scale === 1 ? 0 : centerY * image.height - pane.clientHeight / 2,
    );
  }
}

async function resizeForTracing(image, width, height) {
  const fullSize = document.createElement("canvas");
  fullSize.width = image.naturalWidth;
  fullSize.height = image.naturalHeight;
  fullSize.getContext("2d").drawImage(image, 0, 0);
  const resized = document.createElement("canvas");
  resized.width = width;
  resized.height = height;
  resizer ??= pica();
  return resizer.resize(fullSize, resized);
}

async function openImage(input) {
  const id = ++loadId;
  status.textContent = "Opening image…";
  save.disabled = true;
  const url = URL.createObjectURL(input);
  const image = new Image();
  try {
    image.src = url;
    await image.decode();
    if (id !== loadId) return;
    const { naturalWidth, naturalHeight } = image;
    const scale = Math.min(1, Math.sqrt(maxPixels / (naturalWidth * naturalHeight)));
    const width = Math.max(1, Math.floor(naturalWidth * scale));
    const height = Math.max(1, Math.floor(naturalHeight * scale));
    const prepared = scale < 1 ? await resizeForTracing(image, width, height) : image;
    if (id !== loadId) return;
    original.width = result.width = width;
    original.height = result.height = height;
    originalContext.fillStyle = "#fff";
    originalContext.fillRect(0, 0, width, height);
    originalContext.drawImage(prepared, 0, 0);
    source = originalContext.getImageData(0, 0, width, height);
    target = resultContext.createImageData(width, height);
    fileName = input.name;
    render();
    views.closest("main").style.setProperty("--image-ratio", width / height);
    views.hidden = false;
    setZoom(0);
    const dimensions = `${width} × ${height} px`;
    status.textContent = `${fileName} · ${dimensions}${scale < 1 ? " · resized for tracing" : ""}`;
  } catch {
    if (id === loadId) status.textContent = "Could not open this file. Choose a PNG, JPEG, or another supported image.";
  } finally {
    URL.revokeObjectURL(url);
    if (id === loadId) save.disabled = !source;
  }
}

document.querySelectorAll('input[name="mode"]').forEach(radio => {
  radio.addEventListener("change", () => {
    mode = radio.value;
    updateControls();
    scheduleRender();
  });
});
for (const control of [tone, softness, grain, noise, outline]) {
  control.addEventListener("input", () => {
    settings[mode].tone = Number(tone.value);
    settings[mode].softness = Number(softness.value);
    settings[mode].grain = Number(grain.value);
    settings[mode].noise = Number(noise.value);
    settings[mode].outline = outline.checked;
    updateControls();
    scheduleRender();
  });
}

byId("open").addEventListener("click", () => file.click());
file.addEventListener("change", () => {
  if (file.files.length) openImage(file.files[0]);
  file.value = "";
});
document.addEventListener("dragover", event => {
  if (!event.dataTransfer.types.includes("Files")) return;
  event.preventDefault();
  document.body.classList.add("dragging");
});
document.addEventListener("dragleave", event => {
  if (!event.relatedTarget) document.body.classList.remove("dragging");
});
document.addEventListener("drop", event => {
  if (!event.dataTransfer.files.length) return;
  event.preventDefault();
  document.body.classList.remove("dragging");
  openImage(event.dataTransfer.files[0]);
});

save.addEventListener("click", () => {
  render();
  const name = `${fileName.replace(/\.[^.]+$/, "")}-${mode}.png`;
  result.toBlob(blob => {
    if (!blob) {
      status.textContent = "Could not create the PNG.";
      return;
    }
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, "image/png");
});

zoomOut.addEventListener("click", () => setZoom(zoomIndex - 1));
zoomIn.addEventListener("click", () => setZoom(zoomIndex + 1));
fit.addEventListener("click", () => setZoom(0));
pan.addEventListener("click", () => {
  panEnabled = !panEnabled;
  updatePan();
});

for (const [index, viewport] of viewports.entries()) {
  viewport.addEventListener("scroll", () => {
    const other = viewports[1 - index];
    other.scrollLeft = viewport.scrollLeft;
    other.scrollTop = viewport.scrollTop;
  });
  let drag;
  viewport.addEventListener("pointerdown", event => {
    if (!views.classList.contains("can-pan") || !event.isPrimary
        || event.button !== 0 || event.target !== viewport.firstElementChild) return;
    drag = {
      id: event.pointerId,
      x: event.clientX + viewport.scrollLeft,
      y: event.clientY + viewport.scrollTop,
    };
    viewport.setPointerCapture(drag.id);
    viewport.classList.add("is-panning");
  });
  viewport.addEventListener("pointermove", event => {
    if (drag?.id !== event.pointerId) return;
    viewport.scrollTo(drag.x - event.clientX, drag.y - event.clientY);
  });
  viewport.addEventListener("lostpointercapture", () => {
    drag = undefined;
    viewport.classList.remove("is-panning");
  });
}

updateControls();
