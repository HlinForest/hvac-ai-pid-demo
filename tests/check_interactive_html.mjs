import fs from "node:fs";
import vm from "node:vm";

const input = process.argv[2];
if (!input) throw new Error("usage: node tests/check_interactive_html.mjs <html>");
const documentText = fs.readFileSync(input, "utf8");
const match = documentText.match(/<script>([\s\S]*?)<\/script>/);
if (!match) throw new Error("page has no inline script");

const elements = new Map();
function getElement(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      id,
      value: "0",
      textContent: "",
      innerHTML: "",
      className: "",
      style: {},
      classList: { add() {}, remove() {} },
      onclick: null,
      onchange: null,
      oninput: null,
    });
  }
  return elements.get(id);
}
getElement("algorithm").value = "zn";
getElement("speed").value = "3";
getElement("scrub").value = "0";

let intervalCallback = null;
const context = {
  console,
  document: {
    getElementById: getElement,
    querySelectorAll: () => [],
  },
  setInterval(callback) {
    intervalCallback = callback;
    return 1;
  },
  clearInterval() {
    intervalCallback = null;
  },
};
vm.runInNewContext(match[1], context, { filename: input });

if (!getElement("temp").textContent.includes("°C")) throw new Error("initial draw did not update temperature");
for (const id of ["play", "pause", "reset", "event"]) {
  if (typeof getElement(id).onclick !== "function") throw new Error(`${id} handler is missing`);
}
getElement("play").onclick();
if (typeof intervalCallback !== "function") throw new Error("play did not start the animation timer");
intervalCallback();
if (Number(getElement("scrub").value) !== 1) throw new Error("animation did not advance one sample");

getElement("algorithm").value = "llm";
getElement("algorithm").onchange();
if (getElement("llm-section").style.display !== "block") throw new Error("LLM audit did not open");
if (!getElement("llm-audit").textContent.includes("inspect_history")) throw new Error("Agent audit was not rendered");

getElement("event").onclick();
if (Number(getElement("scrub").value) <= 0) throw new Error("door-event control did not move the timeline");
console.log("INTERACTIVE_HTML_RUNTIME_OK");
