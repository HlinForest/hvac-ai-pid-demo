import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import katex from "../vendor/katex/katex.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const project = path.dirname(here);
const reportPath = path.join(project, "outputs", "engineering_report.html");
const html = fs.readFileSync(reportPath, "utf8");

const equationBlocks = [...html.matchAll(/<div class="equation">([\s\S]*?)<\/div>/g)].map((match) => match[1]);
const inlineBlocks = [...html.matchAll(/<span class="math-inline">([\s\S]*?)<\/span>/g)].map((match) => match[1]);
const displayMath = equationBlocks.flatMap((block) =>
  [...block.matchAll(/\\\[([\s\S]*?)\\\]/g)].map((match) => match[1]),
);
const inlineMath = inlineBlocks.flatMap((block) =>
  [...block.matchAll(/\\\(([\s\S]*?)\\\)/g)].map((match) => match[1]),
);
const formulas = [
  ...displayMath.map((tex) => ({ tex, displayMode: true })),
  ...inlineMath.map((tex) => ({ tex, displayMode: false })),
];

if (formulas.length < 12) {
  throw new Error(`expected at least 12 formulas, found ${formulas.length}`);
}

for (const formula of formulas) {
  katex.renderToString(formula.tex, {
    displayMode: formula.displayMode,
    throwOnError: true,
    strict: false,
  });
}

const requiredAssets = [
  "katex.min.css",
  "katex.min.js",
  "auto-render.min.js",
  path.join("fonts", "KaTeX_Main-Regular.woff2"),
];
for (const asset of requiredAssets) {
  const target = path.join(project, "outputs", "report_assets", "katex", asset);
  if (!fs.existsSync(target)) {
    throw new Error(`missing offline formula asset: ${target}`);
  }
}

if (!html.includes("1.6 实际运行交叉验证")) {
  throw new Error("cross-validation section is missing");
}
if (!html.includes("physical_cross_validation.png") || !html.includes("fopdt_cross_validation.png")) {
  throw new Error("cross-validation figures are missing");
}

console.log(`KaTeX validation passed: ${displayMath.length} display formulas, ${inlineMath.length} inline formulas.`);
