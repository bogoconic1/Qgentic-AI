const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const path = require("path");

const {
  FaCode, FaShieldAlt, FaSearch, FaBookOpen, FaEye,
  FaNetworkWired, FaLayerGroup, FaSyncAlt, FaCogs
} = require("react-icons/fa");

// ---- Color Palette ----
const C = {
  bgDark:    "0F1629",
  bgMedium:  "1A2744",
  bgCard:    "1E2F4D",
  accent:    "00B4D8",
  accentDim: "0891B2",
  accentBg:  "0D3D56",
  white:     "FFFFFF",
  textMain:  "E2E8F0",
  textMuted: "94A3B8",
  textDim:   "64748B",
  success:   "10B981",
  warning:   "F59E0B",
  danger:    "EF4444",
  purple:    "8B5CF6",
};

function renderIconSvg(IconComponent, color, size) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color: "#" + color, size: String(size) })
  );
}
async function iconToBase64Png(IconComponent, color = C.accent, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}
const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.3 });

function addCard(slide, x, y, w, h) {
  slide.addShape("rect", { x, y, w, h, fill: { color: C.bgCard }, shadow: makeShadow(), line: { color: C.accentDim, width: 0.5 } });
}

async function addIconCircle(slide, icon, x, y, color = C.accent) {
  slide.addShape("ellipse", { x, y, w: 0.4, h: 0.4, fill: { color: C.accentBg } });
  const iconData = await iconToBase64Png(icon, color, 256);
  slide.addImage({ data: iconData, x: x + 0.08, y: y + 0.08, w: 0.24, h: 0.24 });
}

const TOTAL = 5;
const LEADERBOARD_IMG = path.resolve("/Users/oh_jason_zhang/Downloads/git_repo/Qgentic-AI/docs/assets/csiro_lb.png");

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Qgentic AI";
  pres.title = "Qgentic AI: An Automated ML Competition Stack";

  // ================================================================
  // SLIDE 1: Title + Problem + Motivation (combined)
  // ================================================================
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    slide.addShape("rect", { x: 0, y: 0, w: 10, h: 0.05, fill: { color: C.accent } });

    // Title block - left side
    slide.addText("QGENTIC AI", {
      x: 0.5, y: 0.3, w: 5.5, h: 0.7,
      fontSize: 40, color: C.white, fontFace: "Arial Black", bold: true, charSpacing: 3, margin: 0
    });
    slide.addText("An Automated ML Competition Stack", {
      x: 0.5, y: 0.95, w: 5.5, h: 0.35,
      fontSize: 16, color: C.accent, fontFace: "Calibri", margin: 0
    });
    slide.addShape("rect", { x: 0.5, y: 1.4, w: 2.0, h: 0.03, fill: { color: C.accent } });

    // Problem statement
    slide.addText("THE PROBLEM", {
      x: 0.5, y: 1.65, w: 5.0, h: 0.3,
      fontSize: 11, color: C.accentDim, fontFace: "Calibri", bold: true, charSpacing: 2, margin: 0
    });
    slide.addText([
      { text: "Kaggle gold medals require 200+ hours of work against thousands of competitors. ", options: { fontSize: 12, color: C.textMain, fontFace: "Calibri" } },
      { text: "Most time (~80%) goes to repetitive execution:", options: { fontSize: 12, color: C.textMain, fontFace: "Calibri" } },
      { text: " debugging crashes, monitoring training, iterating models, evaluating submissions.", options: { fontSize: 12, color: C.textMuted, fontFace: "Calibri" } },
    ], { x: 0.5, y: 1.95, w: 5.0, h: 0.9 });

    // Stat cards
    const stats = [
      { val: "200+", unit: "hrs", desc: "Manual effort\nper gold medal", color: C.danger },
      { val: "~20", unit: "hrs", desc: "With Qgentic AI\n(human strategic input)", color: C.success },
      { val: "90%", unit: "", desc: "Effort reduction\nvia LLM agents", color: C.accent },
    ];
    stats.forEach((s, i) => {
      const sx = 0.5 + i * 1.75;
      addCard(slide, sx, 3.0, 1.55, 1.15);
      slide.addText(s.val + s.unit, {
        x: sx, y: 3.05, w: 1.55, h: 0.5,
        fontSize: 24, color: s.color, fontFace: "Arial Black", bold: true, align: "center", margin: 0
      });
      slide.addText(s.desc, {
        x: sx + 0.1, y: 3.55, w: 1.35, h: 0.5,
        fontSize: 10, color: C.textMuted, fontFace: "Calibri", align: "center", margin: 0
      });
    });

    // Key insight at bottom
    slide.addShape("rect", { x: 0.5, y: 4.35, w: 5.0, h: 0.03, fill: { color: C.bgMedium } });
    slide.addText("Insight: LLMs can now handle the structured execution loop. Keep the human as strategic director.", {
      x: 0.5, y: 4.5, w: 5.0, h: 0.3,
      fontSize: 11, color: C.textMuted, fontFace: "Calibri", italic: true, margin: 0
    });

    // Right side: Related work panel
    addCard(slide, 5.8, 0.3, 3.9, 4.9);
    slide.addText("RELATED WORK & POSITIONING", {
      x: 6.0, y: 0.4, w: 3.5, h: 0.25,
      fontSize: 10, color: C.accentDim, fontFace: "Calibri", bold: true, charSpacing: 2, margin: 0
    });

    // Comparison table
    const hdrOpts = { fill: { color: C.accentDim }, color: C.white, bold: true, fontSize: 9, fontFace: "Calibri", align: "center" };
    const cellOpts = (v) => ({ fill: { color: C.bgMedium }, color: v === "\u2713" ? C.success : (v === "\u2717" ? C.danger : C.textMain), fontSize: 9, fontFace: "Calibri", align: "center" });
    const nameOpts = { fill: { color: C.bgMedium }, color: C.textMain, fontSize: 9, fontFace: "Calibri" };

    slide.addTable([
      [{ text: "", options: hdrOpts }, { text: "Code", options: hdrOpts }, { text: "LLM", options: hdrOpts }, { text: "Comp", options: hdrOpts }, { text: "Multi", options: hdrOpts }, { text: "X-Lrn", options: hdrOpts }],
      [{ text: "Auto-sklearn", options: nameOpts }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }],
      [{ text: "AutoGluon", options: nameOpts }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }],
      [{ text: "Copilot/Cursor", options: nameOpts }, { text: "\u2713", options: cellOpts("\u2713") }, { text: "\u2713", options: cellOpts("\u2713") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }],
      [{ text: "AIDE/AutoKaggle", options: nameOpts }, { text: "\u2713", options: cellOpts("\u2713") }, { text: "\u2713", options: cellOpts("\u2713") }, { text: "\u2713", options: cellOpts("\u2713") }, { text: "\u2717", options: cellOpts("\u2717") }, { text: "\u2717", options: cellOpts("\u2717") }],
      [{ text: "Qgentic AI", options: { fill: { color: C.accentBg }, color: C.accent, fontSize: 9, fontFace: "Calibri", bold: true } },
       { text: "\u2713", options: { fill: { color: C.accentBg }, color: C.success, fontSize: 9, fontFace: "Calibri", align: "center" } },
       { text: "\u2713", options: { fill: { color: C.accentBg }, color: C.success, fontSize: 9, fontFace: "Calibri", align: "center" } },
       { text: "\u2713", options: { fill: { color: C.accentBg }, color: C.success, fontSize: 9, fontFace: "Calibri", align: "center" } },
       { text: "\u2713", options: { fill: { color: C.accentBg }, color: C.success, fontSize: 9, fontFace: "Calibri", align: "center" } },
       { text: "\u2713", options: { fill: { color: C.accentBg }, color: C.success, fontSize: 9, fontFace: "Calibri", align: "center" } }],
    ], {
      x: 6.0, y: 0.75, w: 3.5, colW: [1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
      border: { pt: 0.5, color: C.bgDark },
      rowH: [0.35, 0.28, 0.28, 0.28, 0.28, 0.28]
    });

    // Key differentiators below table
    slide.addText("Key differentiators:", {
      x: 6.0, y: 2.65, w: 3.5, h: 0.2,
      fontSize: 10, color: C.white, fontFace: "Calibri", bold: true, margin: 0
    });
    const diffs = [
      "AutoML: Fixed search spaces, no code generation or reasoning",
      "Copilot/Cursor: Code completion, not end-to-end pipelines",
      "AIDE/AutoKaggle: Closest, but single-agent, no cross-model learning",
      "Qgentic AI: Multi-agent research + parallel execution + shared learning"
    ];
    slide.addText(diffs.map((d, i) => ({
      text: d,
      options: { bullet: true, breakLine: i < diffs.length - 1, fontSize: 10, color: i === 3 ? C.accent : C.textMuted, fontFace: "Calibri", paraSpaceAfter: 4, bold: i === 3 }
    })), { x: 6.0, y: 2.9, w: 3.5, h: 2.2 });

    slide.addText("1 / 5", { x: 8.8, y: 5.2, w: 0.8, h: 0.3, fontSize: 9, color: C.textDim, align: "right", fontFace: "Calibri" });
  }

  // ================================================================
  // SLIDE 2: Architecture (full pipeline diagram + agent details)
  // ================================================================
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    slide.addText("SYSTEM ARCHITECTURE", {
      x: 0.5, y: 0.15, w: 5, h: 0.25,
      fontSize: 9, color: C.accentDim, fontFace: "Calibri", bold: true, charSpacing: 3, margin: 0
    });
    slide.addText("5 Coordinated LLM Agents", {
      x: 0.5, y: 0.35, w: 9, h: 0.45,
      fontSize: 24, color: C.white, fontFace: "Arial Black", bold: true, margin: 0
    });

    // Pipeline flow - top row
    const agents = [
      { name: "Starter", desc: "Task\nClassification", x: 0.3, color: C.purple },
      { name: "Researcher", desc: "Deep Research\nLoop", x: 2.0, color: C.accentDim },
      { name: "Model\nRecommender", desc: "Strategy\nSelection", x: 3.7, color: C.warning },
      { name: "Developer", desc: "Code Gen &\nIteration (xN)", x: 5.4, color: C.success },
      { name: "Results", desc: "Aggregation\n& Ranking", x: 7.4, color: C.accent },
    ];
    agents.forEach(a => {
      slide.addShape("rect", { x: a.x, y: 0.95, w: 1.5, h: 0.7, fill: { color: a.color }, shadow: makeShadow() });
      slide.addText(a.name, {
        x: a.x, y: 0.95, w: 1.5, h: 0.4,
        fontSize: 10, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "bottom", margin: 0
      });
      slide.addText(a.desc, {
        x: a.x, y: 1.35, w: 1.5, h: 0.3,
        fontSize: 7, color: C.textMuted, fontFace: "Calibri", align: "center", valign: "top", margin: 0
      });
    });
    // Arrows
    [1.8, 3.5, 5.2, 6.9].forEach(ax => {
      slide.addText("\u25B6", { x: ax, y: 1.1, w: 0.2, h: 0.3, fontSize: 11, color: C.accent, align: "center" });
    });

    // Orchestrator bar
    slide.addShape("rect", { x: 0.3, y: 1.8, w: 8.6, h: 0.3, fill: { color: C.bgCard }, line: { color: C.accent, width: 0.5, dashType: "dash" } });
    slide.addText("ORCHESTRATOR  \u2014  Phase Sequencing  |  Parallel Execution  |  GPU/CPU Isolation  |  Checkpoint/Rollback", {
      x: 0.5, y: 1.8, w: 8.2, h: 0.3,
      fontSize: 9, color: C.accent, fontFace: "Calibri", align: "center", valign: "middle"
    });

    // Agent detail cards - 2 rows of info
    // Row 1: Starter + Researcher + ModelRecommender
    const detailY1 = 2.3;
    // Starter
    addCard(slide, 0.3, detailY1, 2.8, 1.4);
    slide.addText("StarterAgent", { x: 0.45, y: detailY1 + 0.05, w: 2.5, h: 0.25, fontSize: 12, color: C.purple, fontFace: "Calibri", bold: true, margin: 0 });
    slide.addText("Single LLM call with Google Search. Classifies task type (tabular, NLP, CV, time series) and generates summary for downstream agents.", {
      x: 0.45, y: detailY1 + 0.3, w: 2.5, h: 0.55, fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
    });
    slide.addText("Output: starter_suggestions.json", { x: 0.45, y: detailY1 + 0.9, w: 2.5, h: 0.2, fontSize: 9, color: C.textDim, fontFace: "Consolas", margin: 0 });

    // Researcher
    addCard(slide, 3.3, detailY1, 2.8, 1.4);
    slide.addText("ResearcherAgent", { x: 3.45, y: detailY1 + 0.05, w: 2.5, h: 0.25, fontSize: 12, color: C.accentDim, fontFace: "Calibri", bold: true, margin: 0 });
    slide.addText("Up to 512-step agentic tool-calling loop. Tools: execute_python (EDA + image ingestion), read_research_paper (ArXiv), scrape_web_page (Firecrawl).", {
      x: 3.45, y: detailY1 + 0.3, w: 2.5, h: 0.55, fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
    });
    slide.addText("Output: plan.md", { x: 3.45, y: detailY1 + 0.9, w: 2.5, h: 0.2, fontSize: 9, color: C.textDim, fontFace: "Consolas", margin: 0 });

    // ModelRecommender
    addCard(slide, 6.3, detailY1, 2.8, 1.4);
    slide.addText("ModelRecommender", { x: 6.45, y: detailY1 + 0.05, w: 2.5, h: 0.25, fontSize: 12, color: C.warning, fontFace: "Calibri", bold: true, margin: 0 });
    slide.addText("3-stage: 16 candidates \u2192 paper analysis \u2192 8 models. Per-model: preprocessing, loss, hyperparams, inference. MUST_HAVE vs NICE_TO_HAVE.", {
      x: 6.45, y: detailY1 + 0.3, w: 2.5, h: 0.55, fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
    });
    slide.addText("Output: model_recommendations.json", { x: 6.45, y: detailY1 + 0.9, w: 2.5, h: 0.2, fontSize: 9, color: C.textDim, fontFace: "Consolas", margin: 0 });

    // Row 2: DeveloperAgent (wider, more detail)
    const detailY2 = 3.9;
    addCard(slide, 0.3, detailY2, 8.8, 1.4);
    slide.addText("DeveloperAgent (parallel, one per model)", { x: 0.45, y: detailY2 + 0.05, w: 4.0, h: 0.25, fontSize: 12, color: C.success, fontFace: "Calibri", bold: true, margin: 0 });

    // Iteration loop mini-diagram inside the card
    const loopBoxes = [
      { label: "Generate\ntrain.py", x: 0.5, color: C.accentDim },
      { label: "Guardrails", x: 1.7, color: C.danger },
      { label: "Execute", x: 2.9, color: C.success },
      { label: "Monitor\nLogs 120s", x: 4.1, color: C.warning },
      { label: "Score", x: 5.3, color: C.purple },
      { label: "SOTA\nSearch", x: 6.5, color: C.accent },
    ];
    loopBoxes.forEach((b, i) => {
      slide.addShape("rect", { x: b.x, y: detailY2 + 0.4, w: 1.05, h: 0.5, fill: { color: b.color } });
      slide.addText(b.label, {
        x: b.x, y: detailY2 + 0.4, w: 1.05, h: 0.5,
        fontSize: 9, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
      });
      if (i < loopBoxes.length - 1) {
        slide.addText("\u25B6", { x: b.x + 1.05, y: detailY2 + 0.5, w: 0.15, h: 0.25, fontSize: 8, color: C.textDim, align: "center" });
      }
    });
    // Loop back indicator
    slide.addShape("rect", { x: 0.5, y: detailY2 + 0.95, w: 7.1, h: 0.015, fill: { color: C.accent } });
    slide.addText("\u25C0 iterate", { x: 0.5, y: detailY2 + 0.97, w: 0.8, h: 0.2, fontSize: 7, color: C.accent, fontFace: "Calibri" });

    // Key features text to the right
    slide.addText([
      { text: "Tools: ", options: { bold: true, fontSize: 8, color: C.white, fontFace: "Calibri" } },
      { text: "execute_code, monitor_logs, search_red_flags,\nsearch_sota_suggestions, investigate_library", options: { fontSize: 8, color: C.textMuted, fontFace: "Consolas" } },
    ], { x: 7.7, y: detailY2 + 0.35, w: 1.3, h: 0.8, margin: 0 });

    slide.addText("2 / 5", { x: 8.8, y: 5.2, w: 0.8, h: 0.3, fontSize: 9, color: C.textDim, align: "right", fontFace: "Calibri" });
  }

  // ================================================================
  // SLIDE 3: Methodology (Cross-Model Learning + Guardrails + Infra)
  // ================================================================
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    slide.addText("METHODOLOGY", {
      x: 0.5, y: 0.15, w: 5, h: 0.25,
      fontSize: 9, color: C.accentDim, fontFace: "Calibri", bold: true, charSpacing: 3, margin: 0
    });
    slide.addText("Key Technical Innovations", {
      x: 0.5, y: 0.35, w: 9, h: 0.45,
      fontSize: 24, color: C.white, fontFace: "Arial Black", bold: true, margin: 0
    });

    // LEFT COLUMN: Cross-Model Learning
    addCard(slide, 0.3, 0.95, 4.5, 2.5);
    slide.addText("Cross-Model Learning", {
      x: 0.45, y: 1.0, w: 4.2, h: 0.3,
      fontSize: 14, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });

    // Mini diagram: Model A / Shared Pool / Model B
    slide.addShape("rect", { x: 0.5, y: 1.4, w: 1.2, h: 0.6, fill: { color: C.accentDim } });
    slide.addText("Model A", { x: 0.5, y: 1.4, w: 1.2, h: 0.6, fontSize: 9, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle" });

    slide.addShape("rect", { x: 1.85, y: 1.35, w: 1.6, h: 0.7, fill: { color: C.accentBg }, line: { color: C.accent, width: 1 } });
    slide.addText("Shared Pool", { x: 1.85, y: 1.35, w: 1.6, h: 0.35, fontSize: 9, color: C.accent, fontFace: "Calibri", bold: true, align: "center", valign: "bottom", margin: 0 });
    slide.addText("(thread-safe)", { x: 1.85, y: 1.7, w: 1.6, h: 0.25, fontSize: 7, color: C.textDim, fontFace: "Calibri", align: "center", margin: 0 });

    slide.addShape("rect", { x: 3.6, y: 1.4, w: 1.0, h: 0.6, fill: { color: C.purple } });
    slide.addText("Model B", { x: 3.6, y: 1.4, w: 1.0, h: 0.6, fontSize: 9, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle" });

    // Arrows
    slide.addText("\u2194", { x: 1.65, y: 1.48, w: 0.3, h: 0.3, fontSize: 14, color: C.accent, align: "center" });
    slide.addText("\u2194", { x: 3.35, y: 1.48, w: 0.3, h: 0.3, fontSize: 14, color: C.accent, align: "center" });

    // Entries
    const poolEntries = [
      { t: "\u2713 Cosine annealing (+0.02)", c: C.success },
      { t: "\u2713 MI feature selection (+0.015)", c: C.success },
      { t: "\u2717 Label smoothing (-0.01)", c: C.danger },
      { t: "\u2717 Heavy augmentation (-0.005)", c: C.danger },
    ];
    poolEntries.forEach((e, i) => {
      slide.addText(e.t, { x: 0.6, y: 2.15 + i * 0.28, w: 3.5, h: 0.25, fontSize: 9, color: e.c, fontFace: "Calibri", margin: 0 });
    });

    // RIGHT COLUMN: Safety & Guardrails
    addCard(slide, 5.0, 0.95, 4.7, 2.5);
    slide.addText("Safety & Guardrails", {
      x: 5.15, y: 1.0, w: 4.4, h: 0.3,
      fontSize: 14, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });

    const guardLayers = [
      { name: "AST Static Analysis", desc: "logging.basicConfig order, data partition patterns", color: C.accentDim },
      { name: "LLM Leakage Review", desc: "Gemini 3.1 Pro detects train/test contamination", color: C.warning },
      { name: "LLM Code Safety", desc: "eval/exec, injection, credential leakage (Gemini 2.5 Flash)", color: C.danger },
    ];
    for (let i = 0; i < guardLayers.length; i++) {
      const g = guardLayers[i];
      const gy = 1.4 + i * 0.6;
      await addIconCircle(slide, FaShieldAlt, 5.2, gy + 0.05, g.color);
      slide.addText(g.name, { x: 5.7, y: gy, w: 3.0, h: 0.2, fontSize: 10, color: C.white, fontFace: "Calibri", bold: true, margin: 0 });
      slide.addText(g.desc, { x: 5.7, y: gy + 0.22, w: 3.8, h: 0.25, fontSize: 9, color: C.textMuted, fontFace: "Calibri", margin: 0 });
    }

    slide.addText("Pipeline: evaluate_guardrails() runs before every code execution. Blocked code \u2192 feedback for regeneration.", {
      x: 5.15, y: 3.0, w: 4.4, h: 0.3,
      fontSize: 8, color: C.textDim, fontFace: "Calibri", italic: true, margin: 0
    });

    // BOTTOM ROW: Infrastructure
    addCard(slide, 0.3, 3.65, 9.4, 1.55);
    slide.addText("Infrastructure & Resource Management", {
      x: 0.45, y: 3.7, w: 4.0, h: 0.3,
      fontSize: 13, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });

    // GPU modes
    const gpuModes = [
      { name: "MIG", desc: "Single GPU\npartitioned", color: C.accentDim },
      { name: "Multi-GPU", desc: "Separate GPUs\nper worker", color: C.success },
      { name: "CPU-Only", desc: "CPU affinity\npinning", color: C.warning },
    ];
    gpuModes.forEach((m, i) => {
      const gx = 0.5 + i * 1.25;
      slide.addShape("rect", { x: gx, y: 4.1, w: 1.1, h: 0.65, fill: { color: m.color } });
      slide.addText(m.name, { x: gx, y: 4.1, w: 1.1, h: 0.3, fontSize: 9, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "bottom", margin: 0 });
      slide.addText(m.desc, { x: gx, y: 4.4, w: 1.1, h: 0.3, fontSize: 7, color: C.textMuted, fontFace: "Calibri", align: "center", margin: 0 });
    });

    // Config stats
    const infraStats = [
      { val: "5 days", desc: "Baseline budget", color: C.accent },
      { val: "12 hrs", desc: "Script timeout", color: C.warning },
      { val: "120s", desc: "Log monitor poll", color: C.success },
    ];
    infraStats.forEach((s, i) => {
      const ix = 4.2 + i * 1.6;
      slide.addText(s.val, { x: ix, y: 4.1, w: 1.4, h: 0.35, fontSize: 18, color: s.color, fontFace: "Arial Black", bold: true, margin: 0 });
      slide.addText(s.desc, { x: ix, y: 4.45, w: 1.4, h: 0.2, fontSize: 8, color: C.textMuted, fontFace: "Calibri", margin: 0 });
    });

    // Extra detail
    slide.addText("Isolated conda envs per model  |  ThreadPoolExecutor  |  Queue-based GPU/CPU pools  |  SQLite checkpoints + rollback  |  HITL via INSTRUCTIONS.md", {
      x: 0.45, y: 4.85, w: 9.0, h: 0.2,
      fontSize: 9, color: C.textMuted, fontFace: "Calibri", margin: 0
    });

    slide.addText("3 / 5", { x: 8.8, y: 5.2, w: 0.8, h: 0.3, fontSize: 9, color: C.textDim, align: "right", fontFace: "Calibri" });
  }

  // ================================================================
  // SLIDE 4: Results + Lessons Learned
  // ================================================================
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    slide.addText("RESULTS", {
      x: 0.5, y: 0.15, w: 5, h: 0.25,
      fontSize: 9, color: C.accentDim, fontFace: "Calibri", bold: true, charSpacing: 3, margin: 0
    });
    slide.addText("CSIRO Biomass Competition", {
      x: 0.5, y: 0.35, w: 9, h: 0.45,
      fontSize: 24, color: C.white, fontFace: "Arial Black", bold: true, margin: 0
    });

    // Big stat cards
    const results = [
      { val: "0.63772", label: "Leaderboard Score", color: C.accent },
      { val: "32 / 3,802", label: "Final Ranking", color: C.success },
      { val: "Top 1%", label: "Silver Medal", color: C.warning },
    ];
    results.forEach((r, i) => {
      const rx = 0.3 + i * 3.15;
      addCard(slide, rx, 0.95, 2.9, 0.9);
      slide.addText(r.val, { x: rx, y: 0.98, w: 2.9, h: 0.45, fontSize: 24, color: r.color, fontFace: "Arial Black", bold: true, align: "center", margin: 0 });
      slide.addText(r.label, { x: rx, y: 1.43, w: 2.9, h: 0.25, fontSize: 10, color: C.textMuted, fontFace: "Calibri", align: "center", margin: 0 });
    });

    // Effort comparison + Leaderboard screenshot
    addCard(slide, 0.3, 2.05, 4.5, 1.5);
    slide.addText("Human Effort Reduction", { x: 0.45, y: 2.1, w: 4.2, h: 0.25, fontSize: 12, color: C.white, fontFace: "Calibri", bold: true, margin: 0 });
    slide.addShape("rect", { x: 0.5, y: 2.45, w: 4.0, h: 0.3, fill: { color: C.danger } });
    slide.addText("Traditional: 200+ hours", { x: 0.5, y: 2.45, w: 4.0, h: 0.3, fontSize: 10, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle" });
    slide.addShape("rect", { x: 0.5, y: 2.85, w: 0.4, h: 0.3, fill: { color: C.success } });
    slide.addText("Qgentic AI: ~20 hrs", { x: 1.0, y: 2.85, w: 1.8, h: 0.3, fontSize: 10, color: C.success, fontFace: "Calibri", bold: true, valign: "middle" });
    slide.addText("90% REDUCTION", { x: 3.0, y: 2.85, w: 1.5, h: 0.3, fontSize: 12, color: C.success, fontFace: "Arial Black", bold: true, valign: "middle" });

    // Leaderboard image
    addCard(slide, 5.0, 2.05, 4.7, 1.5);
    slide.addText("Kaggle Leaderboard", { x: 5.15, y: 2.1, w: 4.4, h: 0.2, fontSize: 10, color: C.accent, fontFace: "Calibri", bold: true, margin: 0 });
    slide.addImage({ path: LEADERBOARD_IMG, x: 5.15, y: 2.35, w: 4.4, h: 1.1, sizing: { type: "contain", w: 4.4, h: 1.1 } });

    // Lessons Learned
    slide.addText("LESSONS LEARNED", {
      x: 0.5, y: 3.7, w: 5, h: 0.25,
      fontSize: 9, color: C.accentDim, fontFace: "Calibri", bold: true, charSpacing: 2, margin: 0
    });

    const lessons = [
      { icon: FaEye, title: "LLM Log Monitoring Works", desc: "Catches overfitting, NaN losses, stalled training that humans miss during off-hours", color: C.accent },
      { icon: FaNetworkWired, title: "Cross-Model Learning Prevents Redundancy", desc: "Shared pool eliminates the \"try the same bad idea 8 times\" failure mode", color: C.success },
      { icon: FaLayerGroup, title: "Structured Outputs Are Essential", desc: "Pydantic schemas for LLM responses make multi-agent communication reliable", color: C.purple },
      { icon: FaSyncAlt, title: "Robust Retry Logic Is Critical", desc: "Exponential backoff (1s\u201960s) + 5-min polling on sustained 503 keeps system running", color: C.warning },
    ];
    for (let i = 0; i < lessons.length; i++) {
      const l = lessons[i];
      const lx = (i % 2) * 4.85 + 0.3;
      const ly = 4.05 + Math.floor(i / 2) * 0.7;
      addCard(slide, lx, ly, 4.6, 0.6);
      await addIconCircle(slide, l.icon, lx + 0.1, ly + 0.1, l.color);
      slide.addText(l.title, { x: lx + 0.6, y: ly + 0.03, w: 3.8, h: 0.2, fontSize: 10, color: C.white, fontFace: "Calibri", bold: true, margin: 0 });
      slide.addText(l.desc, { x: lx + 0.6, y: ly + 0.25, w: 3.8, h: 0.25, fontSize: 8, color: C.textMuted, fontFace: "Calibri", margin: 0 });
    }

    slide.addText("4 / 5", { x: 8.8, y: 5.2, w: 0.8, h: 0.3, fontSize: 9, color: C.textDim, align: "right", fontFace: "Calibri" });
  }

  // ================================================================
  // SLIDE 5: Improvement Plan + Conclusion
  // ================================================================
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    slide.addShape("rect", { x: 0, y: 0, w: 10, h: 0.05, fill: { color: C.accent } });

    slide.addText("ROADMAP & CONCLUSION", {
      x: 0.5, y: 0.15, w: 5, h: 0.25,
      fontSize: 9, color: C.accentDim, fontFace: "Calibri", bold: true, charSpacing: 3, margin: 0
    });
    slide.addText("What's Next", {
      x: 0.5, y: 0.35, w: 9, h: 0.45,
      fontSize: 24, color: C.white, fontFace: "Arial Black", bold: true, margin: 0
    });

    // Roadmap items - horizontal layout
    const roadmap = [
      { phase: "Phase 1", title: "Observability", desc: "Weave \u2192 Langfuse\nBetter tracing & cost tracking", color: C.accent },
      { phase: "Phase 2", title: "Enhanced Guardrails", desc: "Lint integration\nImproved leakage detection", color: C.success },
      { phase: "Phase 3", title: "Cross-Worker V2", desc: "Richer shared state:\nembeddings, feature importances", color: C.purple },
      { phase: "Phase 4", title: "Legal Retrieval", desc: "Competition rules\ncompliance (scaffolded)", color: C.warning },
    ];

    // Timeline line
    slide.addShape("rect", { x: 0.5, y: 1.25, w: 9.0, h: 0.03, fill: { color: C.textDim } });

    roadmap.forEach((r, i) => {
      const rx = 0.3 + i * 2.35;
      // Dot
      slide.addShape("ellipse", { x: rx + 0.85, y: 1.1, w: 0.3, h: 0.3, fill: { color: r.color } });
      // Card below
      addCard(slide, rx, 1.55, 2.15, 1.15);
      slide.addText(r.phase, { x: rx + 0.1, y: 1.6, w: 1.0, h: 0.2, fontSize: 8, color: r.color, fontFace: "Calibri", bold: true, margin: 0 });
      slide.addText(r.title, { x: rx + 0.1, y: 1.8, w: 1.95, h: 0.2, fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, margin: 0 });
      slide.addText(r.desc, { x: rx + 0.1, y: 2.05, w: 1.95, h: 0.5, fontSize: 9, color: C.textMuted, fontFace: "Calibri", margin: 0 });
    });

    // Goal callout
    addCard(slide, 0.3, 2.9, 9.4, 0.55);
    slide.addText("\u2605", { x: 0.5, y: 2.95, w: 0.4, h: 0.4, fontSize: 20, color: C.warning, fontFace: "Calibri", align: "center" });
    slide.addText("Ultimate Goal: Kaggle Competitions Grandmaster (2 more gold medals needed)", {
      x: 1.0, y: 2.95, w: 5.5, h: 0.4,
      fontSize: 13, color: C.white, fontFace: "Calibri", bold: true, valign: "middle", margin: 0
    });
    slide.addText("Let the agents compete while you strategize", {
      x: 6.5, y: 2.95, w: 3.0, h: 0.4,
      fontSize: 10, color: C.textMuted, fontFace: "Calibri", italic: true, valign: "middle", margin: 0
    });

    // Key Takeaways
    slide.addText("KEY TAKEAWAYS", {
      x: 0.5, y: 3.65, w: 5, h: 0.25,
      fontSize: 9, color: C.accentDim, fontFace: "Calibri", bold: true, charSpacing: 2, margin: 0
    });

    const takeaways = [
      { text: "Multi-agent decomposition beats monolithic agents", desc: "Specialized agents with distinct tools outperform one agent trying to do everything", color: C.accent },
      { text: "Cross-model learning creates emergent optimization", desc: "Shared suggestion pools enable collective intelligence across parallel baselines", color: C.success },
      { text: "90% effort reduction is achievable today", desc: "Current LLM capabilities are sufficient for structured ML execution loops", color: C.warning },
      { text: "The future is human + AI collaboration", desc: "Humans provide strategic direction; agents handle disciplined execution", color: C.purple },
    ];

    takeaways.forEach((t, i) => {
      const ty = 3.95 + i * 0.35;
      slide.addShape("rect", { x: 0.3, y: ty, w: 0.05, h: 0.3, fill: { color: t.color } });
      slide.addText(t.text, { x: 0.5, y: ty, w: 4.0, h: 0.3, fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, valign: "middle", margin: 0 });
      slide.addText(t.desc, { x: 4.5, y: ty, w: 5.2, h: 0.3, fontSize: 9, color: C.textMuted, fontFace: "Calibri", valign: "middle", margin: 0 });
    });

    // Footer
    slide.addShape("rect", { x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: C.bgMedium } });
    slide.addText("Open Source: github.com/bogoconic1/Qgentic-AI", {
      x: 0.5, y: 5.15, w: 5.0, h: 0.4,
      fontSize: 11, color: C.textMain, fontFace: "Calibri", valign: "middle", margin: 0
    });
    slide.addText("5 / 5", { x: 8.8, y: 5.2, w: 0.8, h: 0.3, fontSize: 9, color: C.textDim, align: "right", fontFace: "Calibri" });
  }

  // Write
  const outputPath = path.resolve("/Users/oh_jason_zhang/Downloads/git_repo/Qgentic-AI/.claude/worktrees/strange-panini/Qgentic_AI_Keynote_Concise.pptx");
  await pres.writeFile({ fileName: outputPath });
  console.log("Concise presentation written to:", outputPath);
}

main().catch(err => { console.error(err); process.exit(1); });
