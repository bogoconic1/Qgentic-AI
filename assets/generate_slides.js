const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const path = require("path");

// Icon imports
const {
  FaClock, FaBrain, FaRobot, FaSearch, FaCode, FaCogs, FaShieldAlt,
  FaServer, FaDatabase, FaChartLine, FaLightbulb, FaRocket, FaTrophy,
  FaExclamationTriangle, FaNetworkWired, FaLayerGroup, FaBookOpen,
  FaSyncAlt, FaFlask, FaCheckCircle, FaTimesCircle, FaArrowRight,
  FaUsers, FaEye, FaLock, FaMedal
} = require("react-icons/fa");

// ---- Color Palette: Midnight Executive + Electric Blue ----
const C = {
  bgDark:    "0F1629",   // deep navy slide background
  bgMedium:  "1A2744",   // slightly lighter navy for cards
  bgCard:    "1E2F4D",   // card background
  accent:    "00B4D8",   // electric blue accent
  accentDim: "0891B2",   // darker accent
  accentBg:  "0D3D56",   // accent background for subtle highlights
  white:     "FFFFFF",
  textMain:  "E2E8F0",   // light gray text
  textMuted: "94A3B8",   // muted text
  textDim:   "64748B",   // dim text
  success:   "10B981",   // green
  warning:   "F59E0B",   // amber
  danger:    "EF4444",   // red
  purple:    "8B5CF6",
};

// ---- Helpers ----
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

function addSlideNumber(slide, num, total) {
  slide.addText(`${num} / ${total}`, {
    x: 8.5, y: 5.2, w: 1.2, h: 0.3,
    fontSize: 9, color: C.textDim, align: "right", fontFace: "Calibri"
  });
}

function addSectionLabel(slide, label) {
  slide.addText(label.toUpperCase(), {
    x: 0.6, y: 0.15, w: 3, h: 0.25,
    fontSize: 9, color: C.accentDim, fontFace: "Calibri", charSpacing: 3, bold: true, margin: 0
  });
}

function addTitle(slide, title) {
  slide.addText(title, {
    x: 0.6, y: 0.38, w: 8.8, h: 0.7,
    fontSize: 26, color: C.white, fontFace: "Arial Black", bold: true, margin: 0
  });
}

function addCard(slide, x, y, w, h) {
  slide.addShape("rect", {
    x, y, w, h,
    fill: { color: C.bgCard },
    shadow: makeShadow(),
    line: { color: C.accentDim, width: 0.5, dashType: "solid" }
  });
}

async function addIconCircle(slide, icon, x, y, color = C.accent) {
  // Circle background
  slide.addShape("ellipse", {
    x, y, w: 0.5, h: 0.5,
    fill: { color: C.accentBg }
  });
  const iconData = await iconToBase64Png(icon, color, 256);
  slide.addImage({ data: iconData, x: x + 0.1, y: y + 0.1, w: 0.3, h: 0.3 });
}

function addBullets(slide, items, x, y, w, h, opts = {}) {
  const textItems = items.map((item, i) => ({
    text: item,
    options: {
      bullet: true,
      breakLine: i < items.length - 1,
      fontSize: opts.fontSize || 14,
      color: opts.color || C.textMain,
      fontFace: "Calibri",
      paraSpaceAfter: 6
    }
  }));
  slide.addText(textItems, { x, y, w, h, valign: "top" });
}

const TOTAL = 18;
const LEADERBOARD_IMG = path.resolve("/Users/oh_jason_zhang/Downloads/git_repo/Qgentic-AI/docs/assets/csiro_lb.png");

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Qgentic AI";
  pres.title = "Qgentic AI: An Automated ML Competition Stack";

  // ========== SLIDE 1: Title ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    // Accent bar at top
    slide.addShape("rect", { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    // Main title
    slide.addText("QGENTIC AI", {
      x: 0.6, y: 1.2, w: 8.8, h: 0.8,
      fontSize: 48, color: C.white, fontFace: "Arial Black", bold: true, charSpacing: 4, margin: 0
    });
    slide.addText("An Automated ML Competition Stack", {
      x: 0.6, y: 2.0, w: 8.8, h: 0.5,
      fontSize: 22, color: C.accent, fontFace: "Calibri", margin: 0
    });
    // Separator line
    slide.addShape("rect", { x: 0.6, y: 2.7, w: 2.5, h: 0.04, fill: { color: C.accent } });
    slide.addText("Reducing Gold-Medal Effort from 200+ Hours to 20", {
      x: 0.6, y: 3.0, w: 8.8, h: 0.4,
      fontSize: 16, color: C.textMuted, fontFace: "Calibri", italic: true, margin: 0
    });
    // Bottom info bar
    slide.addShape("rect", { x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: C.bgMedium } });
    slide.addText("Technical Presentation  |  2026", {
      x: 0.6, y: 5.15, w: 8.8, h: 0.45,
      fontSize: 11, color: C.textDim, fontFace: "Calibri", margin: 0
    });
  }

  // ========== SLIDE 2: The Problem ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Background");
    addTitle(slide, "The Problem");
    addSlideNumber(slide, 2, TOTAL);

    // Big stat callout
    addCard(slide, 0.6, 1.3, 3.0, 1.6);
    slide.addText("200+", {
      x: 0.6, y: 1.35, w: 3.0, h: 0.8,
      fontSize: 52, color: C.accent, fontFace: "Arial Black", bold: true, align: "center", margin: 0
    });
    slide.addText("hours for a gold medal", {
      x: 0.6, y: 2.15, w: 3.0, h: 0.35,
      fontSize: 14, color: C.textMuted, fontFace: "Calibri", align: "center", margin: 0
    });
    slide.addText("against thousands of competitors", {
      x: 0.6, y: 2.5, w: 3.0, h: 0.3,
      fontSize: 11, color: C.textDim, fontFace: "Calibri", align: "center", margin: 0
    });

    // Second stat
    addCard(slide, 0.6, 3.2, 3.0, 1.3);
    slide.addText("~23 min", {
      x: 0.6, y: 3.25, w: 3.0, h: 0.65,
      fontSize: 36, color: C.warning, fontFace: "Arial Black", bold: true, align: "center", margin: 0
    });
    slide.addText("context-switch recovery time", {
      x: 0.6, y: 3.9, w: 3.0, h: 0.3,
      fontSize: 13, color: C.textMuted, fontFace: "Calibri", align: "center", margin: 0
    });

    // Bullet points on right
    addBullets(slide, [
      "Most time goes to repetitive maintenance: debugging, monitoring, iterating models",
      "Checking intermediate training results, restarting crashed runs, evaluating submissions",
      "The bottleneck is human endurance and availability, not intelligence",
      "Gold-medal work is ~80% disciplined execution, ~20% novel thinking"
    ], 4.0, 1.3, 5.5, 3.2);

    // Pie chart - time breakdown
    slide.addChart(pres.charts.DOUGHNUT, [{
      name: "Time", labels: ["Research", "Coding", "Debug/Monitor", "Evaluation"],
      values: [15, 25, 40, 20]
    }], {
      x: 4.2, y: 3.6, w: 2.2, h: 1.6,
      showPercent: true,
      chartColors: [C.accent, C.purple, C.danger, C.warning],
      dataLabelColor: C.white,
      dataLabelFontSize: 8,
      showLegend: true,
      legendPos: "r",
      legendColor: C.textMuted,
      legendFontSize: 8
    });
  }

  // ========== SLIDE 3: Key Insight ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Background");
    addTitle(slide, "The Key Insight");
    addSlideNumber(slide, 3, TOTAL);

    // Left: funnel visual
    // 200+ hours bar
    slide.addShape("rect", { x: 0.6, y: 1.5, w: 4.0, h: 0.6, fill: { color: C.danger } });
    slide.addText("200+ hours manual effort", {
      x: 0.6, y: 1.5, w: 4.0, h: 0.6,
      fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
    });

    // Arrow down
    slide.addText("\u25BC", {
      x: 2.1, y: 2.15, w: 0.8, h: 0.4,
      fontSize: 18, color: C.accent, fontFace: "Calibri", align: "center"
    });

    // Agent execution bar
    slide.addShape("rect", { x: 1.2, y: 2.6, w: 2.6, h: 0.6, fill: { color: C.accentDim } });
    slide.addText("Automated Agent Execution", {
      x: 1.2, y: 2.6, w: 2.6, h: 0.6,
      fontSize: 13, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
    });

    // Arrow down
    slide.addText("\u25BC", {
      x: 2.1, y: 3.25, w: 0.8, h: 0.4,
      fontSize: 18, color: C.accent, fontFace: "Calibri", align: "center"
    });

    // 20 hours bar
    slide.addShape("rect", { x: 1.8, y: 3.7, w: 1.5, h: 0.6, fill: { color: C.success } });
    slide.addText("~20 hrs", {
      x: 1.8, y: 3.7, w: 1.5, h: 0.6,
      fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
    });

    slide.addText("90% reduction", {
      x: 1.5, y: 4.4, w: 2.0, h: 0.3,
      fontSize: 12, color: C.success, fontFace: "Calibri", bold: true, align: "center"
    });

    // Right: key points
    addCard(slide, 5.2, 1.4, 4.4, 3.4);
    const points = [
      { label: "Strategic Thinking", pct: "~20%", desc: "Novel approaches, competition-specific insights", color: C.accent },
      { label: "Execution Work", pct: "~80%", desc: "Code gen, debugging, monitoring, evaluation", color: C.warning },
      { label: "LLM Capability", pct: "NOW", desc: "LLMs can handle structured execution loops", color: C.success },
    ];
    points.forEach((p, i) => {
      const py = 1.6 + i * 1.0;
      slide.addText(p.pct, {
        x: 5.4, y: py, w: 1.0, h: 0.4,
        fontSize: 22, color: p.color, fontFace: "Arial Black", bold: true, margin: 0
      });
      slide.addText(p.label, {
        x: 6.5, y: py, w: 2.9, h: 0.3,
        fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(p.desc, {
        x: 6.5, y: py + 0.3, w: 2.9, h: 0.3,
        fontSize: 11, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });

    slide.addText("Goal: Human as strategic director, agents handle the grind", {
      x: 5.4, y: 4.35, w: 4.0, h: 0.3,
      fontSize: 11, color: C.accent, fontFace: "Calibri", italic: true, margin: 0
    });
  }

  // ========== SLIDE 4: Related Work - AutoML ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Related Work");
    addTitle(slide, "AutoML Systems");
    addSlideNumber(slide, 4, TOTAL);

    // Comparison table
    const headers = [
      [
        { text: "System", options: { fill: { color: C.accentDim }, color: C.white, bold: true, fontSize: 12, fontFace: "Calibri" } },
        { text: "Code Gen", options: { fill: { color: C.accentDim }, color: C.white, bold: true, fontSize: 12, fontFace: "Calibri" } },
        { text: "Reasoning", options: { fill: { color: C.accentDim }, color: C.white, bold: true, fontSize: 12, fontFace: "Calibri" } },
        { text: "Competition\nAware", options: { fill: { color: C.accentDim }, color: C.white, bold: true, fontSize: 12, fontFace: "Calibri" } },
        { text: "Multi-Model", options: { fill: { color: C.accentDim }, color: C.white, bold: true, fontSize: 12, fontFace: "Calibri" } }
      ]
    ];
    const mkRow = (name, cg, r, ca, mm) => [
      { text: name, options: { fill: { color: C.bgCard }, color: C.textMain, fontSize: 12, fontFace: "Calibri" } },
      { text: cg, options: { fill: { color: C.bgCard }, color: cg === "\u2713" ? C.success : C.danger, fontSize: 14, fontFace: "Calibri", align: "center" } },
      { text: r, options: { fill: { color: C.bgCard }, color: r === "\u2713" ? C.success : C.danger, fontSize: 14, fontFace: "Calibri", align: "center" } },
      { text: ca, options: { fill: { color: C.bgCard }, color: ca === "\u2713" ? C.success : C.danger, fontSize: 14, fontFace: "Calibri", align: "center" } },
      { text: mm, options: { fill: { color: C.bgCard }, color: mm === "\u2713" ? C.success : C.danger, fontSize: 14, fontFace: "Calibri", align: "center" } }
    ];

    slide.addTable([
      ...headers,
      mkRow("Auto-sklearn", "\u2717", "\u2717", "\u2717", "\u2713"),
      mkRow("AutoGluon", "\u2717", "\u2717", "\u2717", "\u2713"),
      mkRow("H2O AutoML", "\u2717", "\u2717", "\u2717", "\u2713"),
      mkRow("Qgentic AI", "\u2713", "\u2713", "\u2713", "\u2713"),
    ], {
      x: 0.6, y: 1.3, w: 8.8, colW: [2.2, 1.5, 1.5, 1.8, 1.8],
      border: { pt: 0.5, color: C.bgMedium },
      rowH: [0.45, 0.45, 0.45, 0.45, 0.45]
    });

    addBullets(slide, [
      "AutoML systems optimize within predefined pipelines and fixed search spaces",
      "No ability to read a competition description and devise a novel approach",
      "No code generation -- only hyperparameter tuning and model selection",
      "Qgentic AI combines reasoning, code generation, and competition awareness"
    ], 0.6, 3.7, 8.8, 1.8);
  }

  // ========== SLIDE 5: Related Work - LLM Agents ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Related Work");
    addTitle(slide, "LLM Code Agents");
    addSlideNumber(slide, 5, TOTAL);

    // 2x2 matrix
    const matrixX = 0.6, matrixY = 1.4, matrixW = 4.5, matrixH = 3.5;
    // Axes
    slide.addShape("rect", { x: matrixX + 1.2, y: matrixY, w: 0.02, h: matrixH, fill: { color: C.textDim } });
    slide.addShape("rect", { x: matrixX + 1.2, y: matrixY + matrixH / 2, w: matrixW - 1.2, h: 0.02, fill: { color: C.textDim } });

    // Axis labels
    slide.addText("Full Pipeline", { x: matrixX, y: matrixY + 0.2, w: 1.1, h: 0.4, fontSize: 10, color: C.textMuted, fontFace: "Calibri", align: "right" });
    slide.addText("Coding Assist", { x: matrixX, y: matrixY + 2.0, w: 1.1, h: 0.4, fontSize: 10, color: C.textMuted, fontFace: "Calibri", align: "right" });
    slide.addText("Single-shot", { x: matrixX + 1.5, y: matrixY + 3.5, w: 1.5, h: 0.3, fontSize: 10, color: C.textMuted, fontFace: "Calibri", align: "center" });
    slide.addText("Iterative", { x: matrixX + 3.5, y: matrixY + 3.5, w: 1.5, h: 0.3, fontSize: 10, color: C.textMuted, fontFace: "Calibri", align: "center" });

    // Quadrant items
    slide.addText("AlphaCode", { x: 2.0, y: 1.6, w: 1.6, h: 0.3, fontSize: 11, color: C.warning, fontFace: "Calibri", align: "center" });
    slide.addText("AIDE / AutoKaggle", { x: 3.5, y: 1.6, w: 2.0, h: 0.3, fontSize: 11, color: C.purple, fontFace: "Calibri", align: "center" });
    slide.addText("Copilot / Cursor", { x: 2.0, y: 3.2, w: 1.6, h: 0.3, fontSize: 11, color: C.textMuted, fontFace: "Calibri", align: "center" });

    // Qgentic highlight
    slide.addShape("rect", { x: 3.6, y: 1.95, w: 1.8, h: 0.5, fill: { color: C.accentBg }, line: { color: C.accent, width: 1.5 } });
    slide.addText("Qgentic AI", { x: 3.6, y: 1.95, w: 1.8, h: 0.5, fontSize: 13, color: C.accent, fontFace: "Calibri", bold: true, align: "center", valign: "middle" });

    // Right side - key differentiators
    addCard(slide, 5.5, 1.4, 4.1, 3.5);
    slide.addText("The Gap", {
      x: 5.7, y: 1.5, w: 3.7, h: 0.35,
      fontSize: 16, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });
    addBullets(slide, [
      "No system combines multi-agent research + model recommendation + parallel execution + cross-model learning",
      "Copilot/Cursor: code completion, not end-to-end pipeline execution",
      "AlphaCode/CodeGen: single-shot generation, no iterative refinement",
      "AIDE, MLE-bench, AutoKaggle: closest, but lack multi-agent coordination and shared learning"
    ], 5.7, 1.9, 3.7, 2.8, { fontSize: 12 });
  }

  // ========== SLIDE 6: Architecture Overview ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Architecture");
    addTitle(slide, "System Architecture Overview");
    addSlideNumber(slide, 6, TOTAL);

    // Pipeline flow boxes
    const agents = [
      { name: "Starter\nAgent", x: 0.3, color: C.purple },
      { name: "Researcher\nAgent", x: 2.1, color: C.accentDim },
      { name: "Model\nRecommender", x: 3.9, color: C.warning },
      { name: "Developer\nAgent (x N)", x: 5.7, color: C.success },
      { name: "Results\nAggregation", x: 7.7, color: C.accent },
    ];

    agents.forEach(a => {
      slide.addShape("rect", {
        x: a.x, y: 1.6, w: 1.6, h: 0.95,
        fill: { color: a.color },
        shadow: makeShadow()
      });
      slide.addText(a.name, {
        x: a.x, y: 1.6, w: 1.6, h: 0.95,
        fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
      });
    });

    // Arrows between boxes
    [1.9, 3.7, 5.5, 7.4].forEach(ax => {
      slide.addText("\u25B6", {
        x: ax, y: 1.85, w: 0.2, h: 0.4,
        fontSize: 14, color: C.accent, fontFace: "Calibri", align: "center"
      });
    });

    // Orchestrator bar underneath
    slide.addShape("rect", {
      x: 0.3, y: 2.85, w: 9.0, h: 0.5,
      fill: { color: C.bgCard },
      line: { color: C.accent, width: 1, dashType: "dash" }
    });
    slide.addText("ORCHESTRATOR  \u2014  Phase Sequencing  |  Parallel Execution  |  GPU/CPU Isolation  |  Checkpointing", {
      x: 0.5, y: 2.85, w: 8.6, h: 0.5,
      fontSize: 10, color: C.accent, fontFace: "Calibri", align: "center", valign: "middle"
    });

    // Bottom detail cards
    const details = [
      { title: "LLM Backend", desc: "Gemini 3.1 Pro Preview\nStructured Pydantic outputs", x: 0.3 },
      { title: "Tool Calling", desc: "Agentic loops with\nexecute, search, scrape tools", x: 2.55 },
      { title: "Parallel Workers", desc: "Up to N models in parallel\nwith GPU/CPU isolation", x: 4.8 },
      { title: "Cross-Model Learning", desc: "Shared suggestion pool\nacross all baselines", x: 7.05 },
    ];

    details.forEach(d => {
      addCard(slide, d.x, 3.7, 2.05, 1.3);
      slide.addText(d.title, {
        x: d.x + 0.15, y: 3.8, w: 1.75, h: 0.3,
        fontSize: 11, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(d.desc, {
        x: d.x + 0.15, y: 4.15, w: 1.75, h: 0.7,
        fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });
  }

  // ========== SLIDE 7: StarterAgent ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Architecture");
    addTitle(slide, "StarterAgent \u2014 Task Classification");
    addSlideNumber(slide, 7, TOTAL);

    // Input box
    addCard(slide, 0.6, 1.5, 2.5, 1.2);
    await addIconCircle(slide, FaBookOpen, 1.55, 1.6, C.accent);
    slide.addText("description.md", {
      x: 0.8, y: 2.15, w: 2.1, h: 0.3,
      fontSize: 13, color: C.textMain, fontFace: "Calibri", align: "center", bold: true, margin: 0
    });
    slide.addText("Competition brief", {
      x: 0.8, y: 2.4, w: 2.1, h: 0.2,
      fontSize: 10, color: C.textMuted, fontFace: "Calibri", align: "center", margin: 0
    });

    // Arrow
    slide.addText("\u25B6", { x: 3.2, y: 1.85, w: 0.4, h: 0.4, fontSize: 20, color: C.accent, align: "center" });

    // Agent box
    slide.addShape("rect", { x: 3.8, y: 1.5, w: 2.5, h: 1.2, fill: { color: C.purple }, shadow: makeShadow() });
    slide.addText("StarterAgent", { x: 3.8, y: 1.6, w: 2.5, h: 0.4, fontSize: 15, color: C.white, fontFace: "Calibri", bold: true, align: "center", margin: 0 });
    slide.addText("Single LLM call\n+ Google Search", { x: 3.8, y: 2.0, w: 2.5, h: 0.5, fontSize: 11, color: "D8D0F0", fontFace: "Calibri", align: "center", margin: 0 });

    // Arrow
    slide.addText("\u25B6", { x: 6.4, y: 1.85, w: 0.4, h: 0.4, fontSize: 20, color: C.accent, align: "center" });

    // Output box
    addCard(slide, 7.0, 1.5, 2.6, 1.2);
    slide.addText("starter_suggestions.json", {
      x: 7.1, y: 1.6, w: 2.4, h: 0.3,
      fontSize: 12, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });
    slide.addText("{\n  task_types: [\"tabular\"],\n  summary: \"Predict biomass...\"\n}", {
      x: 7.1, y: 1.9, w: 2.4, h: 0.7,
      fontSize: 9, color: C.textMuted, fontFace: "Consolas", margin: 0
    });

    // Task type cards below
    const types = [
      { label: "Tabular", desc: "Structured data, feature engineering" },
      { label: "NLP", desc: "Text classification, NER, summarization" },
      { label: "Computer Vision", desc: "Image classification, segmentation" },
      { label: "Time Series", desc: "Forecasting, anomaly detection" },
    ];
    types.forEach((t, i) => {
      const tx = 0.6 + i * 2.35;
      addCard(slide, tx, 3.2, 2.1, 1.2);
      slide.addText(t.label, {
        x: tx + 0.15, y: 3.3, w: 1.8, h: 0.35,
        fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(t.desc, {
        x: tx + 0.15, y: 3.7, w: 1.8, h: 0.4,
        fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });

    slide.addText("Downstream agents use task_types to tailor research, model selection, and code generation", {
      x: 0.6, y: 4.6, w: 8.8, h: 0.3,
      fontSize: 11, color: C.textDim, fontFace: "Calibri", italic: true
    });
  }

  // ========== SLIDE 8: ResearcherAgent ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Architecture");
    addTitle(slide, "ResearcherAgent \u2014 Deep Research Loop");
    addSlideNumber(slide, 8, TOTAL);

    // Agentic loop diagram
    // Central loop
    slide.addShape("rect", { x: 0.6, y: 1.5, w: 4.5, h: 3.0, fill: { color: C.bgMedium }, line: { color: C.accentDim, width: 0.5 } });
    slide.addText("AGENTIC LOOP (up to 512 steps)", {
      x: 0.8, y: 1.55, w: 4.1, h: 0.3,
      fontSize: 10, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });

    // Loop steps
    const steps = [
      { label: "LLM Decides Tool", y: 2.0, color: C.accentDim },
      { label: "Execute Tool", y: 2.55, color: C.success },
      { label: "Return Result", y: 3.1, color: C.warning },
      { label: "LLM Reasons", y: 3.65, color: C.purple },
    ];
    steps.forEach((s, i) => {
      slide.addShape("rect", { x: 1.3, y: s.y, w: 2.5, h: 0.4, fill: { color: s.color } });
      slide.addText(s.label, {
        x: 1.3, y: s.y, w: 2.5, h: 0.4,
        fontSize: 12, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
      });
      if (i < steps.length - 1) {
        slide.addText("\u25BC", { x: 2.3, y: s.y + 0.38, w: 0.5, h: 0.2, fontSize: 12, color: C.accent, align: "center" });
      }
    });
    // Loop back arrow text
    slide.addText("\u21BB", { x: 4.0, y: 2.8, w: 0.5, h: 0.5, fontSize: 28, color: C.accent, fontFace: "Calibri" });

    // Tools on right
    const tools = [
      { name: "execute_python", desc: "Run EDA code, generate plots\nAuto-ingests images (max 6/step)", icon: FaCode },
      { name: "read_research_paper", desc: "ArXiv PDF summarization\nExtracts methods, results, ablations", icon: FaBookOpen },
      { name: "scrape_web_page", desc: "Firecrawl web scraping\nCompetition forums, blog posts", icon: FaSearch },
    ];

    for (let i = 0; i < tools.length; i++) {
      const t = tools[i];
      const ty = 1.5 + i * 1.05;
      addCard(slide, 5.5, ty, 4.1, 0.9);
      await addIconCircle(slide, t.icon, 5.7, ty + 0.12, C.accent);
      slide.addText(t.name, {
        x: 6.35, y: ty + 0.05, w: 3.0, h: 0.3,
        fontSize: 13, color: C.white, fontFace: "Consolas", bold: true, margin: 0
      });
      slide.addText(t.desc, {
        x: 6.35, y: ty + 0.38, w: 3.0, h: 0.45,
        fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    }

    // Output
    slide.addShape("rect", { x: 5.5, y: 4.25, w: 4.1, h: 0.5, fill: { color: C.accentBg } });
    slide.addText("Output: plan.md \u2014 research plan for downstream agents", {
      x: 5.65, y: 4.25, w: 3.8, h: 0.5,
      fontSize: 11, color: C.accent, fontFace: "Calibri", bold: true, valign: "middle", margin: 0
    });
  }

  // ========== SLIDE 9: ModelRecommenderAgent ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Architecture");
    addTitle(slide, "ModelRecommenderAgent \u2014 Strategy Selection");
    addSlideNumber(slide, 9, TOTAL);

    // Funnel visualization
    // Stage 1
    slide.addShape("rect", { x: 0.6, y: 1.5, w: 4.0, h: 0.6, fill: { color: C.accentDim } });
    slide.addText("Stage 1: LLM selects ~16 candidates (web search enabled)", {
      x: 0.6, y: 1.5, w: 4.0, h: 0.6,
      fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
    });

    slide.addText("\u25BC", { x: 2.3, y: 2.1, w: 0.5, h: 0.3, fontSize: 14, color: C.accent, align: "center" });

    // Stage 2
    slide.addShape("rect", { x: 1.0, y: 2.4, w: 3.2, h: 0.6, fill: { color: C.warning } });
    slide.addText("Stage 2: Paper analysis for all candidates", {
      x: 1.0, y: 2.4, w: 3.2, h: 0.6,
      fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
    });

    slide.addText("\u25BC", { x: 2.3, y: 3.0, w: 0.5, h: 0.3, fontSize: 14, color: C.accent, align: "center" });

    // Stage 3
    slide.addShape("rect", { x: 1.5, y: 3.3, w: 2.2, h: 0.6, fill: { color: C.success } });
    slide.addText("Stage 3: Refine to ~8 models", {
      x: 1.5, y: 3.3, w: 2.2, h: 0.6,
      fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
    });

    // Right side - 4 recommendation axes
    const axes = [
      { name: "Preprocessing", desc: "Data transforms, augmentation, normalization", color: C.accent },
      { name: "Loss Functions", desc: "Exactly 1 MUST_HAVE, multiple NICE_TO_HAVE", color: C.purple },
      { name: "Hyperparameters", desc: "Learning rate, batch size, architecture configs", color: C.warning },
      { name: "Inference", desc: "TTA, ensembling, post-processing strategies", color: C.success },
    ];

    slide.addText("Per-Model Recommendations", {
      x: 5.2, y: 1.4, w: 4.4, h: 0.3,
      fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, margin: 0
    });

    axes.forEach((a, i) => {
      const ay = 1.85 + i * 0.65;
      slide.addShape("rect", { x: 5.2, y: ay, w: 0.08, h: 0.5, fill: { color: a.color } });
      slide.addText(a.name, {
        x: 5.5, y: ay, w: 2.0, h: 0.25,
        fontSize: 12, color: C.white, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(a.desc, {
        x: 5.5, y: ay + 0.25, w: 4.0, h: 0.25,
        fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });

    // MUST_HAVE vs NICE_TO_HAVE
    addCard(slide, 5.2, 4.1, 4.4, 0.7);
    slide.addText("MUST_HAVE", {
      x: 5.4, y: 4.15, w: 1.5, h: 0.3,
      fontSize: 12, color: C.success, fontFace: "Calibri", bold: true, margin: 0
    });
    slide.addText("= applied in first iteration (NOW)", {
      x: 6.7, y: 4.15, w: 2.8, h: 0.3,
      fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
    });
    slide.addText("NICE_TO_HAVE", {
      x: 5.4, y: 4.45, w: 1.5, h: 0.3,
      fontSize: 12, color: C.warning, fontFace: "Calibri", bold: true, margin: 0
    });
    slide.addText("= fed to SOTA engine for later iterations", {
      x: 6.9, y: 4.45, w: 2.8, h: 0.3,
      fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
    });
  }

  // ========== SLIDE 10: DeveloperAgent ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Methodology");
    addTitle(slide, "DeveloperAgent \u2014 Code Generation & Iteration");
    addSlideNumber(slide, 10, TOTAL);

    // Iteration loop diagram
    const loopSteps = [
      { label: "Generate\ntrain.py", x: 0.6, y: 1.8, w: 1.3, h: 0.7, color: C.accentDim },
      { label: "Guardrails\nCheck", x: 2.2, y: 1.8, w: 1.3, h: 0.7, color: C.danger },
      { label: "Execute\nCode", x: 3.8, y: 1.8, w: 1.3, h: 0.7, color: C.success },
      { label: "Monitor\nLogs (120s)", x: 5.4, y: 1.8, w: 1.3, h: 0.7, color: C.warning },
      { label: "Score\nEvaluation", x: 7.0, y: 1.8, w: 1.3, h: 0.7, color: C.purple },
      { label: "SOTA\nAnalysis", x: 8.3, y: 2.8, w: 1.3, h: 0.7, color: C.accent },
    ];

    loopSteps.forEach(s => {
      slide.addShape("rect", { x: s.x, y: s.y, w: s.w, h: s.h, fill: { color: s.color }, shadow: makeShadow() });
      slide.addText(s.label, {
        x: s.x, y: s.y, w: s.w, h: s.h,
        fontSize: 10, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
      });
    });

    // Arrows
    [1.9, 3.5, 5.1, 6.7].forEach(ax => {
      slide.addText("\u25B6", { x: ax, y: 1.95, w: 0.3, h: 0.35, fontSize: 12, color: C.accent, align: "center" });
    });
    // Down arrow from Score to SOTA
    slide.addText("\u25BC", { x: 8.6, y: 2.5, w: 0.5, h: 0.3, fontSize: 12, color: C.accent, align: "center" });
    // Loop back arrow
    slide.addShape("rect", { x: 0.6, y: 3.25, w: 8.0, h: 0.02, fill: { color: C.accent } });
    slide.addText("\u25C0 Next Iteration", { x: 0.6, y: 3.3, w: 2.0, h: 0.25, fontSize: 10, color: C.accent, fontFace: "Calibri" });

    // Key features below
    const features = [
      { title: "Log Monitoring", desc: "LLM polls every 120s\nDetects crashes, NaN losses,\nstalled training, overfitting" },
      { title: "Stack Trace Fix", desc: "Fine-tuned model first\nFalls back to web search\n+ general LLM diagnosis" },
      { title: "investigate_library", desc: "Reads installed package\nsource code to resolve\nAPI questions accurately" },
      { title: "Code Injection", desc: "Auto-injects CPU affinity,\nGPU assignment, BASE_DIR,\nenvironment setup" },
    ];

    features.forEach((f, i) => {
      const fx = 0.6 + i * 2.4;
      addCard(slide, fx, 3.8, 2.15, 1.4);
      slide.addText(f.title, {
        x: fx + 0.15, y: 3.9, w: 1.85, h: 0.3,
        fontSize: 12, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(f.desc, {
        x: fx + 0.15, y: 4.2, w: 1.85, h: 0.8,
        fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });
  }

  // ========== SLIDE 11: Cross-Model Learning ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Methodology");
    addTitle(slide, "Cross-Model Learning");
    addSlideNumber(slide, 11, TOTAL);

    // Model A
    slide.addShape("rect", { x: 0.6, y: 1.6, w: 2.8, h: 1.4, fill: { color: C.accentDim }, shadow: makeShadow() });
    slide.addText("Model A\n(e.g., XGBoost)", {
      x: 0.6, y: 1.7, w: 2.8, h: 0.6,
      fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, align: "center", margin: 0
    });
    slide.addText("\"Cosine annealing\nimproved score by 0.02\"", {
      x: 0.8, y: 2.3, w: 2.4, h: 0.55,
      fontSize: 10, color: "B0D0E8", fontFace: "Calibri", italic: true, align: "center", margin: 0
    });

    // Model B
    slide.addShape("rect", { x: 0.6, y: 3.4, w: 2.8, h: 1.4, fill: { color: C.purple }, shadow: makeShadow() });
    slide.addText("Model B\n(e.g., LightGBM)", {
      x: 0.6, y: 3.5, w: 2.8, h: 0.6,
      fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, align: "center", margin: 0
    });
    slide.addText("\"Feature selection with\nMI improved by 0.015\"", {
      x: 0.8, y: 4.1, w: 2.4, h: 0.55,
      fontSize: 10, color: "D0B0E8", fontFace: "Calibri", italic: true, align: "center", margin: 0
    });

    // Shared pool in center
    addCard(slide, 3.8, 1.8, 2.8, 2.8);
    slide.addText("Shared Suggestion Pool", {
      x: 3.8, y: 1.9, w: 2.8, h: 0.4,
      fontSize: 13, color: C.accent, fontFace: "Calibri", bold: true, align: "center", margin: 0
    });
    slide.addText("Thread-safe class-level\n_shared_suggestions", {
      x: 3.8, y: 2.3, w: 2.8, h: 0.4,
      fontSize: 10, color: C.textMuted, fontFace: "Consolas", align: "center", margin: 0
    });

    // Suggestion entries
    const entries = [
      { text: "\u2713 Cosine annealing (+0.02)", color: C.success },
      { text: "\u2713 MI feature selection (+0.015)", color: C.success },
      { text: "\u2717 Label smoothing (-0.01)", color: C.danger },
      { text: "\u2717 Heavy augmentation (-0.005)", color: C.danger },
    ];
    entries.forEach((e, i) => {
      slide.addText(e.text, {
        x: 4.0, y: 2.8 + i * 0.35, w: 2.4, h: 0.3,
        fontSize: 10, color: e.color, fontFace: "Calibri", margin: 0
      });
    });

    // Bidirectional arrows
    slide.addText("\u2194", { x: 3.35, y: 2.1, w: 0.5, h: 0.4, fontSize: 20, color: C.accent, align: "center" });
    slide.addText("\u2194", { x: 3.35, y: 3.8, w: 0.5, h: 0.4, fontSize: 20, color: C.accent, align: "center" });

    // Right side - key benefits
    addCard(slide, 7.0, 1.6, 2.7, 3.2);
    slide.addText("Benefits", {
      x: 7.15, y: 1.7, w: 2.4, h: 0.3,
      fontSize: 14, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });
    addBullets(slide, [
      "Prevents redundant exploration of failed strategies",
      "Successful techniques propagate to sibling models",
      "Blacklisted ideas automatically excluded",
      "Emergent collective optimization across all baselines"
    ], 7.15, 2.1, 2.4, 2.5, { fontSize: 11 });
  }

  // ========== SLIDE 12: Safety & Guardrails ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Methodology");
    addTitle(slide, "Safety & Guardrails");
    addSlideNumber(slide, 12, TOTAL);

    // 3 shield layers
    const layers = [
      { name: "Layer 1: AST Static Analysis", desc: "Checks logging.basicConfig order\nDetects train/test data partition patterns\nPython AST-based, zero LLM cost", color: C.accentDim, y: 1.4 },
      { name: "Layer 2: LLM Leakage Review", desc: "Gemini 3.1 Pro reviews code for data leakage\nDetects train/test contamination\nStructured LeakageReviewResponse output", color: C.warning, y: 2.7 },
      { name: "Layer 3: LLM Code Safety", desc: "Checks for eval/exec, command injection\nCredential leakage detection\nGemini 2.5 Flash (fast, low cost)", color: C.danger, y: 4.0 },
    ];

    for (const l of layers) {
      await addIconCircle(slide, FaShieldAlt, 0.7, l.y + 0.15, l.color);
      slide.addText(l.name, {
        x: 1.4, y: l.y, w: 4.0, h: 0.35,
        fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(l.desc, {
        x: 1.4, y: l.y + 0.4, w: 4.5, h: 0.7,
        fontSize: 11, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    }

    // Right side - pipeline flow
    addCard(slide, 6.2, 1.4, 3.4, 3.8);
    slide.addText("Guardrail Pipeline", {
      x: 6.4, y: 1.5, w: 3.0, h: 0.3,
      fontSize: 14, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });

    const pipeSteps = [
      { label: "Code Generated", color: C.accentDim },
      { label: "evaluate_guardrails()", color: C.warning },
      { label: "All checks pass?", color: C.purple },
      { label: "Execute code", color: C.success },
    ];
    pipeSteps.forEach((p, i) => {
      const py = 2.0 + i * 0.7;
      slide.addShape("rect", { x: 6.6, y: py, w: 2.6, h: 0.45, fill: { color: p.color } });
      slide.addText(p.label, {
        x: 6.6, y: py, w: 2.6, h: 0.45,
        fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
      });
      if (i < pipeSteps.length - 1) {
        slide.addText("\u25BC", { x: 7.7, y: py + 0.42, w: 0.4, h: 0.3, fontSize: 12, color: C.accent, align: "center" });
      }
    });

    // Blocked path
    slide.addText("\u2717 Blocked \u2192 feedback to LLM for regeneration", {
      x: 6.4, y: 4.9, w: 3.2, h: 0.25,
      fontSize: 10, color: C.danger, fontFace: "Calibri", italic: true, margin: 0
    });
  }

  // ========== SLIDE 13: Infrastructure ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Methodology");
    addTitle(slide, "Infrastructure \u2014 Parallel Execution");
    addSlideNumber(slide, 13, TOTAL);

    // GPU isolation modes
    slide.addText("GPU Isolation Modes", {
      x: 0.6, y: 1.4, w: 4.5, h: 0.3,
      fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, margin: 0
    });

    const modes = [
      { name: "MIG", desc: "NVIDIA Multi-Instance GPU\nPartition single GPU into\nindependent instances", color: C.accentDim },
      { name: "Multi-GPU", desc: "Assign different GPUs\nto different workers\n(CUDA_VISIBLE_DEVICES)", color: C.success },
      { name: "CPU-Only", desc: "No GPU isolation\nCPU affinity pinning\nvia psutil", color: C.warning },
    ];

    modes.forEach((m, i) => {
      const mx = 0.6 + i * 1.55;
      addCard(slide, mx, 1.8, 1.4, 1.7);
      slide.addText(m.name, {
        x: mx + 0.1, y: 1.9, w: 1.2, h: 0.35,
        fontSize: 13, color: m.color, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(m.desc, {
        x: mx + 0.1, y: 2.3, w: 1.2, h: 0.9,
        fontSize: 9, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });

    // Right side - server diagram
    addCard(slide, 5.2, 1.4, 4.4, 2.1);
    slide.addText("Resource Pool Architecture", {
      x: 5.4, y: 1.5, w: 4.0, h: 0.3,
      fontSize: 13, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });

    // GPU slots
    for (let i = 0; i < 4; i++) {
      const gx = 5.5 + i * 1.0;
      const colors = [C.accentDim, C.success, C.warning, C.purple];
      slide.addShape("rect", { x: gx, y: 2.0, w: 0.85, h: 0.5, fill: { color: colors[i] } });
      slide.addText(`GPU ${i}`, {
        x: gx, y: 2.0, w: 0.85, h: 0.5,
        fontSize: 10, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
      });
      slide.addText(`Worker ${i + 1}`, {
        x: gx, y: 2.55, w: 0.85, h: 0.3,
        fontSize: 9, color: C.textMuted, fontFace: "Calibri", align: "center"
      });
    }

    slide.addText("ThreadPoolExecutor  +  Queue-based dynamic allocation", {
      x: 5.4, y: 2.9, w: 4.0, h: 0.3,
      fontSize: 10, color: C.textDim, fontFace: "Calibri", margin: 0
    });

    // Bottom section - time limits and isolation
    const configs = [
      { label: "5 days", desc: "Baseline development\ntime budget", color: C.accent },
      { label: "12 hrs", desc: "Per-script execution\ntimeout", color: C.warning },
      { label: "120s", desc: "Log monitoring\npoll interval", color: C.success },
      { label: "Isolated", desc: "Conda environment\nper model", color: C.purple },
    ];

    configs.forEach((c, i) => {
      const cx = 0.6 + i * 2.4;
      addCard(slide, cx, 3.9, 2.15, 1.2);
      slide.addText(c.label, {
        x: cx + 0.15, y: 4.0, w: 1.85, h: 0.4,
        fontSize: 22, color: c.color, fontFace: "Arial Black", bold: true, margin: 0
      });
      slide.addText(c.desc, {
        x: cx + 0.15, y: 4.45, w: 1.85, h: 0.5,
        fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });
  }

  // ========== SLIDE 14: Checkpoint & HITL ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Methodology");
    addTitle(slide, "Checkpoint & Human-in-the-Loop");
    addSlideNumber(slide, 14, TOTAL);

    // Timeline diagram
    slide.addText("Version Timeline with Rollback", {
      x: 0.6, y: 1.4, w: 5, h: 0.3,
      fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, margin: 0
    });

    // Timeline line
    slide.addShape("rect", { x: 0.8, y: 2.2, w: 8.5, h: 0.03, fill: { color: C.textDim } });

    // Version dots
    const versions = [
      { label: "V1", score: "0.58", x: 1.2, color: C.textDim },
      { label: "V2", score: "0.61", x: 3.0, color: C.accentDim },
      { label: "V3", score: "0.64", x: 4.8, color: C.success, star: true },
      { label: "V4", score: "0.59", x: 6.6, color: C.danger },
      { label: "Rollback", score: "", x: 8.2, color: C.warning },
    ];

    versions.forEach(v => {
      slide.addShape("ellipse", { x: v.x, y: 2.0, w: 0.4, h: 0.4, fill: { color: v.color } });
      const labelW = v.label.length > 3 ? 1.2 : 0.6;
      slide.addText(v.label, {
        x: v.x - (labelW / 2) + 0.2, y: 1.6, w: labelW, h: 0.3,
        fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, align: "center"
      });
      if (v.score) {
        slide.addText(v.score, {
          x: v.x - 0.2, y: 2.45, w: 0.8, h: 0.25,
          fontSize: 10, color: C.textMuted, fontFace: "Consolas", align: "center"
        });
      }
      if (v.star) {
        slide.addText("\u2605 best", {
          x: v.x - 0.2, y: 2.7, w: 0.8, h: 0.2,
          fontSize: 9, color: C.success, fontFace: "Calibri", align: "center"
        });
      }
    });

    // Rollback arrow
    slide.addShape("rect", { x: 5.4, y: 2.95, w: 3.0, h: 0.02, fill: { color: C.warning } });
    slide.addText("\u25C0 rollback to V3", {
      x: 5.8, y: 3.0, w: 2.0, h: 0.25,
      fontSize: 9, color: C.warning, fontFace: "Calibri", align: "center"
    });

    // SQLite + HITL details
    addCard(slide, 0.6, 3.5, 4.3, 1.7);
    slide.addText("SQLite Checkpoint System", {
      x: 0.8, y: 3.6, w: 3.9, h: 0.3,
      fontSize: 13, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });
    addBullets(slide, [
      "Saves full agent state per version",
      "Scores, suggestions, conversation history",
      "Rollback: move to _trash/, delete DB rows",
      "--rollback-to-version N CLI flag"
    ], 0.8, 3.95, 3.9, 1.1, { fontSize: 11 });

    addCard(slide, 5.2, 3.5, 4.4, 1.7);
    slide.addText("Human-in-the-Loop (INSTRUCTIONS.md)", {
      x: 5.4, y: 3.6, w: 4.0, h: 0.3,
      fontSize: 13, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });
    addBullets(slide, [
      "# Researcher Instructions \u2014 guide research",
      "# Developer Instructions \u2014 guide code gen",
      "# Models \u2014 override model selection",
      "hitl_sota: true \u2192 pause for human approval"
    ], 5.4, 3.95, 4.0, 1.1, { fontSize: 11 });
  }

  // ========== SLIDE 15: Results ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Results");
    addTitle(slide, "CSIRO Biomass Competition");
    addSlideNumber(slide, 15, TOTAL);

    // Big stats
    const stats = [
      { value: "0.63772", label: "Leaderboard Score", color: C.accent },
      { value: "32 / 3,802", label: "Final Ranking", color: C.success },
      { value: "Top 1%", label: "Silver Medal", color: C.warning },
    ];

    stats.forEach((s, i) => {
      const sx = 0.6 + i * 3.15;
      addCard(slide, sx, 1.35, 2.9, 1.1);
      slide.addText(s.value, {
        x: sx, y: 1.4, w: 2.9, h: 0.55,
        fontSize: 28, color: s.color, fontFace: "Arial Black", bold: true, align: "center", margin: 0
      });
      slide.addText(s.label, {
        x: sx, y: 1.95, w: 2.9, h: 0.3,
        fontSize: 12, color: C.textMuted, fontFace: "Calibri", align: "center", margin: 0
      });
    });

    // Effort comparison
    addCard(slide, 0.6, 2.8, 4.2, 1.5);
    slide.addText("Human Effort Reduction", {
      x: 0.8, y: 2.9, w: 3.8, h: 0.3,
      fontSize: 13, color: C.white, fontFace: "Calibri", bold: true, margin: 0
    });
    // Before bar
    slide.addShape("rect", { x: 0.8, y: 3.3, w: 3.8, h: 0.35, fill: { color: C.danger } });
    slide.addText("Traditional: 200+ hours", {
      x: 0.8, y: 3.3, w: 3.8, h: 0.35,
      fontSize: 11, color: C.white, fontFace: "Calibri", bold: true, align: "center", valign: "middle"
    });
    // After bar
    slide.addShape("rect", { x: 0.8, y: 3.8, w: 0.4, h: 0.35, fill: { color: C.success } });
    slide.addText("Qgentic AI: ~20 hrs", {
      x: 1.3, y: 3.8, w: 2.0, h: 0.35,
      fontSize: 11, color: C.success, fontFace: "Calibri", bold: true, valign: "middle"
    });

    slide.addText("90% REDUCTION", {
      x: 3.2, y: 3.8, w: 1.5, h: 0.35,
      fontSize: 13, color: C.success, fontFace: "Arial Black", bold: true, valign: "middle"
    });

    // Leaderboard screenshot
    addCard(slide, 5.1, 2.8, 4.5, 2.4);
    slide.addText("Kaggle Leaderboard", {
      x: 5.3, y: 2.9, w: 4.1, h: 0.3,
      fontSize: 12, color: C.accent, fontFace: "Calibri", bold: true, margin: 0
    });
    slide.addImage({
      path: LEADERBOARD_IMG,
      x: 5.3, y: 3.25, w: 4.1, h: 1.8,
      sizing: { type: "contain", w: 4.1, h: 1.8 }
    });
  }

  // ========== SLIDE 16: Lessons Learned ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Results");
    addTitle(slide, "Lessons Learned");
    addSlideNumber(slide, 16, TOTAL);

    const lessons = [
      {
        icon: FaEye, title: "LLM Log Monitoring Works",
        desc: "Catches overfitting, NaN losses, and stalled training that humans miss during off-hours. Surprisingly effective as a real-time diagnostic tool.",
        color: C.accent
      },
      {
        icon: FaNetworkWired, title: "Cross-Model Learning Prevents Redundancy",
        desc: "Without shared state, agents try the same bad idea across all models. The suggestion pool eliminates the 'try the same thing 8 times' failure mode.",
        color: C.success
      },
      {
        icon: FaLayerGroup, title: "Structured Outputs Are Essential",
        desc: "Pydantic schemas for LLM responses make multi-agent communication reliable. Without them, parsing failures cascade through the pipeline.",
        color: C.purple
      },
      {
        icon: FaSyncAlt, title: "Robust Retry Logic Is Critical",
        desc: "LLM providers hit 503 during peak demand. Exponential backoff (1s, 5s, 10s, 30s, 60s) + 5-min polling on sustained 503 keeps the system running.",
        color: C.warning
      },
    ];

    for (let i = 0; i < lessons.length; i++) {
      const l = lessons[i];
      const ly = 1.3 + i * 1.05;
      addCard(slide, 0.6, ly, 8.8, 0.9);
      await addIconCircle(slide, l.icon, 0.8, ly + 0.15, l.color);
      slide.addText(l.title, {
        x: 1.5, y: ly + 0.05, w: 3.5, h: 0.3,
        fontSize: 14, color: C.white, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(l.desc, {
        x: 1.5, y: ly + 0.38, w: 7.7, h: 0.45,
        fontSize: 11, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    }
  }

  // ========== SLIDE 17: Improvement Plan ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    addSectionLabel(slide, "Improvement Plan");
    addTitle(slide, "Roadmap & Future Work");
    addSlideNumber(slide, 17, TOTAL);

    const items = [
      { phase: "Phase 1", title: "Observability", desc: "Migrate from Weave to Langfuse for better tracing, cost tracking, and debugging capabilities", color: C.accent },
      { phase: "Phase 2", title: "Enhanced Guardrails", desc: "Pre-execution lint integration, improved leakage detection, richer stack trace remediation", color: C.success },
      { phase: "Phase 3", title: "Cross-Worker Learning V2", desc: "Richer shared state: embeddings, feature importances, training curves -- not just text suggestions", color: C.purple },
      { phase: "Phase 4", title: "Legal Retrieval", desc: "Competition rules compliance checking via legal retrieval integration (already scaffolded)", color: C.warning },
    ];

    // Timeline
    slide.addShape("rect", { x: 1.5, y: 1.3, w: 0.04, h: 2.8, fill: { color: C.textDim } });

    items.forEach((item, i) => {
      const iy = 1.3 + i * 0.72;
      // Dot on timeline
      slide.addShape("ellipse", { x: 1.35, y: iy + 0.08, w: 0.3, h: 0.3, fill: { color: item.color } });
      // Content
      slide.addText(item.phase, {
        x: 2.0, y: iy, w: 1.0, h: 0.25,
        fontSize: 10, color: item.color, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(item.title, {
        x: 3.0, y: iy, w: 3.0, h: 0.25,
        fontSize: 13, color: C.white, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(item.desc, {
        x: 3.0, y: iy + 0.25, w: 6.5, h: 0.35,
        fontSize: 10, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });

    // Goal callout
    addCard(slide, 2.0, 4.4, 7.6, 0.8);
    slide.addText("\u2605", {
      x: 2.2, y: 4.45, w: 0.5, h: 0.5,
      fontSize: 24, color: C.warning, fontFace: "Calibri", align: "center"
    });
    slide.addText("Ultimate Goal: Kaggle Competitions Grandmaster", {
      x: 2.8, y: 4.45, w: 5.0, h: 0.3,
      fontSize: 15, color: C.white, fontFace: "Calibri", bold: true, margin: 0
    });
    slide.addText("2 more gold medals needed \u2014 let the agents compete while you strategize", {
      x: 2.8, y: 4.75, w: 6.5, h: 0.25,
      fontSize: 11, color: C.textMuted, fontFace: "Calibri", italic: true, margin: 0
    });
  }

  // ========== SLIDE 18: Conclusion ==========
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bgDark };
    // Accent bar at top
    slide.addShape("rect", { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

    slide.addText("Key Takeaways", {
      x: 0.6, y: 0.5, w: 8.8, h: 0.5,
      fontSize: 32, color: C.white, fontFace: "Arial Black", bold: true, margin: 0
    });

    const takeaways = [
      { text: "Multi-agent decomposition beats monolithic agents", desc: "Specialized agents with distinct tools > one agent trying to do everything", color: C.accent },
      { text: "Cross-model learning creates emergent optimization", desc: "Shared suggestion pools enable collective intelligence across parallel baselines", color: C.success },
      { text: "90% effort reduction is achievable today", desc: "Current LLM capabilities are sufficient for structured ML execution loops", color: C.warning },
      { text: "The future of ML competitions is human + AI collaboration", desc: "Humans provide strategic direction; agents handle the disciplined execution", color: C.purple },
    ];

    takeaways.forEach((t, i) => {
      const ty = 1.3 + i * 0.85;
      slide.addShape("rect", { x: 0.6, y: ty, w: 0.06, h: 0.7, fill: { color: t.color } });
      slide.addText(t.text, {
        x: 0.9, y: ty, w: 8.7, h: 0.35,
        fontSize: 15, color: C.white, fontFace: "Calibri", bold: true, margin: 0
      });
      slide.addText(t.desc, {
        x: 0.9, y: ty + 0.35, w: 8.7, h: 0.3,
        fontSize: 12, color: C.textMuted, fontFace: "Calibri", margin: 0
      });
    });

    // Bottom bar
    slide.addShape("rect", { x: 0, y: 4.65, w: 10, h: 0.975, fill: { color: C.bgMedium } });
    slide.addText("From 200+ hours to 20. Let the agents do the grind.", {
      x: 0.6, y: 4.75, w: 8.8, h: 0.4,
      fontSize: 16, color: C.accent, fontFace: "Calibri", bold: true, italic: true, margin: 0
    });
    slide.addText("Open Source: github.com/bogoconic1/Qgentic-AI", {
      x: 0.6, y: 5.15, w: 8.8, h: 0.3,
      fontSize: 12, color: C.textMuted, fontFace: "Calibri", margin: 0
    });
  }

  // ========== WRITE FILE ==========
  const outputPath = path.resolve("/Users/oh_jason_zhang/Downloads/git_repo/Qgentic-AI/.claude/worktrees/strange-panini/Qgentic_AI_Keynote.pptx");
  await pres.writeFile({ fileName: outputPath });
  console.log("Presentation written to:", outputPath);
}

main().catch(err => { console.error(err); process.exit(1); });
