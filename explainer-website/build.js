// Reads content.md, renders it into index.html.
// Run: node build.js
"use strict";
const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "content.md");
const OUT = path.join(__dirname, "index.html");

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const MIME_TYPES = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif", webp: "image/webp", svg: "image/svg+xml" };

// Local image paths (anything not already a URL) are read from the project
// folder and embedded as a data URI, so the page never depends on external
// file hosting -- it works identically opened locally or published.
function resolveImageSrc(src) {
  if (/^(https?:)?\/\//i.test(src) || src.startsWith("data:")) return src;
  const ext = path.extname(src).slice(1).toLowerCase();
  const mime = MIME_TYPES[ext];
  const filePath = path.join(__dirname, src);
  if (!mime || !fs.existsSync(filePath)) {
    console.warn("Warning: could not find local image \"" + src + "\" in " + __dirname);
    return src;
  }
  const data = fs.readFileSync(filePath).toString("base64");
  return "data:" + mime + ";base64," + data;
}

function slugify(s) {
  return String(s)
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-");
}

// Wraps the first, non-overlapping occurrence of each glossary term in one
// single pass over the plain text, so a term's own definition (which may
// itself contain another term's word) never gets a tooltip nested inside it.
// `used` is shared across the whole build (see build()) so a term only gets
// its tooltip the first time it appears anywhere on the page, not once per
// paragraph.
function linkify(text, glossary, used) {
  const terms = Object.keys(glossary);
  const escaped = esc(text);
  if (!terms.length) return escaped;
  const pattern = terms
    .slice()
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  const re = new RegExp("\\b(" + pattern + ")\\b", "gi");
  return escaped.replace(re, (m) => {
    const key = terms.find((t) => t.toLowerCase() === m.toLowerCase());
    if (!key || used.has(key)) return m;
    used.add(key);
    return '<span class="term" tabindex="0">' + m + '<span class="tooltip" role="tooltip">' + esc(glossary[key]) + "</span></span>";
  });
}

function parseFrontmatter(text) {
  const fm = {};
  text.split("\n").forEach((line) => {
    const m = line.match(/^([A-Za-z]+):\s*(.*)$/);
    if (m) fm[m[1].toLowerCase()] = m[2].trim();
  });
  fm.links = (fm.links || "")
    .split("|")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const idx = part.indexOf("=");
      return { label: part.slice(0, idx).trim(), href: part.slice(idx + 1).trim() };
    });
  return fm;
}

function parsePlainBlocks(body) {
  const chunks = body
    .split(/\n\s*\n/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => {
      const lines = chunk.split("\n").map((l) => l.trim());
      const imgMatch = lines[0].match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
      if (imgMatch) {
        const capMatch = lines[1] && lines[1].match(/^\*(.+)\*$/);
        return { type: "figure", alt: imgMatch[1], src: imgMatch[2], caption: capMatch ? capMatch[1] : imgMatch[1] };
      }
      if (lines.every((l) => l.startsWith("- "))) {
        return { type: "claims", items: lines.map((l) => l.replace(/^- /, "")) };
      }
      if (lines[0].startsWith("> ")) {
        return { type: "quote", text: lines.map((l) => l.replace(/^>\s?/, "")).join(" ") };
      }
      return { type: "p", text: lines.join(" ") };
    });

  // A blank line between two list items is still one list in standard
  // markdown (a "loose list"), not two -- merge consecutive claims blocks
  // so bullets separated by blank lines still number as one list.
  const merged = [];
  chunks.forEach((block) => {
    const prev = merged[merged.length - 1];
    if (block.type === "claims" && prev && prev.type === "claims") prev.items.push(...block.items);
    else merged.push(block);
  });
  return merged;
}

// ::: Label text
// ...body (any normal blocks: paragraphs, quotes, lists, figures)...
// :::
// becomes a collapsed <details> block. Everything outside ::: fences is
// parsed as usual; text inside a fence is parsed recursively so it can
// hold its own paragraphs, quotes, lists, or figures. A fresh RegExp is
// created per call (never shared) since this function recurses and a
// shared global regex's lastIndex would corrupt the outer scan.
function parseBlocks(body) {
  const dropdownRe = /^::: ?(.*)\n([\s\S]*?)\n:::[ \t]*$/gm;
  const blocks = [];
  let lastIndex = 0;
  let match;
  while ((match = dropdownRe.exec(body))) {
    if (match.index > lastIndex) blocks.push(...parsePlainBlocks(body.slice(lastIndex, match.index)));
    blocks.push({ type: "dropdown", label: match[1].trim(), blocks: parseBlocks(match[2]) });
    lastIndex = dropdownRe.lastIndex;
  }
  if (lastIndex < body.length) blocks.push(...parsePlainBlocks(body.slice(lastIndex)));
  return blocks;
}

// Headings: ## is a top-level section, ### and #### nest inside the nearest
// shallower heading (1, 1.1, 1.1.1 style numbering, tracked via a stack).
function parseDocument(raw) {
  const firstHeading = raw.search(/\n#{2,4} /);
  const frontmatterText = firstHeading === -1 ? raw : raw.slice(0, firstHeading);
  const bodyText = firstHeading === -1 ? "" : raw.slice(firstHeading + 1);
  const fm = parseFrontmatter(frontmatterText);

  const chunks = bodyText.split(/\n(?=#{2,4} )/).filter((c) => c.trim());
  const glossary = {};
  const top = [];
  let stack = [];
  let topNum = 1;
  let takeaways = null;
  let relatedWork = null;

  chunks.forEach((chunk) => {
    const lines = chunk.split("\n");
    const headingMatch = lines[0].match(/^(#{2,4})\s*(.*)$/);
    const level = headingMatch[1].length;
    const heading = headingMatch[2].trim();
    const rest = lines.slice(1).join("\n");
    const headingLower = heading.toLowerCase();

    if (level === 2 && headingLower === "glossary") {
      rest
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.startsWith("- "))
        .forEach((l) => {
          const body2 = l.replace(/^- /, "");
          const idx = body2.indexOf(":");
          if (idx !== -1) glossary[body2.slice(0, idx).trim()] = body2.slice(idx + 1).trim();
        });
      stack = [];
      return;
    }

    // These two sections are pulled out of the normal numbered flow: "Main
    // takeaways" renders in the masthead ribbon alongside the title/authors,
    // "Related work" renders in its own tab instead of the main scroll.
    if (level === 2 && headingLower === "main takeaways") {
      takeaways = { heading, blocks: parseBlocks(rest) };
      stack = [];
      return;
    }
    if (level === 2 && headingLower === "related work") {
      relatedWork = { heading, blocks: parseBlocks(rest) };
      stack = [];
      return;
    }

    const node = { level, heading, blocks: parseBlocks(rest), children: [] };

    if (level === 2) {
      node.num = String(topNum++);
      node.id = slugify(heading);
      top.push(node);
      stack = [node];
    } else {
      while (stack.length && stack[stack.length - 1].level >= level) stack.pop();
      const parent = stack[stack.length - 1] || top[top.length - 1];
      node.num = parent.num + "." + (parent.children.length + 1);
      node.id = parent.id + "-" + slugify(heading);
      parent.children.push(node);
      stack.push(node);
    }
  });

  return { fm, sections: top, glossary, takeaways, relatedWork };
}

function renderBlock(block, glossary, used) {
  if (block.type === "p") return "<p>" + linkify(block.text, glossary, used) + "</p>";
  if (block.type === "quote") return "<blockquote>" + linkify(block.text, glossary, used) + "</blockquote>";
  if (block.type === "figure") {
    return (
      "<figure>" +
      '<img src="' + esc(resolveImageSrc(block.src)) + '" alt="' + esc(block.alt) + '" loading="lazy">' +
      (block.caption ? "<figcaption>" + esc(block.caption) + "</figcaption>" : "") +
      "</figure>"
    );
  }
  if (block.type === "claims") {
    return (
      '<ol class="claims">' +
      block.items.map((item) => "<li><span>" + linkify(item, glossary, used) + "</span></li>").join("") +
      "</ol>"
    );
  }
  if (block.type === "dropdown") {
    return (
      '<details class="dropdown"><summary>' + esc(block.label) + "</summary>" +
      '<div class="dropdown-body">' + block.blocks.map((b) => renderBlock(b, glossary, used)).join("\n") + "</div>" +
      "</details>"
    );
  }
  return "";
}

function renderNode(node, glossary, used) {
  const tag = node.level === 2 ? "h2" : node.level === 3 ? "h3" : "h4";
  const numHtml = '<span class="num">' + esc(node.num) + "</span>";
  const heading = "<" + tag + ">" + numHtml + esc(node.heading) + "</" + tag + ">";
  const blocksHtml = node.blocks.map((b) => renderBlock(b, glossary, used)).join("\n");
  const childrenHtml = node.children.map((c) => renderNode(c, glossary, used)).join("\n");
  const wrapTag = node.level === 2 ? "section" : "div";
  const cls = node.level === 2 ? "reveal" : "subsection";
  return "<" + wrapTag + ' id="' + node.id + '" class="' + cls + '">' + heading + blocksHtml + childrenHtml + "</" + wrapTag + ">";
}

function renderNavNode(node) {
  let html = '<li><a class="toc-link" href="#' + node.id + '">' + esc(node.heading) + "</a>";
  if (node.children.length) html += "<ol>" + node.children.map(renderNavNode).join("") + "</ol>";
  html += "</li>";
  return html;
}

function renderNav(sections) {
  return "<ol>" + sections.map(renderNavNode).join("") + "</ol>";
}

function renderHeader(fm, takeaways, glossary, used) {
  const links = (fm.links || []).map((l) => '<a href="' + esc(l.href) + '">' + esc(l.label) + "</a>").join("");
  const takeawaysHtml = takeaways
    ? '<div class="takeaways"><span class="takeaways-label">' + esc(takeaways.heading) + "</span>" +
      takeaways.blocks.map((b) => renderBlock(b, glossary, used)).join("\n") +
      "</div>"
    : "";
  return (
    '<div class="masthead">' +
    '<span class="eyebrow"><span class="dot"></span>' + esc(fm.eyebrow || "") + "</span>" +
    '<h1 class="title">' + esc(fm.title || "") + "</h1>" +
    (fm.subtitle ? '<p class="subtitle">' + esc(fm.subtitle) + "</p>" : "") +
    '<p class="authors">' + esc(fm.authors || "") + "</p>" +
    '<p class="affiliations">' + esc(fm.affiliations || "") + "</p>" +
    '<div class="links-row">' + links + "</div>" +
    takeawaysHtml +
    "</div>"
  );
}

function renderTabs(relatedWork) {
  if (!relatedWork) return "";
  return (
    '<div class="tabs" role="tablist">' +
    '<button class="tab-btn active" id="tabPaper" type="button" role="tab" aria-selected="true" aria-controls="panelPaper">Paper</button>' +
    '<button class="tab-btn" id="tabRelated" type="button" role="tab" aria-selected="false" aria-controls="panelRelated">' + esc(relatedWork.heading) + "</button>" +
    "</div>"
  );
}

function renderRelatedPanel(relatedWork, glossary, used) {
  if (!relatedWork) return "";
  return (
    '<div id="panelRelated" role="tabpanel" aria-labelledby="tabRelated" hidden>' +
    "<h2>" + esc(relatedWork.heading) + "</h2>" +
    relatedWork.blocks.map((b) => renderBlock(b, glossary, used)).join("\n") +
    "</div>"
  );
}

function renderFooter(fm) {
  return (
    "<footer><div>Contact us at " +
    '<a href="mailto:' + esc(fm.contact || "") + '">' + esc(fm.contact || "") + "</a></div>" +
    "<div>" + esc(fm.footer || "") + "</div></footer>"
  );
}

// A purely decorative image, its own third column alongside the sidebar and
// content -- only shown once the viewport is wide enough that empty space
// genuinely exists there beyond those two (see the min-width: 100rem rule).
function renderMarginFigure(fm) {
  if (!fm.marginimage) return "";
  return (
    '<aside class="imagecol"><img src="' + esc(resolveImageSrc(fm.marginimage)) + '" alt="Decorative illustration" loading="lazy"></aside>'
  );
}

const HEAD = `<title>__TITLE__</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;500;600&family=Cabin:wght@400;500;600;700&family=Merriweather:ital,wght@1,300&display=swap');

  :root {
    --bg: #FAFAF9;
    --bg-raised: #f1f0ea;
    --bg-overlay: rgba(250, 250, 249, 0.92);
    --ink: #1c1a12;
    --ink-secondary: #6f6b5c;
    --ink-tertiary: #a39d8a;
    --border: #e3e1d8;
    --border-strong: #d0cdbf;
    --accent: #5D7B6A;
    --accent-hover: #4c6759;
    --accent-soft: #5D7B6A1a;
    --accent-2: #97C3FC;
    --accent-2-soft: #97C3FC1f;
    --box-bg: #24332D;
    --box-ink: #FAFAF9;
    --max-width: 1180px;
    --content-width: 42rem;
    --font-heading: "Source Serif 4", "Palatino Linotype", Palatino, Georgia, serif;
    --font-body: "Cabin", "Trebuchet MS", sans-serif;
    --font-mono: "Cabin", "Trebuchet MS", sans-serif;
    --font-accent: "Merriweather", Georgia, serif;
  }

  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #15140f;
      --bg-raised: #1e1c15;
      --bg-overlay: rgba(21, 20, 15, 0.92);
      --ink: #ece7d8;
      --ink-secondary: #a49d89;
      --ink-tertiary: #6d6858;
      --border: #322e22;
      --border-strong: #423d2c;
      --accent: #8AA79A;
      --accent-hover: #a3bdb1;
      --accent-soft: #8AA79A26;
      --accent-2: #97C3FC;
      --accent-2-soft: #97C3FC26;
    }
  }
  :root[data-theme="dark"] {
    --bg: #15140f;
    --bg-raised: #1e1c15;
    --bg-overlay: rgba(21, 20, 15, 0.92);
    --ink: #ece7d8;
    --ink-secondary: #a49d89;
    --ink-tertiary: #6d6858;
    --border: #322e22;
    --border-strong: #423d2c;
    --accent: #86b39c;
    --accent-hover: #a3c7b3;
    --accent-soft: #86b39c26;
    --accent-2: #7ec2e0;
    --accent-2-soft: #7ec2e026;
  }

  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }

  body {
    background: var(--bg);
    color: var(--ink);
    font-family: var(--font-body);
    font-size: 1.1rem;
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }

  a { color: var(--accent); text-decoration-color: var(--accent-soft); text-underline-offset: 3px; }
  a:hover { color: var(--accent-hover); }
  a:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }

  .eyebrow, nav, .links-row, .mono { font-family: var(--font-mono); letter-spacing: 0.04em; font-weight: 500; }

  .shell { max-width: var(--max-width); margin: 0 auto; display: flex; align-items: flex-start; }

  .topbar {
    display: none; position: sticky; top: 0; z-index: 30; align-items: center; justify-content: space-between;
    padding: 0.85rem 1.25rem; background: var(--bg-overlay); backdrop-filter: blur(6px); border-bottom: 1px solid var(--border);
  }
  .topbar .brand { font-family: var(--font-mono); font-size: 0.8rem; letter-spacing: 0.06em; color: var(--ink-secondary); }
  .menu-btn {
    background: none; border: 1px solid var(--border-strong); border-radius: 6px; color: var(--ink);
    width: 2.25rem; height: 2.25rem; display: flex; align-items: center; justify-content: center; cursor: pointer;
  }
  .menu-btn svg { width: 1.05rem; height: 1.05rem; }

  aside.sidebar {
    /* max-height, not height: a fixed 100vh sticky box gets shoved above the
       viewport once less than a full screen of page remains below it -- the
       classic short-last-section bug. max-height lets it size to its actual
       (much shorter) content instead, so it never needs more room than it uses. */
    position: sticky; top: 0; align-self: flex-start; max-height: 100vh; width: 16.5rem; flex: none;
    overflow-y: auto; padding: 3rem 1.5rem 3rem 0; border-right: 1px solid var(--border);
  }
  .sidebar-inner { padding-left: 0.25rem; }
  .sidebar .site-label {
    display: block; font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-tertiary); margin: 0 0 1.25rem;
  }
  nav ol { list-style: none; margin: 0; padding: 0; }
  nav a.toc-link {
    display: block; padding: 0.4rem 0.5rem; margin: 0 -0.5rem; border-radius: 5px;
    font-size: 0.82rem; color: var(--ink-secondary); text-decoration: none; letter-spacing: 0.02em;
    transition: color 0.15s ease, background 0.15s ease;
  }
  nav a.toc-link:hover { color: var(--ink); background: var(--bg-raised); }
  nav a.toc-link.active { color: var(--accent); background: var(--accent-soft); font-weight: 600; }
  nav ol ol { padding-left: 1rem; margin-top: 0.1rem; }
  nav ol ol a.toc-link { font-size: 0.74rem; color: var(--ink-tertiary); }
  nav ol ol a.toc-link.active { color: var(--accent-2); background: var(--accent-2-soft); }

  main { flex: 1; min-width: 0; padding: 4.5rem clamp(1.5rem, 5vw, 4rem) 8rem; }
  .content-col { max-width: var(--content-width); margin: 0 auto; }

  @keyframes rise-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .masthead { animation: rise-in 0.5s ease both; }

  .eyebrow {
    font-size: 0.72rem; text-transform: uppercase; color: var(--ink-secondary);
    display: inline-flex; align-items: center; gap: 0.5em; padding: 0.3em 0.7em;
    border: 1px solid var(--border-strong); border-radius: 999px; margin-bottom: 1.75rem;
  }
  .eyebrow .dot { width: 0.4em; height: 0.4em; border-radius: 50%; background: var(--accent); flex: none; }

  h1.title {
    font-family: var(--font-heading); font-weight: 600; font-size: clamp(2.1rem, 4.2vw, 3rem);
    line-height: 1.12; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 0.6rem;
  }
  p.subtitle {
    font-family: var(--font-accent); font-weight: 300; font-size: 1.25rem; font-style: italic;
    color: var(--ink-secondary); margin: 0 0 2rem; text-wrap: balance; max-width: 40ch;
  }
  p.authors { font-size: 1.02rem; margin: 0 0 0.35rem; }
  p.affiliations { font-family: var(--font-heading); font-size: 0.78rem; color: var(--ink-secondary); letter-spacing: 0.02em; line-height: 1.8; margin: 0 0 1.75rem; }
  .links-row {
    display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; font-size: 0.78rem; text-transform: uppercase;
    color: var(--ink-secondary); margin-bottom: 0.5rem;
  }
  .links-row a { text-decoration: none; border-bottom: 1px solid var(--accent-soft); padding-bottom: 2px; }
  .links-row a:hover { border-bottom-color: var(--accent); }

  .takeaways { margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid var(--border); }
  .takeaways-label {
    display: block; font-family: var(--font-mono); font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--ink-tertiary); margin-bottom: 0.75rem;
  }
  .takeaways p:last-child { margin-bottom: 0; }

  .tabs { display: flex; gap: 1.75rem; margin: 2.5rem 0 2.5rem; border-bottom: 1px solid var(--border); }
  .tab-btn {
    background: none; border: none; padding: 0 0 0.85rem; margin-bottom: -1px; cursor: pointer;
    font-family: var(--font-mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--ink-secondary); border-bottom: 2px solid transparent;
  }
  .tab-btn:hover { color: var(--ink); }
  .tab-btn.active { color: var(--ink); border-bottom-color: var(--accent); }

  section.reveal { margin-bottom: 3.5rem; scroll-margin-top: 2rem; transition: opacity 0.6s ease, transform 0.6s ease; }
  section.reveal.pre-reveal { opacity: 0; transform: translateY(14px); }
  section.reveal.in-view { opacity: 1; transform: none; }

  h2 { font-family: var(--font-heading); font-weight: 400; font-size: 1.55rem; letter-spacing: -0.005em; margin: 0 0 1.1rem; padding-top: 0.25rem; display: flex; align-items: center; }
  h3 { font-family: var(--font-heading); font-weight: 500; font-size: 1.2rem; margin: 2rem 0 0.9rem; display: flex; align-items: center; }
  h4 { font-family: var(--font-heading); font-weight: 500; font-size: 1.05rem; margin: 1.5rem 0 0.7rem; display: flex; align-items: center; }

  .num { font-family: var(--font-mono); font-size: 0.9em; color: var(--ink-secondary); margin-right: 0.5em; }

  .subsection { margin: 0 0 1.75rem; padding-left: 1.1rem; border-left: 2px solid var(--border); }

  p { margin: 0 0 1.15rem; }
  section > p:last-child, .subsection > p:last-child { margin-bottom: 0; }

  figure { margin: 2rem 0; }
  figure img { display: block; width: 100%; border-radius: 8px; border: 1px solid var(--border); }
  figcaption { margin-top: 0.6rem; font-family: var(--font-heading); font-size: 0.85rem; color: var(--ink-secondary); text-align: center; }

  aside.imagecol { display: none; }
  @media (min-width: 100rem) {
    .shell.has-image { max-width: 96rem; }
    aside.imagecol {
      display: flex; flex: none; width: 22rem; max-height: 100vh; position: sticky; top: 0;
      align-self: flex-start; align-items: flex-start; justify-content: center; padding-top: 4.5rem;
    }
    aside.imagecol img {
      width: 100%; max-width: 18rem; border-radius: 12px; box-shadow: 0 20px 48px rgba(0, 0, 0, 0.22);
    }
  }

  .claims { list-style: none; margin: 1.5rem 0 0; padding: 0; border-top: 1px solid var(--border); counter-reset: claim; }
  .claims li { counter-increment: claim; display: flex; align-items: baseline; gap: 0.75rem; padding: 1rem 0; border-bottom: 1px solid var(--border); }
  .claims li::before {
    content: counter(claim); font-family: var(--font-mono); font-size: 0.85rem; color: var(--ink-secondary); flex: none;
  }

  blockquote {
    margin: 1.75rem 0; padding: 0.25rem 0 0.25rem 1.25rem; border-left: 2px solid var(--accent-soft);
    color: var(--ink-secondary); font-family: var(--font-accent); font-weight: 300; font-style: italic; font-size: 1.05em;
  }

  details.dropdown { margin: 1.75rem 0; border-radius: 8px; background: var(--box-bg); color: var(--box-ink); overflow: hidden; }
  details.dropdown summary {
    cursor: pointer; list-style: none; display: flex; align-items: center; gap: 0.6rem;
    font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--box-ink); padding: 0.85rem 1.1rem; -webkit-tap-highlight-color: transparent;
  }
  details.dropdown summary::-webkit-details-marker { display: none; }
  details.dropdown summary::before { content: "+"; color: var(--box-ink); font-size: 1.1em; line-height: 1; display: inline-block; width: 1em; text-align: center; }
  details.dropdown[open] summary::before { content: "−"; }
  details.dropdown[open] summary { border-bottom: 1px solid rgba(250, 250, 249, 0.2); }
  details.dropdown summary:hover { opacity: 0.75; }
  details.dropdown summary:focus-visible { outline: 2px solid var(--box-ink); outline-offset: -2px; border-radius: 4px; }
  details.dropdown .dropdown-body { padding: 1.1rem 1.1rem 1.25rem; color: var(--box-ink); }
  details.dropdown .dropdown-body > *:last-child { margin-bottom: 0; }
  details.dropdown blockquote { color: var(--box-ink); border-left-color: rgba(250, 250, 249, 0.3); }

  .term { position: relative; cursor: help; border-bottom: 1px dotted var(--accent); color: inherit; }
  .term .tooltip {
    position: absolute; bottom: 135%; left: 50%; transform: translateX(-50%) translateY(4px);
    width: max-content; max-width: 17rem; background: var(--box-bg); color: var(--box-ink);
    font-family: var(--font-mono); font-size: 0.78rem; line-height: 1.5; letter-spacing: 0.01em;
    padding: 0.6em 0.8em; border-radius: 7px; text-align: left; font-style: normal;
    opacity: 0; visibility: hidden; pointer-events: none; transition: opacity 0.15s ease, transform 0.15s ease; z-index: 50;
  }
  .term .tooltip::after { content: ""; position: absolute; top: 100%; left: 50%; transform: translateX(-50%); border: 5px solid transparent; border-top-color: var(--box-bg); }
  .term:hover .tooltip, .term:focus-visible .tooltip, .term.show .tooltip { opacity: 1; visibility: visible; transform: translateX(-50%) translateY(0); }

  footer {
    max-width: var(--content-width); margin: 0 auto; padding: 2.5rem clamp(1.5rem, 5vw, 4rem) 6rem; border-top: 1px solid var(--border);
    font-family: var(--font-heading); font-size: 0.8rem; color: var(--ink-secondary); letter-spacing: 0.01em; line-height: 1.9;
  }
  footer a { color: var(--ink-secondary); text-decoration: underline; text-decoration-color: var(--border-strong); }
  footer a:hover { color: var(--accent); }

  @media (max-width: 63.9rem) {
    .topbar { display: flex; }
    aside.sidebar {
      position: fixed; inset: 3.4rem 0 0 0; height: auto; bottom: 0; width: auto;
      background: var(--bg-overlay); backdrop-filter: blur(8px); border-right: none; padding: 1.5rem;
      transform: translateY(-8px); opacity: 0; visibility: hidden;
      transition: opacity 0.15s ease, transform 0.15s ease, visibility 0.15s; z-index: 20;
    }
    aside.sidebar.open { opacity: 1; transform: translateY(0); visibility: visible; }
    nav a.toc-link { font-size: 0.95rem; padding: 0.55rem 0.6rem; }
    main { padding: 2.75rem 1.25rem 6rem; }
    h1.title { font-size: 1.85rem; }
    p.subtitle { font-size: 1.1rem; }
  }

  @media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    aside.sidebar, nav a.toc-link, section.reveal, .masthead, .term .tooltip { transition: none; animation: none; }
    section.reveal.pre-reveal { opacity: 1; transform: none; }
  }
</style>`;

const SCRIPT = `<script>
(function () {
  "use strict";

  function reducedMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function setupReveal() {
    if (reducedMotion()) return;
    var vh = window.innerHeight;
    var sections = document.querySelectorAll("main section.reveal");
    var toObserve = [];
    sections.forEach(function (s) {
      var rect = s.getBoundingClientRect();
      if (rect.top > vh * 0.6) { s.classList.add("pre-reveal"); toObserve.push(s); }
    });
    if (!toObserve.length || !("IntersectionObserver" in window)) return;
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.remove("pre-reveal");
          entry.target.classList.add("in-view");
          obs.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
    toObserve.forEach(function (s) { obs.observe(s); });
  }

  // Scroll-position based (not IntersectionObserver-band based): on every
  // scroll frame, pick the last heading whose top has crossed the reference
  // line. This stays correct scrolling up through short sections, where an
  // intersection-band approach could skip a section between observer ticks.
  function setupScrollspy() {
    var links = Array.prototype.slice.call(document.querySelectorAll("nav a.toc-link"));
    var targets = links.map(function (a) { return document.querySelector(a.getAttribute("href")); });
    if (!targets.length) return;
    var ticking = false;

    function update() {
      var offset = window.innerHeight * 0.25;
      var activeIdx = 0;
      for (var i = 0; i < targets.length; i++) {
        if (targets[i] && targets[i].getBoundingClientRect().top - offset <= 0) activeIdx = i;
        else if (targets[i]) break;
      }
      links.forEach(function (l) { l.classList.remove("active"); });
      if (links[activeIdx]) links[activeIdx].classList.add("active");
    }

    function onScroll() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function () { update(); ticking = false; });
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
  }

  function setupTooltipTouch() {
    document.querySelectorAll(".term").forEach(function (term) {
      term.addEventListener("click", function (e) {
        if (window.matchMedia && window.matchMedia("(hover: hover)").matches) return;
        e.stopPropagation();
        document.querySelectorAll(".term.show").forEach(function (t) { if (t !== term) t.classList.remove("show"); });
        term.classList.toggle("show");
      });
    });
    document.addEventListener("click", function () {
      document.querySelectorAll(".term.show").forEach(function (t) { t.classList.remove("show"); });
    });
  }

  function setupTabs() {
    var tabPaper = document.getElementById("tabPaper");
    var tabRelated = document.getElementById("tabRelated");
    var panelPaper = document.getElementById("panelPaper");
    var panelRelated = document.getElementById("panelRelated");
    if (!tabPaper || !tabRelated || !panelPaper || !panelRelated) return;

    function show(showPaper) {
      panelPaper.hidden = !showPaper;
      panelRelated.hidden = showPaper;
      tabPaper.classList.toggle("active", showPaper);
      tabRelated.classList.toggle("active", !showPaper);
      tabPaper.setAttribute("aria-selected", String(showPaper));
      tabRelated.setAttribute("aria-selected", String(!showPaper));
    }

    tabPaper.addEventListener("click", function () { show(true); });
    tabRelated.addEventListener("click", function () { show(false); });
  }

  function setupMenu() {
    var btn = document.getElementById("menuBtn");
    var sidebar = document.getElementById("sidebar");
    if (!btn || !sidebar) return;
    btn.addEventListener("click", function () {
      var open = sidebar.classList.toggle("open");
      btn.setAttribute("aria-expanded", String(open));
    });
    sidebar.addEventListener("click", function (e) {
      if (e.target.closest("a")) { sidebar.classList.remove("open"); btn.setAttribute("aria-expanded", "false"); }
    });
  }

  setupMenu();
  setupScrollspy();
  setupTooltipTouch();
  setupReveal();
  setupTabs();
})();
</script>`;

function build() {
  const raw = fs.readFileSync(SRC, "utf8");
  const { fm, sections, glossary, takeaways, relatedWork } = parseDocument(raw);
  // Shared across the whole page: a glossary term only gets its tooltip
  // treatment the first time it appears anywhere, not once per paragraph.
  const used = new Set();

  const body =
    '<div class="topbar"><span class="brand">DRAFT / PAPER SITE</span>' +
    '<button class="menu-btn" id="menuBtn" type="button" aria-expanded="false" aria-controls="sidebar" aria-label="Toggle table of contents">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button></div>' +
    '<div class="shell' + (fm.marginimage ? " has-image" : "") + '"><aside class="sidebar" id="sidebar"><div class="sidebar-inner"><span class="site-label">Contents</span>' +
    '<nav aria-label="Table of contents">' + renderNav(sections) + "</nav></div></aside>" +
    "<main><div class=\"content-col\">" + renderHeader(fm, takeaways, glossary, used) +
    renderTabs(relatedWork) +
    '<div id="panelPaper" role="tabpanel" aria-labelledby="tabPaper">' +
    sections.map((s) => renderNode(s, glossary, used)).join("\n") +
    "</div>" +
    renderRelatedPanel(relatedWork, glossary, used) +
    "</div></main>" +
    renderMarginFigure(fm) +
    "</div>" +
    renderFooter(fm);

  const head = HEAD.replace("__TITLE__", esc(fm.title || "Untitled Paper"));
  const html = head + "\n" + body + "\n" + SCRIPT + "\n";
  fs.writeFileSync(OUT, html);
  console.log("Built " + OUT + " from " + SRC + " (" + sections.length + " top-level sections, " + Object.keys(glossary).length + " glossary terms)");
}

build();
