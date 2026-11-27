// Turn the clean sketches of src/ into hand-drawn SVGs for the documentation (doc/docs/img/science/).
//
//   cd doc/sketches && npm install roughjs@4 jsdom@26 && node roughen.js
//
// Lines, rectangles, circles, ellipses and paths are redrawn with rough.js (https://roughjs.com, MIT), arrowheads
// become two hand-drawn strokes, and the text uses Patrick Hand (Patrick Wagesreiter, SIL Open Font License 1.1),
// embedded as a subset in each file (fonts/patrick-hand-subset.woff2). Greek letters, arrows and math symbols, which
// the font lacks, are set in a math font. The output is deterministic: each element has a fixed seed.
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const rough = require('roughjs');

const SRC = path.join(__dirname, 'src');
const OUT = path.join(__dirname, '..', 'docs', 'img', 'science');
const FONT = fs.readFileSync(path.join(__dirname, 'fonts', 'patrick-hand-subset.woff2')).toString('base64');
const HAND = "'Patrick Hand', 'Comic Sans MS', cursive";
const MATH = "'Cambria Math', 'STIX Two Math', 'STIX Two Text', Georgia, 'DejaVu Serif', serif";
const SYMBOLS = /[Ͱ-Ͽ←-⇿∀-⋿]+/g;   // Greek, arrows, mathematical operators
const SHAPES = ['line', 'rect', 'circle', 'ellipse', 'path', 'polyline', 'polygon'];
const SVGNS = 'http://www.w3.org/2000/svg';
const SCALE = 1.35;   // Patrick Hand is narrower than the sans-serif of the sources: larger text, same width

const gen = rough.generator();
let seed = 1;

function num(el, name, dflt = 0) {
  const v = el.getAttribute(name);
  return v === null ? dflt : parseFloat(v);
}

function style(el) {
  // presentation attributes of the element, inherited from its <g> parents when absent
  const get = (name) => {
    for (let e = el; e && e.getAttribute; e = e.parentNode) {
      if (e.getAttribute(name) !== null) return e.getAttribute(name);
    }
    return null;
  };
  return {
    stroke: get('stroke'),
    strokeWidth: parseFloat(get('stroke-width') || '1'),
    fill: get('fill'),
    dash: get('stroke-dasharray'),
    opacity: get('opacity'),
    markerEnd: get('marker-end'),
    markerStart: get('marker-start'),
  };
}

function markerColor(doc, ref) {
  const id = ref.replace(/^url\(#/, '').replace(/\)$/, '');
  const m = doc.getElementById(id);
  const p = m && m.querySelector('path');
  return (p && p.getAttribute('fill')) || '#333';
}

function arrowhead(doc, x0, y0, x1, y1, color, width) {
  // two strokes from the tip, 25 degrees on each side of the line
  const a = Math.atan2(y1 - y0, x1 - x0);
  const len = 9 + 1.5 * width;
  const g = doc.createElementNS(SVGNS, 'g');
  for (const s of [-1, 1]) {
    const b = a + Math.PI + s * 0.45;
    const d = gen.line(x1, y1, x1 + len * Math.cos(b), y1 + len * Math.sin(b),
      { seed: seed++, roughness: 0.6, bowing: 0.5, stroke: color, strokeWidth: width });
    for (const p of gen.toPaths(d)) g.appendChild(pathEl(doc, p, color, width));
  }
  return g;
}

function pathEl(doc, p, stroke, width) {
  const e = doc.createElementNS(SVGNS, 'path');
  e.setAttribute('d', p.d);
  e.setAttribute('stroke', p.stroke === 'none' ? 'none' : (p.stroke || stroke));
  e.setAttribute('stroke-width', p.strokeWidth || width);
  e.setAttribute('fill', p.fill || 'none');
  e.setAttribute('stroke-linecap', 'round');
  e.setAttribute('stroke-linejoin', 'round');
  return e;
}

function roughen(doc, el, viewWidth) {
  const st = style(el);
  const tag = el.tagName.toLowerCase();
  if (tag === 'rect' && num(el, 'width') >= viewWidth && num(el, 'x') === 0) return;   // background
  const fill = st.fill && st.fill !== 'none' ? st.fill : undefined;
  const stroke = st.stroke && st.stroke !== 'none' ? st.stroke : 'none';
  const opts = {
    seed: seed++, roughness: 0.9, bowing: 0.8, stroke, strokeWidth: st.strokeWidth * 1.1,
    fill, fillStyle: 'solid', disableMultiStroke: !!st.dash || st.strokeWidth < 1.2,
    disableMultiStrokeFill: true, preserveVertices: false,
  };
  let d;
  let ends = null;
  if (tag === 'line') {
    const [x1, y1, x2, y2] = ['x1', 'y1', 'x2', 'y2'].map((a) => num(el, a));
    d = gen.line(x1, y1, x2, y2, opts);
    ends = [x1, y1, x2, y2];
  } else if (tag === 'rect') {
    d = gen.rectangle(num(el, 'x'), num(el, 'y'), num(el, 'width'), num(el, 'height'), opts);
  } else if (tag === 'circle') {
    const r = num(el, 'r');
    d = gen.circle(num(el, 'cx'), num(el, 'cy'), 2 * r, { ...opts, roughness: r < 8 ? 0.5 : 0.9 });
  } else if (tag === 'ellipse') {
    d = gen.ellipse(num(el, 'cx'), num(el, 'cy'), 2 * num(el, 'rx'), 2 * num(el, 'ry'), opts);
  } else if (tag === 'path') {
    d = gen.path(el.getAttribute('d'), opts);
  } else {
    d = gen.polygon(el.getAttribute('points').trim().split(/[\s,]+/).map(Number)
      .reduce((acc, v, i, all) => (i % 2 ? acc : acc.concat([[v, all[i + 1]]])), []), opts);
  }
  const g = doc.createElementNS(SVGNS, 'g');
  if (st.dash) g.setAttribute('stroke-dasharray', st.dash);
  if (st.opacity) g.setAttribute('opacity', st.opacity);
  for (const p of gen.toPaths(d)) g.appendChild(pathEl(doc, p, stroke, opts.strokeWidth));
  if (ends && st.markerEnd) {
    g.appendChild(arrowhead(doc, ends[0], ends[1], ends[2], ends[3], markerColor(doc, st.markerEnd), st.strokeWidth));
  }
  if (ends && st.markerStart) {
    g.appendChild(arrowhead(doc, ends[2], ends[3], ends[0], ends[1], markerColor(doc, st.markerStart), st.strokeWidth));
  }
  el.parentNode.replaceChild(g, el);
}

function mathRuns(doc, node) {
  // set the symbols the handwriting font lacks in a math font
  for (const child of Array.from(node.childNodes)) {
    if (child.nodeType === 1) { mathRuns(doc, child); continue; }
    if (child.nodeType !== 3 || !SYMBOLS.test(child.nodeValue)) continue;
    SYMBOLS.lastIndex = 0;
    const text = child.nodeValue;
    const frag = doc.createDocumentFragment();
    let last = 0;
    for (const m of text.matchAll(SYMBOLS)) {
      if (m.index > last) frag.appendChild(doc.createTextNode(text.slice(last, m.index)));
      const t = doc.createElementNS(SVGNS, 'tspan');
      t.setAttribute('font-family', MATH);
      t.setAttribute('font-size', '0.92em');
      t.textContent = m[0];
      frag.appendChild(t);
      last = m.index + m[0].length;
    }
    if (last < text.length) frag.appendChild(doc.createTextNode(text.slice(last)));
    node.replaceChild(frag, child);
  }
}

fs.mkdirSync(OUT, { recursive: true });
for (const name of fs.readdirSync(SRC).filter((f) => f.endsWith('.svg')).sort()) {
  seed = 1;
  const dom = new JSDOM(fs.readFileSync(path.join(SRC, name), 'utf8'), { contentType: 'image/svg+xml' });
  const doc = dom.window.document;
  const svg = doc.documentElement;
  const viewWidth = parseFloat(svg.getAttribute('viewBox').split(/\s+/)[2]);
  const shapes = SHAPES.flatMap((t) => Array.from(svg.getElementsByTagName(t)))
    .filter((e) => !e.closest('defs'));
  shapes.sort((a, b) => (a.compareDocumentPosition(b) & 4 ? -1 : 1));
  for (const el of shapes) roughen(doc, el, viewWidth);
  for (const d of Array.from(svg.getElementsByTagName('defs'))) d.parentNode.removeChild(d);
  for (const t of Array.from(svg.getElementsByTagName('text'))) {
    mathRuns(doc, t);
    const size = parseFloat(t.getAttribute('font-size') || svg.getAttribute('font-size') || '13');
    t.setAttribute('font-size', (size * SCALE).toFixed(1));
    for (const s of Array.from(t.getElementsByTagName('tspan'))) {
      // sub- and superscripts: a smaller shift than the browsers' default, which collides with the next line
      const shift = s.getAttribute('baseline-shift');
      if (shift === 'sub') s.setAttribute('baseline-shift', '-22%');
      if (shift === 'super') s.setAttribute('baseline-shift', '38%');
      if (s.getAttribute('font-size') && !s.getAttribute('font-size').endsWith('em')) {
        s.setAttribute('font-size', (parseFloat(s.getAttribute('font-size')) * SCALE).toFixed(1));
      }
    }
  }
  svg.setAttribute('font-family', HAND);
  const defs = doc.createElementNS(SVGNS, 'defs');
  const css = doc.createElementNS(SVGNS, 'style');
  css.textContent = `@font-face { font-family: 'Patrick Hand'; src: url(data:font/woff2;base64,${FONT}) format('woff2'); }`;
  defs.appendChild(css);
  svg.insertBefore(defs, svg.firstChild);
  fs.writeFileSync(path.join(OUT, name), svg.outerHTML + '\n');
  console.log(name);
}
