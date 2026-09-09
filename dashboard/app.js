/* Parlamonitor dashboard.
 *
 * Forms follow the data's job, not habit:
 *  - "where does this MP sit" is a comparison against a population, so it is a
 *    percentile strip with the population visible (emphasis), NOT a radar. A
 *    radar encodes magnitude as radius, so area grows as the square and a 2x
 *    score reads as 4x; its axis order is arbitrary and changes the silhouette;
 *    and it cannot show 88 comparators at once.
 *  - an ordered sentiment scale is a diverging stacked bar centred on neutral.
 *  - topic x metric is a heatmap, on per-column z-scores so one diverging ramp
 *    serves every column and 0 always means "the ciklus average".
 *  - the network is a force layout.
 *
 * Colour follows the entity: a faction keeps its slot whatever is filtered.
 */
const T = {
  metrics: {
    readability_lix:      { hu: "Olvashatóság (LIX)",        hint: "magasabb = nehezebb" },
    diversity_mattr:      { hu: "Szókincs (MATTR)",          hint: "magasabb = változatosabb" },
    words_per_sentence:   { hu: "Szó / mondat",              hint: "" },
    sentiment_valence:    { hu: "Hangulat",                  hint: "−1 negatív … +1 pozitív" },
    emotion_anger:        { hu: "Düh",                       hint: "" },
    emotion_joy:          { hu: "Öröm",                      hint: "" },
    laughter_per_minute:  { hu: "Derültség / perc",          hint: "amit kiváltott" },
    applause_per_minute:  { hu: "Taps / perc",               hint: "amit kiváltott" },
    heckles_received:     { hu: "Kapott közbeszólás",        hint: "" }
  },
  sentiment: ["nagyon negatív", "negatív", "semleges", "pozitív", "nagyon pozitív"],
  emotions:  { anger: "düh", joy: "öröm", sadness: "szomorúság", fear: "félelem" },
  heat: {
    sentiment_valence: "Hangulat", emotion_anger: "Düh", emotion_joy: "Öröm",
    lix: "LIX", mattr: "Szókincs",
    reaction_applause_per_hour: "Taps/ó", reaction_laughter_per_hour: "Derültség/ó",
    reaction_heckling_per_hour: "Közbeszólás/ó", reaction_bell_per_hour: "Csengő/ó"
  }
};

const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
const tip = document.getElementById("tip");
/* d3's colour interpolators cannot parse a CSS `var(...)` string -- a scale
 * given one silently produces black. Anywhere a scale interpolates (as opposed
 * to assigning a colour straight to an attribute), resolve the variable to its
 * computed value first. */
const cssVar = name =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? "—" : Number(v).toLocaleString("hu-HU", { maximumFractionDigits: d });

let DATA, factionColour, state = { mp: null, party: "—" };

function showTip(html, event) {
  tip.innerHTML = html;
  tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + w > innerWidth - 8) x = event.clientX - w - pad;
  if (y + h > innerHeight - 8) y = event.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.opacity = 0; };

/* ------------------------------------------------------------------ boot */
d3.json("data/dashboard.json").then(data => {
  DATA = data;
  const factions = data.parties.map(p => p.faction);
  // Fixed order, assigned once: filtering must never repaint the survivors.
  factionColour = f => {
    const i = factions.indexOf(f);
    return i >= 0 ? SERIES[i % SERIES.length] : "var(--neutral)";
  };
  buildControls();
  buildTabs();
  renderProfile();
  renderTopics();
  renderNetwork();
  renderSpeeches();
});

function selectView(view) {
  const buttons = [...document.querySelectorAll(".tab[data-view]")];
  if (!buttons.some(b => b.dataset.view === view)) return;
  buttons.forEach(b => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  document.querySelectorAll("[data-panel]").forEach(p =>
    p.classList.toggle("hidden", p.dataset.panel !== view));
}

function buildTabs() {
  // Hash routing, so the header nav and a pasted link both land on the right
  // panel instead of scrolling to a hidden one.
  document.querySelectorAll(".tab[data-view]").forEach(btn => {
    btn.addEventListener("click", () => {
      history.replaceState(null, "", "#" + btn.dataset.view);
      selectView(btn.dataset.view);
    });
  });
  addEventListener("hashchange", () => selectView(location.hash.slice(1)));
  if (location.hash) selectView(location.hash.slice(1));
  const prefersDark = () => matchMedia("(prefers-color-scheme: dark)").matches;
  document.getElementById("theme").addEventListener("click", () => {
    const set = document.documentElement.getAttribute("data-theme");
    const dark = set ? set === "dark" : prefersDark();
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
    // The heatmap resolves CSS variables to real colours for interpolation, so
    // it has to be rebuilt when the theme changes; the network reads them too.
    renderTopics(); renderNetwork();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!document.documentElement.getAttribute("data-theme")) {
      renderTopics(); renderNetwork();
    }
  });
}

function buildControls() {
  const eligible = DATA.people
    .filter(p => !p.below_min_speeches && p.speaker)
    .sort((a, b) => d3.descending(a.speeches, b.speeches));
  state.mp = eligible[0].speaker_id;

  const mpSel = document.getElementById("pick-mp");
  mpSel.innerHTML = eligible.map(p =>
    `<option value="${p.speaker_id}">${p.speaker} — ${p.faction ?? "nincs frakció"} (${p.speeches})</option>`
  ).join("");
  mpSel.addEventListener("change", e => { state.mp = e.target.value; renderProfile(); });

  const partySel = document.getElementById("pick-party");
  partySel.innerHTML = `<option>—</option>` +
    DATA.parties.map(p => `<option>${p.faction}</option>`).join("");
  partySel.addEventListener("change", e => { state.party = e.target.value; renderProfile(); });
}

/* -------------------------------------------------------------- profile */
function renderProfile() {
  const person = DATA.people.find(p => p.speaker_id === state.mp);
  if (!person) return;
  const party = state.party !== "—" ? state.party : person.faction;

  document.getElementById("kpis").innerHTML = [
    ["Felszólalás", fmt(person.speeches, 0), `${fmt(person.minutes, 0)} perc`],
    ["Olvashatóság", fmt(person.lix_word_weighted, 1), "LIX, szóval súlyozva"],
    ["Hangulat", fmt(person.sentiment_valence, 3), "−1 … +1"],
    ["Derültség", fmt(person.laughter, 0), `${fmt(person.laughter_per_minute, 3)} / perc`],
    ["Taps", fmt(person.applause, 0), `${fmt(person.applause_per_minute, 3)} / perc`],
    ["Kapott közbeszólás", fmt(person.heckles_received, 0), `${fmt(person.hecklers, 0)} különböző embertől`],
    ["Adott közbeszólás", fmt(person.heckles_given, 0), `${fmt(person.targets, 0)} különböző célpontra`]
  ].map(([k, v, s]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");

  drawStrips(person, party);
  drawSentiment(person);
  drawEmotions(person);
}

function drawStrips(person, party) {
  const host = d3.select("#strips").html("");
  const keys = Object.keys(DATA.distributions);
  const W = Math.min(880, host.node().clientWidth || 880), LABEL = 190, R = 16;

  keys.forEach(key => {
    const dist = DATA.distributions[key];
    const meta = T.metrics[key] || { hu: key, hint: "" };
    const values = dist.values;
    const x = d3.scaleLinear()
      .domain(d3.extent(values, d => d.v)).nice()
      .range([LABEL + 10, W - 74]);

    const mine = values.find(d => d.id === person.speaker_id);
    const peers = party
      ? values.filter(d => (DATA.people.find(p => p.speaker_id === d.id) || {}).faction === party)
      : [];
    const median = peers.length ? d3.median(peers, d => d.v) : null;
    const pct = mine
      ? Math.round(100 * values.filter(d => d.v <= mine.v).length / values.length)
      : null;

    const svg = host.append("svg").attr("width", W).attr("height", R * 2 + 6)
      .attr("role", "img")
      .attr("aria-label", `${meta.hu}: ${mine ? fmt(mine.v, 3) : "nincs adat"}`);

    svg.append("text").attr("x", 0).attr("y", R + 4)
      .attr("fill", "var(--text-primary)").style("font-size", "12.5px")
      .text(meta.hu);
    svg.append("text").attr("x", 0).attr("y", R + 17)
      .attr("fill", "var(--text-muted)").style("font-size", "10.5px")
      .text(meta.hint);

    // axis rule, hairline and recessive
    svg.append("line").attr("x1", x.range()[0]).attr("x2", x.range()[1])
      .attr("y1", R).attr("y2", R).attr("stroke", "var(--grid)").attr("stroke-width", 1);

    svg.selectAll("circle.pop").data(values).join("circle").attr("class", "pop")
      .attr("cx", d => x(d.v)).attr("cy", R).attr("r", 3.5)
      .attr("fill", "var(--dot-population)").attr("opacity", .55);

    if (median !== null) {
      svg.append("line").attr("x1", x(median)).attr("x2", x(median))
        .attr("y1", R - 9).attr("y2", R + 9)
        .attr("stroke", factionColour(party)).attr("stroke-width", 2);
    }
    if (mine) {
      svg.append("circle").attr("cx", x(mine.v)).attr("cy", R).attr("r", 6.5)
        .attr("fill", factionColour(person.faction))
        .attr("stroke", "var(--surface-1)").attr("stroke-width", 2);
      // Direct label: the contrast check obligates a visible value.
      svg.append("text").attr("x", W - 68).attr("y", R + 4)
        .attr("fill", "var(--text-primary)").style("font-size", "12px")
        .style("font-variant-numeric", "tabular-nums")
        .text(`${fmt(mine.v, key === "sentiment_valence" ? 3 : 2)}`);
      svg.append("text").attr("x", W - 68).attr("y", R + 16)
        .attr("fill", "var(--text-muted)").style("font-size", "10px")
        .text(`${pct}. percentilis`);
    }
    svg.on("mousemove", e => showTip(
      `<strong>${meta.hu}</strong><br>${person.speaker}: ${mine ? fmt(mine.v, 3) : "—"}` +
      (pct !== null ? ` (${pct}. percentilis)` : "") +
      (median !== null ? `<br>${party} mediánja: ${fmt(median, 3)}` : "") +
      `<br><span style="color:var(--text-muted)">n = ${values.length} képviselő</span>`, e))
      .on("mouseleave", hideTip);
  });

  document.getElementById("strip-note").innerHTML =
    `A pontok a legalább ${DATA.min_speeches} felszólalással rendelkező képviselők ` +
    `(n = ${DATA.distributions.readability_lix.n}). Aki ennél kevesebbszer szólalt fel, ` +
    `nem szerepel: az arányszámai néhány mondaton alapulnának.`;
}

function drawSentiment(person) {
  const keys = ["sentiment_very_negative", "sentiment_negative", "sentiment_neutral",
    "sentiment_positive", "sentiment_very_positive"];
  const cols = ["var(--div-neg-2)", "var(--div-neg-1)", "var(--div-mid)",
    "var(--div-pos-1)", "var(--div-pos-2)"];
  const vals = keys.map(k => person[k] ?? 0);
  const total = d3.sum(vals) || 1;
  const share = vals.map(v => v / total);

  const W = 400, H = 46, mid = share[0] + share[1] + share[2] / 2;
  const x = d3.scaleLinear().domain([-mid, 1 - mid]).range([8, W - 8]);
  const svg = d3.select("#sentiment-bar").html("").append("svg")
    .attr("width", W).attr("height", H).attr("role", "img")
    .attr("aria-label", "Hangulateloszlás, semlegesre központozva");

  let cursor = -mid;
  share.forEach((s, i) => {
    const x0 = x(cursor), x1 = x(cursor + s);
    // 2px surface gap between segments rather than a border around them
    svg.append("rect").attr("x", x0).attr("y", 8)
      .attr("width", Math.max(0, x1 - x0 - 2)).attr("height", 22)
      .attr("rx", (i === 0 || i === 4) ? 4 : 0)
      .attr("fill", cols[i])
      .on("mousemove", e => showTip(`${T.sentiment[i]}: <strong>${fmt(s * 100, 1)}%</strong>`, e))
      .on("mouseleave", hideTip);
    cursor += s;
  });
  svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 4).attr("y2", 34)
    .attr("stroke", "var(--text-secondary)").attr("stroke-width", 1);
  svg.append("text").attr("x", x(0)).attr("y", 44).attr("text-anchor", "middle")
    .attr("fill", "var(--text-muted)").style("font-size", "10px").text("semleges");

  document.getElementById("sentiment-legend").innerHTML = T.sentiment.map((s, i) =>
    `<span><i class="swatch" style="background:${cols[i]}"></i>${s}</span>`).join("");
}

function drawEmotions(person) {
  const rows = [["anger", "emotion_xlm_anger"], ["joy", "emotion_xlm_joy"],
    ["sadness", "emotion_xlm_sadness"], ["fear", "emotion_xlm_fear"]];
  const corpus = k => d3.mean(DATA.people.filter(p => !p.below_min_speeches), p => p[k]);
  const W = 400, RH = 30;
  const svg = d3.select("#emotion-bars").html("").append("svg")
    .attr("width", W).attr("height", rows.length * RH + 8);
  const x = d3.scaleLinear().domain([0, 1]).range([92, W - 54]);

  rows.forEach(([label, key], i) => {
    const v = person[key] ?? 0, mean = corpus(key) ?? 0, y = i * RH + 10;
    svg.append("text").attr("x", 0).attr("y", y + 12).attr("fill", "var(--text-primary)")
      .style("font-size", "12.5px").text(T.emotions[label]);
    svg.append("rect").attr("x", x(0)).attr("y", y + 2).attr("height", 16)
      .attr("width", x(1) - x(0)).attr("fill", "var(--surface-2)").attr("rx", 4);
    svg.append("rect").attr("x", x(0)).attr("y", y + 2).attr("height", 16)
      .attr("width", Math.max(0, x(v) - x(0))).attr("fill", "var(--series-1)").attr("rx", 4)
      .on("mousemove", e => showTip(
        `<strong>${T.emotions[label]}</strong><br>${person.speaker}: ${fmt(v, 3)}` +
        `<br><span style="color:var(--text-muted)">ciklusátlag: ${fmt(mean, 3)}</span>`, e))
      .on("mouseleave", hideTip);
    svg.append("line").attr("x1", x(mean)).attr("x2", x(mean))
      .attr("y1", y).attr("y2", y + 20)
      .attr("stroke", "var(--text-secondary)").attr("stroke-width", 2);
    svg.append("text").attr("x", W - 48).attr("y", y + 14)
      .attr("fill", "var(--text-primary)").style("font-size", "12px")
      .style("font-variant-numeric", "tabular-nums").text(fmt(v, 3));
  });

  document.getElementById("emotion-note").innerHTML =
    `A függőleges vonal a ciklus átlaga. Csak a többnyelvű modell négy csatornáját ` +
    `mutatjuk. A magyar érzelemmodell <code>fear</code> és <code>sadness</code> ` +
    `csatornája megbukott az utólagos ellenőrzésen, ezért <span class="warn">nem ` +
    `szerepel</span> — lásd „A projektről”.`;
}

/* --------------------------------------------------------------- topics */
function renderTopics() {
  const cols = Object.keys(T.heat);
  const topics = DATA.topics.filter(t => t.topic >= 0 && t.n_speeches >= 8);
  // Per-column z-scores: one diverging ramp then serves every column, and 0
  // always means "the ciklus average" rather than something column-specific.
  const stats = {};
  cols.forEach(c => {
    const vals = topics.map(t => t[c]).filter(v => v !== null && v !== undefined);
    stats[c] = { mean: d3.mean(vals), sd: d3.deviation(vals) || 1 };
  });
  const z = (t, c) => (t[c] === null || t[c] === undefined)
    ? null : (t[c] - stats[c].mean) / stats[c].sd;

  const LABEL = 300, CELL = 74, RH = 26, W = LABEL + cols.length * CELL + 10;
  const H = topics.length * RH + 54;
  const colour = d3.scaleLinear()
    .domain([-2, -1, 0, 1, 2])
    .range([cssVar("--div-neg-2"), cssVar("--div-neg-1"), cssVar("--div-mid"),
      cssVar("--div-pos-1"), cssVar("--div-pos-2")])
    .interpolate(d3.interpolateLab)
    .clamp(true);

  const svg = d3.select("#heatmap").html("").append("svg")
    .attr("width", W).attr("height", H).attr("role", "img")
    .attr("aria-label", "Témák és mércék hőtérképe, szórásegységben");

  cols.forEach((c, j) => {
    svg.append("text").attr("x", LABEL + j * CELL + CELL / 2).attr("y", 30)
      .attr("text-anchor", "middle").attr("fill", "var(--text-secondary)")
      .style("font-size", "11px").text(T.heat[c]);
  });

  topics.forEach((t, i) => {
    const y = 44 + i * RH;
    svg.append("text").attr("x", 0).attr("y", y + 15)
      .attr("fill", "var(--text-primary)").style("font-size", "12px")
      .text((t.name_hu || `#${t.topic}`).slice(0, 44))
      .append("title").text(t.name_hu || "");
    svg.append("text").attr("x", LABEL - 34).attr("y", y + 15)
      .attr("fill", "var(--text-muted)").style("font-size", "10.5px")
      .style("font-variant-numeric", "tabular-nums").text(t.n_speeches);

    cols.forEach((c, j) => {
      const zv = z(t, c);
      svg.append("rect").attr("x", LABEL + j * CELL).attr("y", y)
        .attr("width", CELL - 2).attr("height", RH - 2).attr("rx", 3)
        .attr("fill", zv === null ? "var(--surface-2)" : colour(zv))
        .on("mousemove", e => showTip(
          `<strong>${t.name_hu}</strong><br>${T.heat[c]}: ${fmt(t[c], 3)}` +
          `<br><span style="color:var(--text-muted)">${zv === null ? "" :
            (zv >= 0 ? "+" : "") + fmt(zv, 2) + " szórás a ciklusátlagtól"}` +
          `<br>${t.n_speeches} felszólalás · ${fmt(t.minutes, 0)} perc</span>`, e))
        .on("mouseleave", hideTip);
    });
  });

  document.getElementById("heat-legend").innerHTML =
    `<span><i class="swatch" style="background:var(--div-neg-2)"></i>−2 szórás</span>` +
    `<span><i class="swatch" style="background:var(--div-mid)"></i>ciklusátlag</span>` +
    `<span><i class="swatch" style="background:var(--div-pos-2)"></i>+2 szórás</span>` +
    `<span style="color:var(--text-muted)">a név melletti szám: hány felszólalás</span>`;

  document.getElementById("topic-keywords").innerHTML = topics.map(t =>
    `<div style="margin-bottom:10px"><strong style="font-size:13px">${t.name_hu}</strong>
     <span style="color:var(--text-muted);font-size:12px"> · ${t.n_speeches} felszólalás</span><br>
     ${(t.keywords || []).map(k => `<span class="chip">${k}</span>`).join("")}</div>`).join("");
}

/* -------------------------------------------------------------- network */
function renderNetwork() {
  const W = Math.min(1100, document.querySelector(".wrap").clientWidth - 44), H = 620;
  const filterSel = document.getElementById("net-filter");
  const minSel = document.getElementById("net-min");

  function draw() {
    const mode = filterSel.value, minW = +minSel.value;
    const links = DATA.network.links
      .filter(l => l.weight >= minW)
      .filter(l => mode === "all" || l.crossing === mode)
      .map(l => ({ ...l }));
    const keep = new Set(links.flatMap(l => [
      typeof l.source === "object" ? l.source.id : l.source,
      typeof l.target === "object" ? l.target.id : l.target]));
    const nodes = DATA.network.nodes.filter(n => keep.has(n.id)).map(n => ({ ...n }));

    const svg = d3.select("#graph").html("").append("svg")
      .attr("width", W).attr("height", H).attr("viewBox", [0, 0, W, H])
      .attr("role", "img")
      .attr("aria-label", "Ki kit szakított félbe: irányított hálózat");

    svg.append("defs").append("marker").attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10").attr("refX", 20).attr("markerWidth", 5)
      .attr("markerHeight", 5).attr("orient", "auto")
      .append("path").attr("d", "M0,-4L9,0L0,4").attr("fill", "var(--grid)");

    const r = d3.scaleSqrt()
      .domain([0, d3.max(nodes, n => n.heckles_received) || 1]).range([4, 24]);

    const link = svg.append("g").selectAll("line").data(links).join("line")
      .attr("stroke", d => d.crossing === "cross-bench"
        ? "var(--series-2)" : "var(--neutral)")
      .attr("stroke-opacity", .4)
      .attr("stroke-width", d => Math.min(6, 1 + Math.sqrt(d.weight)))
      .attr("marker-end", "url(#arrow)");

    const node = svg.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", d => r(d.heckles_received))
      .attr("fill", d => factionColour(d.faction === "unknown" ? null : d.faction))
      .attr("stroke", "var(--surface-1)").attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("mousemove", (e, d) => showTip(
        `<strong>${d.label}</strong> — ${d.faction ?? "nincs frakció"}<br>` +
        `kapott: ${d.heckles_received} (${d.hecklers} embertől)<br>` +
        `adott: ${d.heckles_given} (${d.targets} célpontra)`, e))
      .on("mouseleave", hideTip)
      .on("click", (e, d) => {
        const sel = document.getElementById("pick-mp");
        if ([...sel.options].some(o => o.value === d.id)) {
          sel.value = d.id; state.mp = d.id; renderProfile();
          document.querySelector('.tab[data-view="profile"]').click();
          document.getElementById("profile").scrollIntoView({ behavior: "smooth" });
        }
      });

    // Direct labels for the heaviest nodes only: a name on every node is chaos.
    // A surface-coloured halo under the text keeps them readable where the
    // centre of the hairball forces two labels to overlap anyway.
    const top = new Set(nodes.slice()
      .sort((a, b) => d3.descending(a.heckles_received, b.heckles_received))
      .slice(0, 6).map(n => n.id));
    const label = svg.append("g").selectAll("text")
      .data(nodes.filter(n => top.has(n.id))).join("text")
      .attr("fill", "var(--text-primary)").style("font-size", "11.5px")
      .style("font-weight", "600")
      .style("paint-order", "stroke")
      .style("stroke", "var(--surface-1)").style("stroke-width", "3.5px")
      .style("stroke-linejoin", "round")
      .style("pointer-events", "none").text(d => d.label);

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(95).strength(.22))
      .force("charge", d3.forceManyBody().strength(-330))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collide", d3.forceCollide(d => r(d.heckles_received) + 9))
      .on("tick", () => {
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("cx", d => d.x = Math.max(26, Math.min(W - 26, d.x)))
            .attr("cy", d => d.y = Math.max(26, Math.min(H - 26, d.y)));
        label
          .attr("x", d => d.x + r(d.heckles_received) + 5)
          .attr("y", (d, i) => d.y + 4 + (i % 2 ? 11 : -7));
      });
    setTimeout(() => sim.stop(), 6000);

    document.getElementById("net-legend").innerHTML =
      DATA.parties.map(p =>
        `<span><i class="swatch" style="background:${factionColour(p.faction)}"></i>${p.faction}</span>`).join("") +
      `<span><i class="swatch" style="background:var(--series-2);opacity:.5"></i>a két oldal között</span>` +
      `<span><i class="swatch" style="background:var(--neutral);opacity:.5"></i>azonos oldalon belül</span>` +
      `<span style="color:var(--text-muted)">${nodes.length} képviselő · ${links.length} él</span>`;
  }
  filterSel.onchange = draw; minSel.onchange = draw;
  draw();
}

/* ------------------------------------------------------------- speeches */
const SP_COLS = [
  ["speaker", "Képviselő", "text"], ["faction", "Frakció", "text"],
  ["date", "Dátum", "text"], ["n_words", "Szó", "num"],
  ["lix", "LIX", "num"], ["mattr", "MATTR", "num"],
  ["sentiment_valence", "Hangulat", "num"], ["emotion_xlm_anger", "Düh", "num"],
  ["reaction_applause", "Taps", "num"], ["reaction_laughter", "Derültség", "num"],
  ["reaction_heckling", "Közbeszólás", "num"], ["textrank_keywords", "Kulcsszavak", "text"]
];
let spSort = { key: "n_words", dir: -1 };

function renderSpeeches() {
  const factions = [...new Set(DATA.speeches.map(s => s.faction).filter(Boolean))].sort();
  document.getElementById("sp-faction").innerHTML =
    `<option value="">Mind</option>` + factions.map(f => `<option>${f}</option>`).join("");
  const topics = DATA.topics.filter(t => t.topic >= 0);
  document.getElementById("sp-topic").innerHTML =
    `<option value="">Mind</option>` +
    topics.map(t => `<option value="${t.topic}">${t.name_hu}</option>`).join("");

  document.getElementById("sp-head").innerHTML = SP_COLS.map(([k, label, kind]) =>
    `<th data-key="${k}" class="${kind}">${label}</th>`).join("");
  document.querySelectorAll("#sp-head th").forEach(th => th.addEventListener("click", () => {
    const key = th.dataset.key;
    spSort = { key, dir: spSort.key === key ? -spSort.dir : -1 };
    document.querySelectorAll("#sp-head th").forEach(o => o.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", spSort.dir === 1 ? "ascending" : "descending");
    drawSpeeches();
  }));
  ["sp-search", "sp-faction", "sp-topic"].forEach(id =>
    document.getElementById(id).addEventListener("input", drawSpeeches));
  drawSpeeches();
}

function drawSpeeches() {
  const q = document.getElementById("sp-search").value.trim().toLowerCase();
  const f = document.getElementById("sp-faction").value;
  const t = document.getElementById("sp-topic").value;
  let rows = DATA.speeches.filter(s =>
    (!f || s.faction === f) &&
    (!t || String(s.topic) === t) &&
    (!q || (s.speaker || "").toLowerCase().includes(q) ||
      (s.textrank_keywords || "").toLowerCase().includes(q)));

  rows = rows.slice().sort((a, b) => {
    const x = a[spSort.key], y = b[spSort.key];
    if (x === y) return 0;
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    return (x > y ? 1 : -1) * spSort.dir;
  }).slice(0, 400);

  document.getElementById("sp-body").innerHTML = rows.map(s => `<tr>` +
    SP_COLS.map(([k, , kind]) => {
      let v = s[k];
      if (k === "textrank_keywords") v = (v || "").split(";").slice(0, 5).join(", ");
      else if (kind === "num") v = fmt(v, k === "sentiment_valence" ? 3 : 2);
      else if (k === "lix" && !s.readability_reliable) v = `${fmt(v, 1)} ⚠`;
      return `<td class="${kind}">${v ?? "—"}</td>`;
    }).join("") + `</tr>`).join("");

  const total = DATA.speeches.length;
  document.getElementById("sp-count").textContent =
    `${rows.length} sor látszik a szűrésnek megfelelő találatokból (összesen ${total} felszólalás; legfeljebb 400 jelenik meg).`;
}
