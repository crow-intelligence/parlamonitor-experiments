# Parlamonitor dashboard

A static page. No server, no build step: `index.html`, `app.js`, and one JSON
bundle it fetches.

```bash
uv run python scripts/build_dashboard_data.py   # writes data/dashboard.json
python3 -m http.server 8765 --directory dashboard
# then open http://127.0.0.1:8765/
```

It must be served over HTTP rather than opened as `file://`, because it fetches
the bundle.

`data/dashboard.json` (~2.2 MB) is derived and therefore gitignored, like
everything else under `data/derived/`. Rebuild it with the command above; it
takes a second and needs no models.

## Why these forms

**Not a radar chart.** A radar encodes magnitude as radius, so area grows as the
square and a 2x score reads as 4x; its axis order is arbitrary and rotating it
changes the silhouette a reader interprets; and the metrics here are
incommensurable (LIX ~20-85, MATTR 0-1, valence -1...+1, laughs 0-176), so a
radar could only show them normalised, which hides the values.

The profile view uses **percentile strips** instead: every eligible MP as a
faint dot, the selected one filled, the party median as a tick. The question is
"is this MP unusual, and which way", and that is a comparison against a
population -- so the population is drawn.

Elsewhere: a **diverging stacked bar** centred on neutral for the ordered
sentiment scale; a **heatmap on per-column z-scores** for topic x metric, so one
diverging ramp serves every column and 0 always means the cycle average; a
**force layout** for the interruption network.

### The two network sizings

The network offers node size by **in-degree** (how often a member was
interrupted) or **out-degree** (how often they interrupted others). Two
decisions make the pair comparable rather than two unrelated pictures:

- **One shared radius scale**, over the larger of the two maxima (289 received,
  247 given). Independent scales would make everyone look equally central in
  both views and hide the asymmetry, which is the whole point.
- **The layout is not re-run when you switch.** Nodes resize in place, so what
  you see is the change itself. Only the collide force is updated and the
  simulation gently reheated to relax overlaps.

The labelled six follow the active metric, so the names change with the view.
Edge thickness is the edge weight -- how many times that ordered pair happened
-- on a sqrt scale over 1-64, because the median edge is 1 and a linear scale
would render almost everything hairline.

The two views are deep-linkable: `?size=in` and `?size=out`.

## Colour

Four categorical slots, validated with the dataviz skill's checker:

```
validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light   # all pass
validate_palette.js "#3987e5,#d95926,#199e70,#c98500" --mode dark    # all pass
```

Light mode returns a contrast warning, which is why every strip carries a
direct value label and the speeches tab exists as a full table view.

**These are not party brand colours.** The brand hex values for TISZA, KDNP and
Mi Hazank are not reliably known here, and a wrong party colour in a political
dashboard is a factual error, not a styling choice. The slots are assigned in
fixed order so a filter never repaints the survivors. Swap them for real brand
values and re-run the validator if you have them.

Dark mode is a selected set of steps against the dark surface, not an automatic
flip, and it is validated separately.

## Robustness the strips needed

A percentile strip shows the whole population, so a single outlier squashes
everyone else onto one edge. Two records did exactly that and both were the
same artefact — the notary's roll-call, a list of names with no sentence
punctuation. It scores LIX 594 and MHD 26.3 against corpus medians of 40 and
2.4.

The fix is upstream of the chart, not in it: `readability_reliable` flags the
record for LIX, and the syntax metrics cap sentence length at 120 tokens. Both
thresholds come from the measured distribution rather than taste — sentence
length here is 15 tokens at the median, 102 at the 99.9th percentile, and that
"sentence" is 517.

## Gotchas found while building this

- **`d3.scaleLinear` cannot interpolate CSS `var()` strings** -- a scale given one
  silently produces black, which is what the whole heatmap did until it was
  rendered and looked at. It now resolves variables to computed values first
  (`cssVar()`), which is also why it re-renders on a theme change.
- **Hardcoding `data-theme="light"` on `<html>` defeats `prefers-color-scheme`
  entirely.** The attribute is set only by the toggle.
- **Force-layout labels collide** in the centre of a hairball. Only the six
  heaviest nodes are labelled, with a surface-coloured halo under the text.
