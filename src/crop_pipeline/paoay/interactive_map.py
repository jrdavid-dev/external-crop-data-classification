"""Offline map of the professor’s study boundary, reference notes, and predictions."""

import base64
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape
from shapely.ops import transform as project_geometry

from crop_pipeline.paoay import references, whole_barangay

COLORS = ("#fff", "#e08438", "#437f46", "#8b69af", "#e9c766", "#86b5d4", "#dbe0e1", "#a6a6ae")
HINTS = {
    1: "Check whether carrot was growing here during the mapped dates.",
    2: "Check whether cabbage was growing here during the mapped dates.",
    3: "Check whether potato was growing here during the mapped dates.",
    4: "Prioritize a dated crop observation. The model scores did not separate the target crops clearly.",
    5: "Check for missing imagery or collect a dated field record before assigning a crop.",
    6: "No crop label assigned. Check the field boundary if this location is known to be farmed.",
    7: "Check whether this is farmland before assigning a crop label.",
}


def _pixel_polygons(geometry, crs, inverse):
    projection = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    projected = project_geometry(projection, shape(geometry))
    polygons = list(projected.geoms) if projected.geom_type == "MultiPolygon" else [projected]
    return [[[list(map(lambda n: round(n, 3), inverse * (x, y))) for x, y in ring.coords]
              for ring in [polygon.exterior, *polygon.interiors]] for polygon in polygons]


def export(handoff, reference_handoff, destination):
    """Write a standalone view from verified rasters without resampling predictions."""
    handoff, destination = Path(handoff), Path(destination)
    saved = json.loads(handoff.read_text())
    for name, digest in saved["files"].items():
        if references.checksum(handoff.parent / name) != digest:
            raise ValueError(f"Map raster or receipt changed: {name}")
    bundle = references.load_reference_bundle(reference_handoff)
    roi = bundle["roi"]["features"][0]
    with rasterio.open(handoff.parent / "crop_map.tif") as raster:
        values = raster.read(1)
        transform = raster.transform
        width, height, crs = raster.width, raster.height, raster.crs.to_string()
    inverse = ~transform
    geometry = project_geometry(Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform,
                                shape(roi["geometry"]))
    inside = geometry_mask([mapping(geometry)], out_shape=values.shape,
                           transform=transform, invert=True)
    if np.any(values[~inside] != 0) or np.any(values[inside] == 0):
        raise ValueError("Map grid does not exactly match the verified professor study boundary")
    with rasterio.open(handoff.parent / "crop_probabilities.tif") as raster:
        if raster.shape != values.shape or raster.transform != transform or raster.crs.to_string() != crs:
            raise ValueError("Crop probabilities do not share the map grid")
        probability = raster.read()
    valid = np.all(np.isfinite(probability) & (probability >= 0) & (probability <= 1), axis=0)
    best = np.full(values.shape, 255, dtype=np.uint8)
    best[valid] = np.round(100 * probability[:, valid].max(axis=0)).astype(np.uint8)
    areas = []
    for area in references.reference_areas(bundle):
        if area["geometry_status"] == "valid":
            areas.append({"name": area["name"], "notes": area.get("notes", "").strip(),
                          "polygons": _pixel_polygons(area["geometry"], crs, inverse)})
    def encode(array):
        return base64.b64encode(array.astype(np.uint8).tobytes()).decode("ascii")
    payload = {"width": width, "height": height, "codes": encode(values), "score": encode(best),
               "outline": _pixel_polygons(roi["geometry"], crs, inverse), "areas": areas,
               "labels": whole_barangay.PREDICTION_CODES, "colors": COLORS, "hints": HINTS,
               "start": saved["period"]["start"], "end": saved["period"]["end"],
               "mode": saved["identity"]["mode"], "training": whole_barangay.fixed_training_period(saved)}
    # Keep names and labels as data even if a later KMZ includes script-like characters.
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(HTML.replace("__MAP_DATA__", data), encoding="utf-8")
    return destination


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paoay study-area crop map</title>
<style>
*{box-sizing:border-box} body{margin:0;background:#f3f6f4;color:#183228;font:15px system-ui,sans-serif}
main{max-width:1150px;margin:24px auto;padding:0 18px} h1{font-size:clamp(24px,4vw,36px);margin:0}
header p{color:#49665a;margin:8px 0 18px} .toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
button{font:inherit;background:#fff;border:1px solid #9ab1a6;border-radius:6px;padding:6px 12px;color:inherit;cursor:pointer}
button:focus-visible,canvas:focus-visible{outline:2px solid #255f49;outline-offset:2px}
.toolbar label{margin-left:8px} .mapbox{position:relative;overflow:hidden;border:1px solid #aabcb1;border-radius:10px;background:white}
canvas{display:block;width:100%;height:clamp(340px,65vh,700px);cursor:grab;touch-action:none} canvas:active{cursor:grabbing}
#tooltip{position:absolute;z-index:2;background:#17372df5;color:white;border-radius:8px;padding:9px 11px;
max-width:min(360px,85%);font-size:13px;pointer-events:none;box-shadow:0 4px 15px #15302444}
#tooltip b,#tooltip span{display:block} #tooltip span{margin-top:4px}
#notes{white-space:pre-wrap;line-height:1.35;max-height:clamp(110px,42vh,320px);overflow:hidden}
#notes-panel{margin:8px 0 14px;background:white;border:1px solid #aabcb1;border-radius:8px;padding:9px 12px}
#notes-detail{white-space:pre-wrap;margin:4px 0 0;color:#37574b}
[hidden]{display:none!important}
#details{margin:11px 0;color:#37574b;min-height:1.4em}
#legend{display:flex;gap:8px 18px;flex-wrap:wrap;margin:12px 0} #legend span{display:inline-flex;align-items:center;gap:6px}
#legend i{display:inline-block;width:13px;height:13px;border-radius:3px;border:1px solid #0003}
footer{color:#49665a;font-size:13px;line-height:1.5}
</style>
</head>
<body><main>
<header><h1>Paoay study-area crop map</h1><p id="subtitle"></p></header>
<div class="toolbar">
<button id="zoom-in" type="button" aria-label="Zoom in">+</button>
<button id="zoom-out" type="button" aria-label="Zoom out">−</button>
<button id="reset" type="button">Fit study boundary</button>
<label><input id="areas" type="checkbox" checked> Reference areas</label>
</div>
<div class="mapbox"><canvas id="map" tabindex="0" aria-label="Crop map. Drag or use arrow keys to pan. Plus and minus zoom. Hover or tap a pixel to inspect it."></canvas>
<div id="tooltip" hidden><b id="label"></b><span id="notes"></span><span id="score"></span><span id="hint"></span></div></div>
<p id="details">Hover or tap a red reference area to see its original KMZ crop notes.</p>
<div id="notes-panel" hidden><strong id="notes-title"></strong><p id="notes-detail"></p></div>
<strong>Threshold-qualified crop prediction</strong>
<div id="legend" aria-label="Map legend"></div>
<footer>Crop labels require a top model score of at least 60% and a lead of at least 10 percentage points over the next crop.
Local accuracy is unvalidated. <span id="season-note"></span>
Farmland source: <a href="https://doi.org/10.5281/zenodo.19303406">SEA_Cropland10 2024</a>.</footer>
</main>
<script type="application/json" id="map-data">__MAP_DATA__</script>
<script>
"use strict";
const d = JSON.parse(document.getElementById("map-data").textContent);
const get = (id) => document.getElementById(id);
const canvas = get("map"), ctx = canvas.getContext("2d"), tooltip = get("tooltip"), areas = get("areas");
const codes = Uint8Array.from(atob(d.codes), v => v.charCodeAt(0));
const scores = Uint8Array.from(atob(d.score), v => v.charCodeAt(0));
const raster = document.createElement("canvas");
raster.width = d.width; raster.height = d.height;
const rasterContext = raster.getContext("2d");
const pixels = rasterContext.createImageData(d.width, d.height);
const colors = d.colors.map(c => c.length === 4 ? [255,255,255] :
    [1,3,5].map(i => parseInt(c.slice(i, i+2),16)));
codes.forEach((code,index) => {
  if (!code) return;
  pixels.data.set([...colors[code],255], index*4);
});
rasterContext.putImageData(pixels,0,0);
get("subtitle").textContent = `${d.start} to ${d.end} (end exclusive) · Professor’s study boundary · ${d.mode === "exploratory_aligned_calendar_proxy"
 ? "JECAM-adapted forest on a calendar proxy"
 : d.mode === "exploratory_shifted_calendar" ? "exploratory calendar shift" : "calendar matches training"}`;
if (d.mode === "exploratory_shifted_calendar") {
  const clarity = d.start === "2024-01-01" && d.end === "2024-05-01"
    ? "January–April 2024 was chosen for clear imagery. " : "";
  get("season-note").textContent = clarity + (d.training ? `The fixed model was trained on ${d.training} imagery.` : "");
}
Object.entries(d.labels).forEach(([key,label]) => {
  const item = document.createElement("span"), swatch = document.createElement("i");
  swatch.style.background = d.colors[key]; item.append(swatch,document.createTextNode(label)); get("legend").append(item);
});
const state = {scale:1, fit:1, x:0, y:0, width:0, height:0, drag:null};
function drawShape(polygons, stroke, width, dash) {
  ctx.beginPath();
  for (const polygon of polygons) for (const ring of polygon) {
    ring.forEach(([x,y],index) => index ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
    ctx.closePath();
  }
  ctx.strokeStyle=stroke; ctx.lineWidth=width/state.scale;
  ctx.setLineDash(dash.map(v => v/state.scale)); ctx.stroke(); ctx.setLineDash([]);
}
function render() {
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.clearRect(0,0,state.width,state.height);
  ctx.save(); ctx.translate(state.x,state.y); ctx.scale(state.scale,state.scale);
  ctx.imageSmoothingEnabled=false; ctx.drawImage(raster,0,0);
  drawShape(d.outline,"#fff",6,[]);
  drawShape(d.outline,"#9c1122",3.5,[]);
  if (areas.checked) for (const area of d.areas) {
    drawShape(area.polygons,"#fff",5,[]);
    drawShape(area.polygons,"#d4132c",2.8,[5,3]);
  }
  ctx.restore();
}
function fit() {
  const rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1;
  state.width=rect.width; state.height=rect.height;
  canvas.width=Math.round(rect.width*ratio); canvas.height=Math.round(rect.height*ratio);
  state.fit=Math.min((rect.width-28)/d.width,(rect.height-28)/d.height);
  state.scale=state.fit; state.x=(state.width-d.width*state.scale)/2;
  state.y=(state.height-d.height*state.scale)/2; tooltip.hidden=true; render();
}
function zoom(factor,x=state.width/2,y=state.height/2) {
  const scale=Math.max(state.fit*.7,Math.min(state.fit*12,state.scale*factor));
  state.x=x-(x-state.x)*scale/state.scale; state.y=y-(y-state.y)*scale/state.scale;
  state.scale=scale; tooltip.hidden=true; render();
}
function point(event) {const r=canvas.getBoundingClientRect(); return [event.clientX-r.left,event.clientY-r.top];}
function inRing(x,y,ring) {
  let inside=false;
  for(let i=0,j=ring.length-1;i<ring.length;j=i++){
    const [a,b]=ring[i], [c,e]=ring[j];
    if ((b>y)!==(e>y) && x<(c-a)*(y-b)/(e-b)+a) inside=!inside;
  }
  return inside;
}
function inspect(px,py) {
  const x=(px-state.x)/state.scale, y=(py-state.y)/state.scale;
  const col=Math.floor(x), row=Math.floor(y);
  if(col<0||row<0||col>=d.width||row>=d.height){tooltip.hidden=true; return;}
  const index=row*d.width+col, code=codes[index];
  const matches = areas.checked ? d.areas.filter(
    area=>area.polygons.some(r=>inRing(x,y,r[0])&&!r.slice(1).some(h=>inRing(x,y,h)))
  ) : [];
  if (!code && !matches.length) {tooltip.hidden=true; return;}
  const title = matches.map(area=>area.name).join(" / ");
  const notes = matches.map(area=>[
    matches.length>1 ? area.name : "",
    area.notes || "No crop note was supplied for this reference area."
  ].filter(Boolean).join("\n")).join("\n\n");
  get("label").textContent = title ? `${title} · Original KMZ notes` : d.labels[code];
  get("notes").textContent = notes;
  get("notes").hidden = !notes;
  const modelScore = scores[index]===255 ? "" : ` · top score ${scores[index]}%`;
  get("score").textContent = code ? `Model map: ${d.labels[code]}${modelScore}` : "";
  get("hint").textContent = (!matches.length && code) ? d.hints[code] : "";
  get("details").textContent = [get("score").textContent,get("hint").textContent].filter(Boolean).join(" · ");
  if (matches.length) {
    get("notes-title").textContent = `${title} · Original KMZ notes`;
    get("notes-detail").textContent = notes;
    get("notes-panel").hidden = false;
  }
  tooltip.hidden=false;
  tooltip.style.left=Math.max(4,Math.min(px+14,state.width-tooltip.offsetWidth-7))+"px";
  tooltip.style.top=Math.max(4,Math.min(py+14,state.height-tooltip.offsetHeight-7))+"px";
}
canvas.addEventListener("pointermove",e=>{
  const [x,y]=point(e);
  if(state.drag){
    if (Math.abs(x-state.drag.sx)+Math.abs(y-state.drag.sy)>3) state.drag.moved=true;
    if(state.drag.moved){state.x+=x-state.drag.x;state.y+=y-state.drag.y;tooltip.hidden=true;render();}
    state.drag.x=x;state.drag.y=y;
  } else inspect(x,y);
});
canvas.addEventListener("pointerdown",e=>{
  const [x,y]=point(e);state.drag={x,y,sx:x,sy:y,moved:false};canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointerup",e=>{
  const drag=state.drag;state.drag=null;if(drag && !drag.moved) inspect(...point(e));
});
canvas.addEventListener("pointercancel",()=>{state.drag=null;tooltip.hidden=true;});
canvas.addEventListener("pointerleave",()=>{tooltip.hidden=true;});
canvas.addEventListener("wheel",e=>{e.preventDefault();zoom(e.deltaY<0?1.2:1/1.2,...point(e));},{passive:false});
canvas.addEventListener("keydown",e=>{
  if(e.key==="+"||e.key==="=") zoom(1.25);
  else if(e.key==="-") zoom(.8);
  else if(e.key.startsWith("Arrow")) {
    state.x+=e.key==="ArrowLeft"?24:e.key==="ArrowRight"?-24:0;
    state.y+=e.key==="ArrowUp"?24:e.key==="ArrowDown"?-24:0;tooltip.hidden=true;render();
  } else return;
  e.preventDefault();
});
get("zoom-in").addEventListener("click",()=>zoom(1.25));
get("zoom-out").addEventListener("click",()=>zoom(.8));
get("reset").addEventListener("click",fit);
areas.addEventListener("change",()=>{tooltip.hidden=true;get("notes-panel").hidden=true;render();});
window.addEventListener("resize",fit);
fit();
</script>
</body></html>
"""
