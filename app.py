from __future__ import annotations
import os
import hashlib
from datetime import datetime
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import sqlite3
import streamlit as st
from collections import OrderedDict
import streamlit.components.v1 as components
try:
    from utils.rule_checker import (
        run_suburb_checks, run_road_checks, run_building_checks, run_blocks_checks,
        FloorPlanChecker, failures_to_frame
    )
except Exception:
    from rule_checker import (
        run_suburb_checks, run_road_checks, run_building_checks, run_blocks_checks,
        FloorPlanChecker, failures_to_frame
    )
try:
    from validation_rules import RULES
except Exception:
    RULES = {}
from utils.registry_db import ensure_schema_planB, upsert_dataset_row, begin_run, write_violations, finish_run
HIDE_RULE_PICKER = False  
def build_rule_name_map(dtype: str) -> "OrderedDict[str, str]":
    base = OrderedDict([("Auto (based on dataset type)", "auto")])

    if dtype == "blocks":
        base.update({
            "Invalid geometry": "InvalidGeometry",
            "Polygon overlap": "PolygonOverlap",
            "Short edge (< threshold)": "ShortEdge",
            "Wrong orientation (outer ring CW)": "WrongOrientation",
            "Non-manifold edges": "NonmanifoldEdge",
            "Min interior angle too small": "MinInteriorAngle",
            "Excessive precision": "ExcessivePrecision",
            "Outside bounding box": "OutsideBBOX",
            "Touches only": "TouchesOnly",
            "Parts self-overlap": "PartsSelfOverlap",
            "Sliver polygons": "SliverPolygons",
        })
    elif dtype == "building":
        base.update({
            "Geometry not empty": "GeometryNotEmpty",
            "Geometry validity": "GeometryValid",
            "Polygons only": "BuildingPolygonsOnly",
            "Overlapping buildings": "OverlappingBuildings",
            "Holes/area ratio too large": "HolesAreaRatioLimit",
            "Total holes area too large": "HolesTotalAreaLimit",
            "Minimum area below threshold": "MinimumArea",
            "Parts self-overlap (MultiPolygon)": "PartsSelfOverlap",
            "Sliver polygons": "SliverPolygons",
            "Duplicate geometries": "DuplicateGeometries",
            "CRS not in allowed set": "ExpectedCRS",
        })
    elif dtype == "road":
        base.update({
            "Endpoint candidates": "EndpointCandidate",
            "Isolated segment": "IsolatedSegment",
        })
    elif dtype == "suburb":
        base.update({
            "Invalid geometry": "InvalidGeometry",
            "Overlaps": "Overlap",                     
            "Sliver polygons": "SliverPolygon",
            "Area <= 0": "AreaNonPositive",
            "Topological gaps": "TopologicalGaps",
            "Small holes": "SmallHole",
            "Non-contiguous parts": "NonContiguousParts",
            "Boundary misalignment": "BoundaryMisalignment",
            "Very short shared boundary": "VeryShortSharedBoundary",
            "Enclave inside another": "Enclave",
            "CRS not projected": "CRSNotProjected",
            "Centroid outside polygon": "CentroidOutside",
            "Duplicate geometry": "DuplicateGeometry",
            "Name near-duplicates": "NearDuplicateName",
            "Neighbour attr mismatch": "NeighbourAttributeMismatch",
            "Suburb↔Postcode inconsistency (1→N)": "SuburbToMultiplePostcodes",
            "Postcode↔Suburb inconsistency (1→N)": "PostcodeToMultipleSuburbs",
        })
    elif dtype == "floor":
        base.update({
            "Geometry validity": "GeometryValidity",
            "Overlaps (between blocks)": "GeometryOverlap",
            "NULL values": "NullValues",
            "Negative floor_space": "NegativeFloorSpace",
            "Sum of components > total": "TotalConsistency",
            "CRS missing": "CRSCheck",
            "Too many uses per block": "ConflictBlockUses",
        })
    try:
        from validation_rules import RULES as _RULES
        for k in _RULES.keys():
            if k not in base.values() and k not in base:
                base[k] = k
    except Exception:
        pass
    return base
st.markdown("""
<style>
#bg-root, #bg-root::before, #bg-root::after { position:fixed; inset:0; pointer-events:none; }
#bg-root { z-index:-3; background:#0b0f17; }
#bg-root::before{
  content:""; z-index:-3;
  background:
    repeating-linear-gradient(90deg, rgba(0,255,255,.065) 0 1px, transparent 1px 40px),
    repeating-linear-gradient(0deg,  rgba(0,255,255,.065) 0 1px, transparent 1px 40px);
  -webkit-mask-image: radial-gradient(1200px 800px at 50% 35%, #000 45%, transparent 80%);
          mask-image: radial-gradient(1200px 800px at 50% 35%, #000 45%, transparent 80%);
  animation:gridShift 40s linear infinite;
}
@keyframes gridShift{ from{transform:translateY(0)} to{transform:translateY(40px)} }
#bg-root::after{
  content:""; z-index:-3;
  background: conic-gradient(from 180deg at 50% 50%,
    rgba(0,255,255,0.00), rgba(0,255,255,0.10),
    rgba(255,0,180,0.11), rgba(0,255,255,0.00));
  filter: blur(42px); opacity:.85;
  animation: sweep 22s linear infinite;
}
@keyframes sweep{ 0%{transform:translateX(-18%) rotate(0deg)} 100%{transform:translateX(18%) rotate(360deg)} }
.bg-stars,.bg-stars2,.bg-scan {
  position:fixed; inset:0; pointer-events:none; z-index:-3;
}
.bg-stars{
  background-image:
    radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,.60), transparent 60%),
    radial-gradient(2px 2px at 70% 20%, rgba(255,255,255,.35), transparent 60%),
    radial-gradient(1.5px 1.5px at 35% 75%, rgba(255,255,255,.45), transparent 60%),
    radial-gradient(1.5px 1.5px at 85% 60%, rgba(255,255,255,.35), transparent 60%),
    radial-gradient(1.2px 1.2px at 55% 50%, rgba(255,255,255,.50), transparent 60%);
  animation: twinkle 8s ease-in-out infinite alternate;
}
.bg-stars2{
  background-image:
    radial-gradient(1.2px 1.2px at 15% 40%, rgba(255,255,255,.35), transparent 60%),
    radial-gradient(1.2px 1.2px at 65% 30%, rgba(255,255,255,.30), transparent 60%),
    radial-gradient(1px 1px at 45% 65%, rgba(255,255,255,.25), transparent 60%),
    radial-gradient(1px 1px at 80% 70%, rgba(255,255,255,.25), transparent 60%);
  animation: twinkle2 12s ease-in-out infinite alternate-reverse;
  filter: blur(.3px);
}
@keyframes twinkle{ from{opacity:.45; transform:scale(1)} to{opacity:.8; transform:scale(1.015)} }
@keyframes twinkle2{ from{opacity:.3; transform:scale(1)} to{opacity:.65; transform:scale(1.02)} }
.bg-scan{
  background: repeating-linear-gradient(0deg, rgba(255,255,255,.02) 0 1px, transparent 1px 3px);
  mix-blend-mode: overlay; opacity:.25;
}
.block-container{ max-width:1100px; margin:0 auto; position:relative; z-index:10; }
@keyframes glitchAnim{
  0% { clip-path: inset(0 0 0 0); transform:translate(0,0) }
  20%{ clip-path: inset(0 0 60% 0); transform:translate(1px,-1px) }
  40%{ clip-path: inset(40% 0 0 0); transform:translate(-1px,1px) }
  60%{ clip-path: inset(0 0 70% 0); transform:translate(2px,0) }
  80%{ clip-path: inset(30% 0 0 0); transform:translate(-2px,0) }
  100%{ clip-path: inset(0 0 0 0); transform:translate(0,0) }
}
.glitch{
  text-align:center; margin:32px 0 16px; font-size:42px; letter-spacing:.5px;
  position:relative; color:#e7f2ff; text-shadow:0 0 14px rgba(0,255,255,.4); font-weight:800;
}
.glitch::before, .glitch::after{
  content: attr(data-text); position:absolute; left:0; right:0; top:0; text-align:center;
  mix-blend-mode: screen; opacity:.65; filter: blur(.3px);
}
.glitch::before{ color:#00ffff; transform:translateX(2px); animation:glitchAnim 3s infinite linear alternate; }
.glitch::after { color:#ff40b0; transform:translateX(-2px); animation:glitchAnim 2.8s infinite linear alternate-reverse; }
div[data-testid="stVerticalBlock"] > div:first-child{
  position:relative; background: rgba(255,255,255,.05);
  backdrop-filter: blur(6px); border-radius: 16px; padding: 24px 28px;
  border: 1px solid rgba(255,255,255,.08); box-shadow: 0 10px 35px rgba(0,0,0,.45);
}
div[data-testid="stVerticalBlock"] > div:first-child::before{
  content:""; position:absolute; inset:-1px; border-radius: 17px; pointer-events:none;
  background: linear-gradient(135deg, rgba(0,255,255,.5), rgba(255,0,160,.5), rgba(0,255,255,.5));
  filter: blur(8px); opacity:.35; animation: borderGlow 8s linear infinite;
}
@keyframes borderGlow{
  0%{ filter: blur(8px) brightness(1) } 50%{ filter: blur(10px) brightness(1.2) } 100%{ filter: blur(8px) brightness(1) }
}
div[data-baseweb="select"]:focus-within,
div[data-testid="stFileUploaderDropzone"]:focus-within{
  box-shadow: 0 0 0 2px rgba(0,255,255,.22), inset 0 0 20px rgba(0,255,255,.15);
  border-color: rgba(0,255,255,.55) !important;
}
.stButton>button{
  border:1px solid rgba(0,255,255,.45);
  background: radial-gradient(120% 120% at 50% -20%, rgba(0,255,255,.25), rgba(255,0,160,.18) 60%, rgba(0,0,0,.35));
  color:#eaffff; text-transform:uppercase; letter-spacing:.06em; font-weight:700;
  box-shadow: 0 0 20px rgba(0,255,255,.18), inset 0 0 12px rgba(0,255,255,.12);
  transition: transform .15s ease, box-shadow .2s ease, filter .2s ease;
  backdrop-filter: blur(3px);
}
.stButton>button:hover{
  transform: translateY(-2px);
  box-shadow: 0 0 28px rgba(0,255,255,.35), inset 0 0 14px rgba(0,255,255,.18);
  filter: saturate(1.1);
}
div[data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08);
}
[data-testid="stAppViewContainer"]{ position: relative; z-index: 0; background: transparent !important; }
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after{
  content:""; position: fixed; top:0; bottom:0; width:160px; pointer-events:none;
  z-index:2; mix-blend-mode: screen; opacity:.68;
  background: radial-gradient(140px 70% at 50% 50%,
              rgba(0,255,255,.88), rgba(123,97,255,.55) 45%,
              rgba(255,0,200,.45) 70%, transparent 80%);
  filter: blur(36px);
  animation: fxHue 14s linear infinite, fxFloat 9s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"]::before{ left:0;  transform: translateX(-26px); }
[data-testid="stAppViewContainer"]::after { right:0; transform: translateX(26px) scaleX(-1); }
.preview-cap-label {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  font-weight: 600;
  font-size: 0.95rem;
  color: #e5e7eb;   /*  */
  padding: 0 14px;
  z-index: 3;       /*  */
}
.preview-cap-label.left {
  left: 18px;
  text-align: left;
}
.preview-cap-label.right {
  right: 18px;
  text-align: right;
}
</style>
<div id="bg-root"></div>
<div class="bg-stars"></div>
<div class="bg-stars2"></div>
<div class="bg-scan"></div>
<style>
#bg-root, #bg-root::before, #bg-root::after,
.bg-stars, .bg-stars2, .bg-scan,
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after{
  pointer-events:none !important;
}
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after{
  z-index:-1 !important;  /*  */
}
</style>
<style>
div[data-testid="stSlider"]{
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 8px 24px rgba(0,0,0,.35);
  backdrop-filter: blur(6px);
  margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<h1 class="glitch" data-text="Spatial Data Validation Demo">
  Spatial Data Validation Demo
</h1>
""", unsafe_allow_html=True)
lp, center, rp = st.columns([1, 2, 1])
with center:
    dtype = st.selectbox(
        "Select dataset type",
        ["suburb", "blocks", "road", "building", "floor"],  
        index=0,
    )
    DB_DIR = os.environ.get("VALIDATION_DB_DIR", ".")
    def default_db_for(_dtype: str) -> str:
        return os.path.join(DB_DIR, f"validation_registry_{_dtype}.db")
    db_name = default_db_for(dtype)
    try:
        dev_param = st.query_params.get("dev", "0")
        if isinstance(dev_param, list):
            dev_param = dev_param[0]
    except Exception:
        dev_param = "0"
    if dev_param == "1":
        st.caption(f"DEV: results will be saved to `{db_name}`")
    file_format = st.selectbox("Select file format", ["GeoJSON", "CSV", "XML", "KML"], index=0)
    ext_map = {"GeoJSON": ["geojson", "json"], "CSV": ["csv"], "XML": ["xml"], "KML": ["kml"]}
    uploaded_file = st.file_uploader("Browse files", type=ext_map[file_format])
    if not HIDE_RULE_PICKER:
        rule_name_map = build_rule_name_map(dtype)
        selected_rule_names = st.multiselect(
            "Select validation function(s)",
            options=list(rule_name_map.keys()),
            default=["Auto (based on dataset type)"],
            help="Auto = run all checks for the selected dataset type"
        )
        selected_rule_keys = [rule_name_map[name] for name in selected_rule_names]
    else:
        selected_rule_keys = []  
def load_geodata(_file, _fmt: str) -> gpd.GeoDataFrame:
    import json
    import re
    import pandas as pd
    import geopandas as gpd
    from shapely import wkt as _wkt
    from shapely.geometry import shape
    def _looks_like_wkt(s: pd.Series) -> bool:
        return s.astype(str).str.match(
            r"^\s*(MULTI)?(POINT|LINESTRING|POLYGON)\s*\(",
            case=False, na=False
        ).any()
    def _looks_like_json(s: pd.Series) -> bool:
        t = s.astype(str).str.strip()
        return (t.str.startswith("{") | t.str.startswith("[")).any()
    def _tabular_df_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
        cols_lc = {c.lower(): c for c in df.columns}
        lon_syn = ["lon","long","longitude","lng","x","point_x","xcoord","easting"]
        lat_syn = ["lat","latitude","y","point_y","ycoord","northing"]
        lon_key = next((cols_lc.get(s) for s in lon_syn if s in cols_lc), None)
        lat_key = next((cols_lc.get(s) for s in lat_syn if s in cols_lc), None)
        if lon_key and lat_key:
            crs_txt = st.text_input("CRS for (lon/lat) columns", "EPSG:4326")
            crs_txt = crs_txt if crs_txt.upper().startswith("EPSG") else f"EPSG:{crs_txt}"
            return gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df[lon_key], df[lat_key]), crs=crs_txt
            )
        name_candidates = [cols_lc.get(k) for k in
                           ["wkt","geometry","geom","the_geom","shape","geom_wkt","geojson"]
                           if k in cols_lc]
        content_candidates = []
        for c in df.columns:
            s = df[c].dropna()
            if s.empty:
                continue
            if _looks_like_wkt(s) or _looks_like_json(s):
                content_candidates.append(c)
        candidates = list(dict.fromkeys([c for c in (name_candidates + content_candidates) if c]))
        if not candidates:
            raise ValueError("Geometry not detected. Please provide: (lon,lat)/(x,y) columns; or a geometry column containing WKT/GeoJSON (e.g., ‘geometry’/'geom'/‘the_geom’/'wkt').")
        geom_col = st.selectbox("Select geometry column (WKT/GeoJSON)", candidates, index=0)
        ser = df[geom_col].astype(str)
        if _looks_like_wkt(ser):
            geom = ser.apply(_wkt.loads)
            def _second_abs_gt90(text):
                m = re.search(r"[-+]?\d+\.?\d*\s+([-+]?\d+\.?\d*)", text)
                return abs(float(m.group(1))) > 90 if m else False
            swap_default = bool(ser.dropna().astype(str).map(_second_abs_gt90).mean() > 0.5)
            if st.checkbox("WKT columns are (lat,lon) order (requires swapping XY)", value=swap_default):
                from shapely.ops import transform
                def _swap_xy(g):
                    if g is None or g.is_empty: return g
                    return transform(lambda x,y,z=None:(y,x) if z is None else (y,x,z), g)
                geom = geom.apply(_swap_xy)
        else:
            def _to_shape(txt):
                try:
                    obj = json.loads(txt)
                    gobj = obj.get("geometry", obj) if isinstance(obj, dict) else obj
                    return shape(gobj)
                except Exception:
                    return None
            geom = ser.apply(_to_shape)

        preset = st.selectbox(
            "CRS preset (AUS quick picks)",
            ["(manual)", "EPSG:28355  GDA94 / MGA Zone 55",
             "EPSG:7855  GDA2020 / MGA Zone 55",
             "EPSG:3111  VicGrid94",
             "EPSG:7899  VicGrid2020"],
            index=0,
        )
        manual = st.text_input("CRS of the geometry column", "EPSG:4326")
        crs_txt = manual if preset == "(manual)" else preset.split()[0]
        crs_txt = crs_txt if crs_txt.upper().startswith("EPSG") else f"EPSG:{crs_txt}"
        gdf = gpd.GeoDataFrame(df, geometry=geom)
        try: gdf = gdf.set_crs(crs_txt)
        except Exception: pass
        return gdf[~gdf.geometry.isna()].copy()
    if _fmt == "CSV":
        df = pd.read_csv(_file)
        return _tabular_df_to_gdf(df)
    if _fmt in {"GeoJSON","XML","KML"}:
        gdf0 = gpd.read_file(_file)  
        if gdf0.crs is None:
            try:
                txt = _file.getvalue().decode("utf-8", "ignore")
            except Exception:
                try:
                    _file.seek(0)
                    txt = _file.read().decode("utf-8", "ignore")
                except Exception:
                    txt = ""
            import re
            m = re.search(r'srsName="[^"]*EPSG::(\d+)"', txt)
            if m:
                gdf0 = gdf0.set_crs(epsg=int(m.group(1)))  
            else:
                gdf0 = gdf0.set_crs(epsg=4326)  
        if gdf0.geometry.geom_type.isin(["Point","MultiPoint"]).all():
            df_attr = pd.DataFrame(gdf0.drop(columns="geometry"))
            try:
                gdf_alt = _tabular_df_to_gdf(df_attr)
                if gdf_alt.geometry.geom_type.isin(
                    ["LineString","MultiLineString","Polygon","MultiPolygon"]
                ).any():
                    return gdf_alt
            except Exception:
                pass
        return gdf0

    if _fmt == "XML":
        try:
            df = pd.read_xml(_file)
            if df is not None and not df.empty:
                return _tabular_df_to_gdf(df)
        except Exception:
            try:
                return gpd.read_file(_file)
            except Exception as e2:
                raise ValueError(f"XML cannot be parsed into tables or GIS：{e2}") from e2
    raise ValueError(f"Unsupported format: {_fmt}")
def ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS validations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_type TEXT NOT NULL,
        feature_hash TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('invalid','resolved')),
        error_count INTEGER,
        output_error TEXT,
        rule_keys TEXT,
        validated_at TEXT,
        resolved_at TEXT,
        lat REAL,
        lon REAL,
        UNIQUE(feature_hash, dataset_type)
    )
    """)
    conn.commit()
def run_by_dtype(_gdf: gpd.GeoDataFrame, _dtype: str) -> pd.DataFrame:
    expected = {
        "suburb": {"Polygon", "MultiPolygon"},
        "building": {"Polygon", "MultiPolygon"},
        "blocks": {"Polygon", "MultiPolygon"},
        "road": {"LineString", "MultiLineString"},
        "floor": {"Polygon", "MultiPolygon"},  
    }
    kinds = set(_gdf.geometry.geom_type.unique().tolist())
    if _dtype in expected and kinds.isdisjoint(expected[_dtype]):
        st.error(f"`{_dtype}` expects {sorted(expected[_dtype])}, got {sorted(kinds)}")
        st.stop()
    if _dtype == "suburb":
        min_area = st.number_input("Sliver area threshold (m²)", min_value=0.0, value=30.0, step=10.0)
        return run_suburb_checks(_gdf, min_area_m2=min_area)
    elif _dtype == "road":
        c1, c2 = st.columns(2)
        with c1: start_col = st.text_input("Start node column", "start_node_name")
        with c2: end_col   = st.text_input("End node column",   "end_node_name")
        return run_road_checks(_gdf, start_col=start_col, end_col=end_col)
    elif _dtype == "building":
        c1, c2, c3 = st.columns(3)
        with c1:
            overlap_tol = st.number_input("Overlap tolerance (m²)", min_value=0.0, value=0.05, step=0.05)
        with c2:
            holes_ratio = st.number_input("Max holes/area ratio", min_value=0.0, value=0.20, step=0.05)
        with c3:
            min_area = st.number_input("Min building area (m²)", min_value=0.0, value=1.0, step=0.5)

        c4, c5 = st.columns(2)
        with c4:
            holes_total = st.number_input("Max total holes area (m²)", min_value=0.0, value=2.0, step=0.5)
        with c5:
            sliver_ratio = st.number_input("Min area/perimeter ratio", min_value=0.0, value=0.20, step=0.05)

        return run_building_checks(
            _gdf,
            overlap_tol_m2=overlap_tol,
            holes_ratio_max=holes_ratio,
            min_area_m2=min_area,
            hole_total_area_max_m2=holes_total,
            sliver_ratio_min=sliver_ratio,
        )
    elif _dtype == "blocks":
        min_len = st.number_input("Short edge threshold (m)", min_value=0.0, value=1.0, step=0.5)
        return run_blocks_checks(_gdf, min_len_m=min_len)
    elif _dtype == "floor":
        default_path = "case_study/melbourne_city/geoscape-geoscape-melbourne-buildings-jun22-na.geojson"
        gdf_local = _gdf if _gdf is not None and not _gdf.empty else gpd.read_file(default_path)
        checker = FloorPlanChecker()
        all_fails = checker.run_all(gdf_local)        
        report = failures_to_frame(all_fails)         
        overlap_only = [f for f in all_fails if f.error_type == "GeometryOverlap"]
        overlap_table = FloorPlanChecker.inspect_overlaps(gdf_local, overlap_only)
        if not overlap_table.empty:
            st.subheader("Overlap pairs (floor blocks)")
            st.dataframe(overlap_table, use_container_width=True)
        return report
    return pd.DataFrame()
if uploaded_file is not None:
    try:
        gdf = load_geodata(uploaded_file, file_format)
        st.caption(f"geom types loaded: {sorted(set(gdf.geometry.geom_type))}")
        st.success("File loaded successfully!")
        try:
            _chk = gdf.copy()
            if _chk.crs is None:
                st.info("Hint: The current data lacks a CRS. If it is CSV with (lon,lat) representing latitude and longitude, retain ‘EPSG:4326’ in the sidebar. If it is a metric projection, enter the corresponding EPSG code (e.g., GDA2020/MGA55=EPSG:7855).")
            else:
                gtmp = _chk.to_crs(epsg=4326)
                lat_ok = gtmp.geometry.centroid.y.between(-90, 90).mean() > 0.95
                lon_ok = gtmp.geometry.centroid.x.between(-180, 180).mean() > 0.95
                if not (lat_ok and lon_ok):
                    st.warning("The coordinates appear to be outside the latitude/longitude range. If this is a CSV file: Please verify that the CRS in the sidebar is correctly configured (currently being converted to EPSG:4326 for preview).")
        except Exception:
            pass
        try:
            import numpy as np
            from shapely.ops import unary_union
            from shapely.geometry.base import BaseGeometry
            def _fix_geom(g):
                if g is None:
                    return None
                try:
                    if hasattr(g, "is_empty") and g.is_empty:
                        return None
                    try:
                        if hasattr(g, "is_valid") and not g.is_valid:
                            g = g.buffer(0)
                    except Exception:
                        pass
                    if getattr(g, "geom_type", "") == "GeometryCollection":
                        polys = [x for x in g.geoms if x.geom_type in ("Polygon", "MultiPolygon")]
                        if polys:
                            try:
                                return unary_union(polys)
                            except Exception:
                                return polys[0]
                        return g
                    return g
                except Exception:
                    return None
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.map(_fix_geom)
            gdf = gdf[~gdf.geometry.isna()].copy()
            try:
                gdf = gdf.explode(index_parts=False, ignore_index=True)  
            except TypeError:
                gdf = gdf.explode(index_parts=False).reset_index(drop=True)  
        except Exception:
            pass
        preview = gdf.copy()
        try:
            if preview.crs is not None:
                preview = preview.to_crs(epsg=4326)
        except Exception:
            pass
        geom = preview.geometry
        is_point = geom.geom_type == "Point"
        lat = pd.Series(index=geom.index, dtype="float64")
        lon = pd.Series(index=geom.index, dtype="float64")
        if is_point.any():
            lat.loc[is_point] = geom[is_point].y
            lon.loc[is_point] = geom[is_point].x
        if (~is_point).any():
            cent = geom[~is_point].centroid
            lat.loc[~is_point] = cent.y
            lon.loc[~is_point] = cent.x
        preview["latitude"] = pd.to_numeric(lat, errors="coerce")
        preview["longitude"] = pd.to_numeric(lon, errors="coerce")
        preview = preview.dropna(subset=["latitude", "longitude"]).copy()
        st.slider(
            "Preview zoom",
            5, 15,
            value=st.session_state.get("preview_zoom", 8),
            key="preview_zoom"
        )
        def current_point_radius() -> int:
            z = float(st.session_state.get("preview_zoom", 8))
            return int(max(10, 250 * (2 ** (8 - z))))
        def current_line_alpha() -> int:
            return int(st.session_state.get("line_alpha", 127))  
        def current_line_width() -> float:
            return float(st.session_state.get("line_width", 1.0)) 
        PM_AUTO  = "Auto (includes points, lines, and surfaces)"
        PM_LNPL  = "Line / surface"
        PM_POINT = "Point"
        preview_mode = st.radio(
            "Preview",
            [PM_AUTO, PM_LNPL, PM_POINT],
            horizontal=True,
            index=0,
            key="preview_mode",
        )
        mask_points = preview.geometry.geom_type.isin(["Point", "MultiPoint"])
        mask_lnpl   = preview.geometry.geom_type.isin(
            ["LineString", "MultiLineString", "Polygon", "MultiPolygon"]
        )
        pts_df  = preview.loc[mask_points, ["longitude", "latitude"]].copy()
        lnpl_gdf = preview.loc[mask_lnpl].copy()
        if lnpl_gdf.empty and not pts_df.empty and dtype in {"blocks","suburb","building","floor"}:
            st.warning(
                "The current data is identified as a point feature, but the selected dataset type requires a polygon."
            )
            try:
                st.session_state["preview_mode"] = "Point"
            except Exception:
                pass
        def greedy_color_polygons(gdf_polys):
            from shapely.strtree import STRtree
            import numpy as np
            if gdf_polys.empty:
                return []
            gdf = gdf_polys.reset_index(drop=True).copy()
            geoms = gdf.geometry.values
            mask_poly = gdf.geometry.geom_type.isin(["Polygon","MultiPolygon"])
            idxs = np.where(mask_poly.values)[0].tolist()
            if not idxs:
                return [0] * len(gdf)  
            tree = STRtree(geoms[idxs])
            adj = {i:set() for i in idxs}
            for pos, i in enumerate(idxs):
                g = geoms[i]
                for j_local in tree.query(g):
                    j = idxs[j_local]
                    if j == i:
                        continue
                    if g.touches(geoms[j]) and g.boundary.intersection(geoms[j].boundary).length > 0:
                        adj[i].add(j)
                        adj[j].add(i)
            colors = [-1] * len(gdf)
            for i in idxs:
                used = {colors[n] for n in adj[i] if colors[n] != -1}
                c = 0
                while c in used:
                    c += 1
                colors[i] = c
            for k in range(len(gdf)):
                if k not in idxs:
                    colors[k] = 0
            return colors
        def _view_state_from_coords(df_xy: pd.DataFrame) -> pdk.ViewState:
            try:
                from pydeck.data_utils import viewport
                vs = viewport.compute_view(
                    df_xy.rename(columns={"longitude": "longitude", "latitude": "latitude"})
                )
                z = float(st.session_state.get("preview_zoom", vs["zoom"]))
                return pdk.ViewState(
                    latitude=float(vs["latitude"]),
                    longitude=float(vs["longitude"]),
                    zoom=z,
                )
            except Exception:
                lat_c = float(df_xy["latitude"].median())
                lon_c = float(df_xy["longitude"].median())
                lat_span = float(df_xy["latitude"].max() - df_xy["latitude"].min())
                lon_span = float(df_xy["longitude"].max() - df_xy["longitude"].min())
                import math
                auto_z = max(2.0, min(16.0, math.log2(360.0 / max(lat_span, lon_span, 0.001))))
                z = float(st.session_state.get("preview_zoom", auto_z))
                return pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=z)
        layers = []
        def _add_points_layer(df_points: pd.DataFrame):
            if df_points.empty:
                st.info("There are no points in this dataset that can be plotted.")
                return False
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_points,
                get_position='[longitude, latitude]',
                get_radius=current_point_radius(),
                get_fill_color=[255, 99, 132, 170],
                pickable=False,
            )
            layers.append(layer)
            st.caption(f"Preview points drawn: {len(df_points)}")
            return True
        def _add_lnpl_layer(gdf_lnpl: gpd.GeoDataFrame):
            if gdf_lnpl.empty:
                st.info("There are no lines and surfaces in this dataset that can be plotted.")
                return False
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                max_preview = st.slider(
                    "Max features for geometry preview (for huge datasets)",
                    min_value=500, max_value=200000, value=200000, step=500,
                    help="Applies only when drawing lines/surfaces",
                    key="max_features_preview",
                )
            with row1_col2:
                fill_alpha = st.slider("Fill Opacity (0-255)", 0, 255, 127, key="fill_alpha") 
            gdf_src = gdf_lnpl.iloc[:max_preview].copy()
            gdf_col = gdf_src.copy()
            try:
                colors_idx = greedy_color_polygons(gdf_col)
            except Exception:
                colors_idx = [0] * len(gdf_col)
            palette = [
                [250,   0, 255, fill_alpha],  # neon magenta
                [200,  80, 255, fill_alpha],  # electric violet
                [140, 120, 255, fill_alpha],  # periwinkle
                [ 80, 140, 255, fill_alpha],  # azure / cornflower
                [  0, 160, 255, fill_alpha],  # dodger blue
                [  0, 196, 255, fill_alpha],  # cyan
                [  0, 230, 230, fill_alpha],  # aqua
                [  0, 210, 180, fill_alpha],  # teal-mint
                [  0, 255, 160, fill_alpha],  # neon mint
                [255, 140, 230, fill_alpha],  # pink highlight
            ]
            gdf_col = gdf_col.reset_index(drop=True)
            gdf_col["__fill__"] = [palette[c % len(palette)] for c in colors_idx]
            for c in gdf_col.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(gdf_col[c]):
                        gdf_col[c] = gdf_col[c].astype(str)
                except Exception:
                    pass
            import json
            gj = json.loads(gdf_col.to_json())
            for feat, rgba in zip(gj["features"], gdf_col["__fill__"].tolist()):
                feat.setdefault("properties", {})["fill"] = rgba
            layer = pdk.Layer(
                "GeoJsonLayer",
                data=gj,
                pickable=True,
                stroked=True,
                filled=True,
                opacity=1.0,
                get_line_width=current_line_width(),
                line_width_min_pixels=max(1, int(current_line_width())),
                get_line_color=[180, 220, 255, current_line_alpha()],
                get_fill_color="properties.fill",
            )
            layers.append(layer)
            st.session_state["lnpl_drawn"] = len(gdf_src)
            st.session_state["pts_skipped"] = max(0, len(preview) - len(gdf_src))
            return True
        if preview_mode == PM_POINT:
            drew = _add_points_layer(pts_df)
        elif preview_mode == PM_LNPL:
            drew = _add_lnpl_layer(lnpl_gdf)
        else:  
            if not lnpl_gdf.empty:
                drew = _add_lnpl_layer(lnpl_gdf)
            else:
                drew = _add_points_layer(pts_df)
        st.markdown(
            f"""
            <div style='display:flex; gap:18px; align-items:center; font-size:0.95rem;'>
            <span>Preview geometries drawn: <code>{st.session_state.get('lnpl_drawn', len(lnpl_gdf))}</code>
                <span style='opacity:.75'>(points skipped: {st.session_state.get('pts_skipped', max(0, len(preview)-len(lnpl_gdf)))})</span>
            </span>
            <span>points = <code>{len(pts_df)}</code></span>
            <span>lines/polys = <code>{len(lnpl_gdf)}</code></span>
            <span>layers drawn = <code>{len(layers)}</code></span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if not layers:
            st.warning("No preview layers were drawn.")
        if (preview_mode == PM_POINT) or (preview_mode == PM_AUTO and lnpl_gdf.empty):
            view_df = pts_df[["longitude", "latitude"]] if not pts_df.empty else preview[["longitude", "latitude"]]
        else:
            use_df = lnpl_gdf if not lnpl_gdf.empty else preview
            view_df = use_df[["longitude", "latitude"]]

        if view_df.empty:
            view_df = preview[["longitude", "latitude"]]
        view_state_preview = _view_state_from_coords(view_df)
        st.session_state["preview_center"] = {
            "lat": float(view_state_preview.latitude),
            "lon": float(view_state_preview.longitude),
            "zoom": float(view_state_preview.zoom),
        }
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state_preview,
                map_provider="carto",
                map_style="dark",
                tooltip={"text": "type: {type}\n(rule: {rule_key})"}
            )
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.slider("Border Opacity (0-255)", 0, 255,
                    int(st.session_state.get("line_alpha", 127)),
                    key="line_alpha")
        with col_b:
            st.slider("Border width (px)", 0.5, 5.0,
                    float(st.session_state.get("line_width", 1.0)),
                    0.5, key="line_width")
        if st.button("Run Validation"):
            result_df = run_by_dtype(gdf, dtype)
            if result_df is None:
                st.error("Runner returned no result.")
                st.stop()
            df = result_df.copy()  
            st.caption(
                f"debug -> dtype={dtype}, source rows={len(gdf)}, "
                f"geom types={sorted(set(gdf.geometry.geom_type))}, "
                f"violations={len(df)}"
            )
            if not df.empty and "error_type" in df.columns:
                st.write("debug: counts by error_type")
                st.write(df["error_type"].value_counts())
            empty = pd.Series([""] * len(df), index=df.index)
            err = df["error_type"].astype(str) if "error_type" in df.columns else empty
            msg = df["message"].astype(str)    if "message"    in df.columns else empty
            df["output_error"] = err.str.cat(msg, sep=": ").str.strip(": ").fillna("")
            if "error_count" not in df.columns:
                df["error_count"] = 1
            if "rule_key" not in df.columns:
                df["rule_key"] = df["error_type"] if "error_type" in df.columns else "unknown"
            df_all = df.copy()  
            AUTO_TOKENS = {"auto", "Auto", "Auto (based on dataset)", "Auto (based on dataset type)"}
            is_auto = (not selected_rule_keys) or bool(set(selected_rule_keys) & AUTO_TOKENS)
            if selected_rule_keys and not is_auto:
                df = df[df["rule_key"].isin(selected_rule_keys)]
            total_all = len(df_all)
            total_sel = len(df)
            st.subheader("Validation Report")
            c1, c2, c3 = st.columns(3)
            c1.metric("All issues (all rules)", total_all)
            c2.metric("Issues (selected rules)", total_sel)
            c3.metric("Triggered rules", int(df["error_type"].nunique()) if total_sel else 0)
            if total_sel == 0:
                if is_auto:
                    st.markdown("""
                    <style>
                    .ok-banner{background:linear-gradient(90deg,#14532d,#166534);
                    border:1px solid #16a34a55;color:#dcfce7;padding:16px 18px;border-radius:14px;
                    font-weight:700;box-shadow:0 6px 18px rgba(16,185,129,.25);font-size:1.05rem}
                    .ok-sub{opacity:.85;font-weight:500;margin-top:6px;font-size:.95rem}
                    </style>
                    <div class="ok-banner">✅ No validation issues found.</div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ No issues for the selected rules.", icon="✅")
                    if total_all > 0 and "error_type" in df_all.columns:
                        st.caption("Other rules with issues (top):")
                        st.dataframe(df_all["error_type"].value_counts().head(5).rename_axis("error_type").reset_index(name="count"))
                st.stop()
            src = gdf.copy()
            try:
                if src.crs is not None:
                    src = src.to_crs(epsg=4326)   
            except Exception:
                pass
            if "geometry" not in df.columns and "index" in df.columns and "geometry" in src.columns:
                df["geometry"] = df["index"].map(src.geometry)
            if "geometry" in df.columns:
                try:
                    if "lon" not in df.columns:
                        df["lon"] = df["geometry"].centroid.x
                    if "lat" not in df.columns:
                        df["lat"] = df["geometry"].centroid.y
                except Exception:
                    pass
            import numpy as np
            import pandas as pd
            def _is_all_noneish(s: pd.Series) -> bool:
                if s.empty:
                    return True
                m = s.isna()
                m = m | (s.astype(object) == "")
                m = m | s.astype(str).str.strip().str.lower().isin({"none", "nan", "nat"})
                return bool(m.all())
            PIN_FIRST = [c for c in [
                "index", "error_type", "message", "output_error", "rule_key",
                "error_count", "lon", "lat"
            ] if c in df.columns]
            others = [c for c in df.columns if c not in PIN_FIRST]
            noneish_cols = [c for c in others if _is_all_noneish(df[c])]
            not_none_cols = [c for c in others if c not in noneish_cols]
            df = df[PIN_FIRST + not_none_cols + noneish_cols]
            st.dataframe(df, use_container_width=True)
            tmp = gdf.copy()
            try:
                if tmp.crs is not None:
                    tmp = tmp.to_crs(epsg=4326)
            except Exception:
                pass
            tmp["lat"] = tmp.geometry.centroid.y
            tmp["lon"] = tmp.geometry.centroid.x
            clean_df = df.copy()
            if "index" in clean_df.columns:
                clean_df = clean_df.set_index("index", drop=False)
            if "geometry" not in clean_df.columns and "geometry" in gdf.columns:
                clean_df["geometry"] = gdf.geometry.reindex(clean_df.index)
            for c in ("lat", "lon"):
                if c not in clean_df.columns:
                    clean_df[c] = tmp[c].reindex(clean_df.index)
            if "geometry" in clean_df.columns:
                try:
                    cent_lat = clean_df["geometry"].apply(lambda g: float(g.centroid.y) if g is not None else None)
                    cent_lon = clean_df["geometry"].apply(lambda g: float(g.centroid.x) if g is not None else None)
                    clean_df["lat"] = clean_df["lat"].fillna(cent_lat)
                    clean_df["lon"] = clean_df["lon"].fillna(cent_lon)
                except Exception:
                    pass
            try:
                ctr = tmp.geometry.unary_union.centroid
                clean_df["lat"] = clean_df["lat"].fillna(float(ctr.y))
                clean_df["lon"] = clean_df["lon"].fillna(float(ctr.x))
            except Exception:
                pc = st.session_state.get("preview_center", {"lat": -37.8136, "lon": 144.9631})
                clean_df["lat"] = clean_df["lat"].fillna(float(pc["lat"]))
                clean_df["lon"] = clean_df["lon"].fillna(float(pc["lon"]))
            def make_hash(row) -> str:
                parts = [
                    dtype,
                    str(row.get("index", "")),
                    str(row.get("error_type", "")),
                    str(row.get("message", "")),
                ]
                try:
                    geom = row.get("geometry", None)
                    if geom is not None:
                        parts.append(geom.wkb_hex)  
                except Exception:
                    pass
                return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

            clean_df["feature_hash"] = clean_df.apply(make_hash, axis=1)
            clean_df["rule_keys"] = clean_df.get("rule_key", "")
            conn_planb = sqlite3.connect(db_name)
            ensure_schema_planB(conn_planb)  
            file_bytes = uploaded_file.getvalue() if uploaded_file is not None else b""
            dataset_id = upsert_dataset_row(
                conn_planb,
                dtype=dtype,
                fmt=file_format,
                file_name=(uploaded_file.name if uploaded_file else ""),
                file_bytes=file_bytes,
                row_count=len(gdf),
            )
            run_id = begin_run(conn_planb, dataset_id, triggered_by="user")
            _ = write_violations(conn_planb, run_id, df, dtype)
            finish_run(conn_planb, run_id, status="succeeded")
            conn_planb.close()
            conn = sqlite3.connect(db_name)
            ensure_schema(conn)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prev_invalid = {
                x[0] for x in conn.execute(
                    "SELECT feature_hash FROM validations WHERE dataset_type=? AND status='invalid'", (dtype,)
                ).fetchall()
            }
            current = set(clean_df["feature_hash"])
            upsert_sql = """
            INSERT INTO validations(dataset_type, feature_hash, status, error_count, output_error, rule_keys, validated_at, lat, lon)
            VALUES (?, ?, 'invalid', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(feature_hash, dataset_type) DO UPDATE SET
                status='invalid',
                error_count=excluded.error_count,
                output_error=excluded.output_error,
                rule_keys=excluded.rule_keys,
                validated_at=excluded.validated_at,
                resolved_at=NULL,
                lat=excluded.lat,
                lon=excluded.lon
            """
            for _, r in clean_df.iterrows():
                conn.execute(upsert_sql, (
                    dtype, r["feature_hash"],
                    int(r.get("error_count", 1)),
                    str(r.get("output_error", "")),
                    str(r.get("rule_keys", "")),
                    now, float(r.get("lat", 0) or 0), float(r.get("lon", 0) or 0),
                ))
            to_resolve = list(prev_invalid - current)
            if to_resolve:
                conn.executemany(
                    "UPDATE validations SET status='resolved', resolved_at=? WHERE dataset_type=? AND feature_hash=?",
                    [(now, dtype, fh) for fh in to_resolve]
                )
            conn.commit()
            conn.close()
            if dtype == "road":
                df_fail = df_all  
                key_col = next((c for c in ["index","idx","row_index","feature_index"] if c in df_fail.columns), None)
                if key_col is None:
                    df_fail = df_fail.reset_index().rename(columns={"index":"__row_index"})
                    key_col = "__row_index"
                seg_mask = df_fail["error_type"].astype(str).str.contains(
                    r"(IsolatedRoad|IsolatedSegment)", case=False, na=False
                )
                seg_idx = (
                    pd.to_numeric(df_fail.loc[seg_mask, key_col], errors="coerce")
                    .dropna().astype(int).unique()
                )
                gdf_pos = gdf.reset_index().rename(columns={"index":"__row_index"})
                road_hits = gdf_pos[gdf_pos["__row_index"].isin(seg_idx)].copy()
                st.write("matched road segments:", len(road_hits))
                if road_hits.empty and ("geometry" in df_fail.columns) and (~df_fail["geometry"].isna()).any():
                    road_hits = gpd.GeoDataFrame(
                        df_fail.loc[seg_mask & df_fail["geometry"].notna()].copy(),
                        geometry="geometry",
                        crs=getattr(gdf, "crs", None)
                    )
                if road_hits.empty:
                    st.warning("There are violation records, but no matching line segments were found (mostly endpoint-level issues).")
                else:
                    try:
                        if road_hits.crs and road_hits.crs.to_epsg() != 4326:
                            road_hits = road_hits.to_crs(epsg=4326)
                    except Exception:
                        pass
                    def _geom_to_paths(g):
                        if g.geom_type == "LineString":
                            return [[float(x), float(y)] for x, y in g.coords]
                        elif g.geom_type == "MultiLineString":
                            return [[[float(x), float(y)] for x, y in ls.coords] for ls in g.geoms]
                        return None
                    paths = []
                    for _, g in road_hits.geometry.items():
                        p = _geom_to_paths(g)
                        if p is None: 
                            continue
                        if isinstance(p[0][0], list):   
                            for one in p:
                                paths.append({"path": one})
                        else:                            
                            paths.append({"path": p})
                    if not paths:
                        st.warning("Segments have been matched, but a drawable path cannot be generated.")
                    else:
                        layer_lines = pdk.Layer(
                            "PathLayer",
                            data=paths,
                            get_path="path",
                            get_width=3,
                            width_min_pixels=2,
                            get_color=[255, 80, 60, 200],
                            pickable=True,
                        )
                        ctr = road_hits.geometry.centroid
                        view_state = pdk.ViewState(
                            latitude=float(ctr.y.mean()),
                            longitude=float(ctr.x.mean()),
                            zoom=float(st.session_state.get("preview_zoom", 11)),
                        )
                        st.pydeck_chart(pdk.Deck(
                            layers=[layer_lines],
                            initial_view_state=view_state,
                            map_provider="carto",
                            map_style="dark",
                            tooltip={"text": "isolated road segment"},
                        ))
            else:
                with sqlite3.connect(db_name) as conn_read:
                    df_map = pd.read_sql_query(
                        "SELECT feature_hash, status, output_error, rule_keys, lat, lon FROM validations WHERE dataset_type=?",
                        conn_read, params=(dtype,)
                    )
                if not df_map.empty and df_map["lat"].notna().any():
                    time_cols = [c for c in ["updated_at", "validated_at", "resolved_at"] if c in df_map.columns]
                    if time_cols:
                        tcol = "updated_at" if "updated_at" in df_map.columns else time_cols[0]
                        df_map = (df_map.sort_values(tcol).drop_duplicates(subset=["feature_hash"], keep="last"))
                    df_plot = df_map.dropna(subset=["lat","lon"]).copy()
                    df_plot = df_plot[(df_plot["lat"].between(-90, 90)) & (df_plot["lon"].between(-180, 180))]
                    df_plot = df_plot[~((df_plot["lat"] == 0) & (df_plot["lon"] == 0))]
                    if df_plot.empty:
                        st.info("No plottable points in results (all coordinates were missing or out of range).")
                    else:
                        status_norm = df_plot["status"].astype(str).str.strip().str.lower()
                        df_resolved = df_plot[status_norm.eq("resolved")]
                        df_invalid  = df_plot[status_norm.eq("invalid")]
                        color_invalid  = [220, 60, 50, 200]   
                        color_resolved = [ 25,160,80, 200]   
                        layers = []
                        if not df_resolved.empty:
                            layers.append(pdk.Layer(
                                "ScatterplotLayer",
                                data=df_resolved,
                                get_position='[lon, lat]',
                                radius_units="pixels",
                                get_radius=10,            
                                radius_min_pixels=6,
                                radius_max_pixels=60,
                                get_fill_color=color_resolved,
                                pickable=True,
                            ))
                        if not df_invalid.empty:
                            layers.append(pdk.Layer(
                                "ScatterplotLayer",
                                data=df_invalid,
                                get_position='[lon, lat]',
                                radius_units="pixels",
                                get_radius=12,            
                                radius_min_pixels=8,
                                radius_max_pixels=80,
                                get_fill_color=color_invalid,
                                pickable=True,
                            ))
                        sub = df_invalid if not df_invalid.empty else df_plot
                        pref_zoom = st.session_state.get("preview_zoom", None)
                        try:
                            from pydeck.data_utils import viewport
                            vs = viewport.compute_view(
                                sub.rename(columns={"lon": "longitude", "lat": "latitude"})[["longitude", "latitude"]]
                            )
                            zval = float(pref_zoom) if pref_zoom is not None else float(vs["zoom"])
                            view_state = pdk.ViewState(latitude=float(vs["latitude"]), longitude=float(vs["longitude"]), zoom=zval)
                        except Exception:
                            lat_c = float(sub["lat"].median()); lon_c = float(sub["lon"].median())
                            lat_span = float(sub["lat"].max() - sub["lat"].min())
                            lon_span = float(sub["lon"].max() - sub["lon"].min())
                            import math
                            span = max(lat_span, lon_span, 0.001)
                            auto_z = max(2.0, min(16.0, math.log2(360.0 / span)))
                            zval = float(pref_zoom) if pref_zoom is not None else auto_z
                            view_state = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zval)
                        pc = st.session_state.get("preview_center")
                        if pc and dtype != "blocks":
                            view_state = pdk.ViewState(latitude=pc["lat"], longitude=pc["lon"], zoom=pc["zoom"])
                        chart_key = f"results_map_{round(view_state.latitude,5)}_{round(view_state.longitude,5)}_{int(view_state.zoom)}"
                        tooltip = {"text": "status: {status}\nrules: {rule_keys}\nerror: {output_error}"}
                        st.pydeck_chart(pdk.Deck(
                            layers=layers,
                            initial_view_state=view_state,
                            map_provider="carto",
                            map_style="dark",
                            tooltip=tooltip,
                        ), key=chart_key)
                        st.caption(f"debug plot counts → invalid: {len(df_invalid)}  |  resolved: {len(df_resolved)}")
            st.success(f"Saved. New/updated: {len(current)}; resolved this run: {len(to_resolve)}")
            with st.expander("Presentation mode: show ONE example for ONE rule"):
                all_rules = sorted(RULES.keys())
                if not all_rules:
                    st.info("No RULES available for demo.")
                else:
                    rule_for_demo = st.selectbox("Pick a rule", all_rules, index=0, key="demo_rule_select")
                    rule_col = (
                        df["rule_key"]
                        if "rule_key" in df.columns
                        else pd.Series([""] * len(df), index=df.index)
                    )
                    mask = rule_col.astype(str).str.contains(rule_for_demo, na=False)
                    demo = df.loc[mask].head(1)

                    if demo.empty:
                        st.info("No example for this rule in current run.")
                    else:
                        st.dataframe(demo[["output_error", "error_count", "rule_key"]])
            st.divider()
            st.subheader("Export")
            scope = st.radio(
                "Which data to export?",
                options=("Selected rules (shown above)", "All rules"),
                horizontal=True,
                key="export_scope",
            )
            export_df = df if scope.startswith("Selected") else df_all
            if export_df.empty:
                st.info("Nothing to export.")
            else:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                fname = f"validation_{dtype}_{ts}.csv"
                csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Download CSV Report",
                    data=csv_bytes,
                    file_name=fname,
                    mime="text/csv",
                    key="download_csv_bottom",
                )
                col1, col2 = st.columns([3,1])
                with col1:
                    save_dir = st.text_input("Save to folder on server", DB_DIR, key="save_dir_csv")
                with col2:
                    if st.button("Save CSV to server", key="save_csv_bottom"):
                        import os
                        os.makedirs(save_dir, exist_ok=True)
                        out_path = os.path.join(save_dir, fname)
                        with open(out_path, "wb") as f:
                            f.write(csv_bytes)
                        st.success(f"Saved to: {out_path}")
    except Exception as e:
        st.error(f"Failed to load or validate data: {e}")
else:
    st.info("Upload a dataset to begin.")
