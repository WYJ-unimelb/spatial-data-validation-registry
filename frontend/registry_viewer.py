import sqlite3
import pandas as pd
import streamlit as st
import pydeck as pdk
from pathlib import Path
import sys, os

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
UTILS_DIR = PROJECT_ROOT / "utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from registry_db import ensure_schema_planB

st.set_page_config(page_title="Validation Registry Viewer", layout="centered")
st.title("Validation Registry Viewer")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    dtype = st.selectbox("Dataset type", ["suburb", "blocks", "road", "building", "floor"], index=0)
with c2:
    DB_DIR = os.environ.get("VALIDATION_DB_DIR", str(PROJECT_ROOT))
    db_name = os.path.join(DB_DIR, f"validation_registry_{dtype}.db")
    st.caption(f"DB: {db_name}")
with c3:
    status_filter = st.multiselect("Status filter", ["invalid", "waived"], default=["invalid"])

conn = sqlite3.connect(db_name)
conn.execute("PRAGMA foreign_keys = ON")

conn = sqlite3.connect(db_name)
ensure_schema_planB(conn)  

ds_df = pd.read_sql_query(
    "SELECT dataset_id, name, dataset_type, created_at FROM datasets WHERE dataset_type=? ORDER BY created_at DESC",
    conn, params=(dtype,)
)
if ds_df.empty:
    st.info("No datasets found for this type yet. Run the main app first.")
    st.stop()

ds_options = {f"{r.name}  (#{int(r.dataset_id)})": int(r.dataset_id) for _, r in ds_df.iterrows()}
ds_label = st.selectbox("Dataset", list(ds_options.keys()))
dataset_id = ds_options[ds_label]

runs = pd.read_sql_query("""
  SELECT run_id, status, started_at, completed_at
  FROM validation_run
  WHERE dataset_id=?
  ORDER BY started_at DESC
""", conn, params=(dataset_id,))
if runs.empty:
    st.info("No runs yet for this dataset.")
    st.stop()

labels = [f"{r.run_id} [{r.status}] @ {r.started_at}" for r in runs.itertuples()]
default_idx = 0
for i, row in enumerate(runs.itertuples()):
    if row.status == "succeeded":
        default_idx = i
        break
run_label = st.selectbox("Run", labels, index=default_idx)
run_id = run_label.split()[0]

q = """
SELECT violation_id, rule_id, feature_hash, error_type, message, error_count,
       latitude AS lat, longitude AS lon, status
FROM v_violation_latest
WHERE run_id=?
"""
df = pd.read_sql_query(q, conn, params=(run_id,))
conn.close()

if df.empty:
    st.success("No violations in this run 🎉")
    st.stop()

df = df[df["status"].isin(status_filter)]

st.subheader("Summary")
st.write(df.groupby(["status"]).size().rename("count"))
st.write(df.groupby(["error_type", "status"]).size().rename("count").reset_index().sort_values("count", ascending=False))

st.subheader("Violations (latest state)")
st.dataframe(df[["violation_id","status","error_type","message","lat","lon"]], use_container_width=True)

if df["lat"].notna().any():
    df_map = df.dropna(subset=["lat","lon"]).copy()
    color_map = {"invalid": [220,60,50,180], "waived": [140,140,140,180]}
    default_color = [120,120,120,140]
    df_map["color"] = df_map["status"].map(color_map).apply(
        lambda v: v if isinstance(v, (list, tuple)) else default_color)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[lon, lat]',
        get_radius=40,
        get_fill_color='color',
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=float(df_map["lat"].mean()), longitude=float(df_map["lon"].mean()), zoom=9,
    )
    tooltip = {"text": "status: {status}\\n{error_type}: {message}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_provider="carto", map_style="dark", tooltip=tooltip))
else:
    st.info("No valid lat/lon to plot.")
