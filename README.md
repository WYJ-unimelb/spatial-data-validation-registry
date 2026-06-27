# 🧭 Spatial Data Validation Demo

A lightweight **Streamlit + GeoPandas + Shapely + PyDeck + SQLite** demo app for **geospatial data quality validation**.  
It supports multiple input formats, applies rule-based validation depending on dataset type, stores results in a registry database, and visualizes issues interactively.

---

## 🚀 Features

- **Supported Dataset Types**
  - `suburb`, `blocks`, `road`, `building`, `floor`
  - Dynamically changes available validation functions and sliders by type.

- **Supported File Formats**
  - **GeoJSON / CSV / XML / KML**
  - CSV files auto-detect `(lon, lat)` or `WKT/GeoJSON` geometry columns.
  - XML is first flattened and then converted into geometries.

- **Rule Runner**
  - Encapsulated functions such as:
    - `run_suburb_checks()`, `run_road_checks()`, `run_building_checks()`, `run_blocks_checks()`, `FloorPlanChecker()`
  - All return standardized DataFrames with error types, messages, and optional metadata.

- **Database Registry (Plan B version)**
  - Stores dataset info, rules, validation runs, and violations.
  - Uses triggers and views to maintain the latest issue states.

- **UI & Visualization**
  - Minimal dark neon design.
  - Streamlit controls at the top; PyDeck interactive maps for preview and result layers.

---

## 📂 Directory Structure

```
.streamlit/          # Optional: Streamlit theme config
app.py               # Main UI (upload → rule selection → run → DB write → visualization)
rule_checker.py      # Validation logic for each dataset type
registry_db.py       # SQLite schema and data persistence
```

---

## 🧩 Environment Setup

### 1. Install dependencies

Requires **Python 3.10+**

```bash
pip install streamlit geopandas shapely pydeck pandas sqlite-utils lxml fiona pyproj
```

### 2. Optional: Theme configuration
You can customize `.streamlit/config.toml` for dark/light themes, or edit the embedded CSS inside `app.py`.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

### Steps:

1. Select **Dataset Type** and **File Format**.
2. Upload your file.
3. (Optional) Select specific validation functions, or keep “Auto” to run all.
4. Adjust thresholds (e.g., min area, short edge, hole ratio).
5. Click **Run Validation**.

Results will:
- Be displayed on the map (colored by issue type).
- Be stored in a SQLite registry named `validation_registry_{dtype}.db`.

---

## 🧠 Validation Rules Overview

| Dataset Type | Example Checks |
|---------------|----------------|
| **Suburb** | Geometry validity, overlaps, thin polygons, zero area, gaps, small holes, disconnected parts, duplicated shapes, centroid outside, name similarity, postcode consistency |
| **Road** | Isolated endpoints, disconnected segments |
| **Building** | Empty/invalid geometry, overlap ratio, small area, long thin shapes, inner holes, invalid CRS |
| **Blocks** | Invalid geometry, overlaps, short edges, angular distortion, precision issues |
| **Floor** | FloorPlanChecker summarizing geometry layout & overlap consistency |

Each rule returns:
```
index | rule_name | message | geometry | severity
```

---

## 🗄️ Database Schema (Plan-B Summary)

- **datasets**  
  Track file metadata (MD5 hash, type, format, number of features).

- **rules**  
  Key → human-readable rule names per dataset type.

- **validation_run**  
  Each validation execution, with start/end timestamps and status.

- **violations**  
  Stores rule violations (unique by run_id + rule_id + feature hash).

- **violation_status**  
  Keeps issue resolution state (`invalid` / `waived`).  
  Triggers auto-insert first status; `v_violation_latest` view shows the newest one.

---

## ⚙️ Developer Guide

### Extend Validation Rules

1. Add new rule functions in `rule_checker.py` that return a `List[Failure]`.
2. Append the function to the appropriate `run_*_checks()` aggregation.
3. Register the rule name and dataset type in `app.py → build_rule_name_map()`.

### Database Operations

- `ensure_schema_planB()` — initialize tables if missing  
- `upsert_dataset_row()` — ensure dataset uniqueness by hash  
- `begin_run()` / `write_violations()` / `finish_run()` — manage full write cycle

---

## 🧾 FAQ

**Q: My CSV has no geometry column.**  
A: Ensure columns include `(lon, lat)` or a `geometry/WKT` field.

**Q: Area or length seems incorrect.**  
A: Ensure CRS is projected (e.g., EPSG:3111). The app can auto-reproject for metric accuracy.

**Q: Where is my output stored?**  
A: SQLite DB file is created as `validation_registry_{dtype}.db` in the same directory.  
   Override with environment variable `VALIDATION_DB_DIR`.

---

## 🧑‍💻 Example

Example console output:
```
Dataset type: building
File loaded: 1250 geometries
Applied 7 rules, 42 violations found
Results written to validation_registry_building.db
```

---

## 🪪 License

For educational and demonstration purposes only.  
Please include attribution when reusing code or database schema design.

---

**Author:** Group 19 — Spatial Data Validation Project  
**Frameworks:** Streamlit · GeoPandas · Shapely · PyDeck · SQLite  
**Last Updated:** 2025-10-13
