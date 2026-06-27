from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import warnings
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, base
from shapely.strtree import STRtree
from shapely.validation import explain_validity
import numpy as np


CRSType = Union[int, str]

# -------------------------
# Failure record & Rule base
# -------------------------

@dataclass
class Failure:
    index: Any
    error_type: str
    message: str
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"index": self.index, "error_type": self.error_type, "message": self.message}
        if self.extras:
            d.update(self.extras)
        return d


class Rule:
    """Base class for all validation rules."""
    name: str = "UnnamedRule"
    applies_to: Set[str] = frozenset({"common"})
    description: str = ""
    category: str = "general"

    def __init__(self, **params):
        self.params = params

    def check(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        failures = self._check(gdf)
        if not failures:
            return pd.DataFrame(columns=["index", "error_type", "message"])
        return pd.DataFrame([f.to_dict() for f in failures])

    # Must be implemented by subclasses; return a list[Failure]
    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, applies_to={self.applies_to}, params={self.params})"


# -------------------------
# Utilities
# -------------------------

def _ensure_geometry(gdf: gpd.GeoDataFrame) -> None:
    if "geometry" not in gdf:
        raise ValueError("GeoDataFrame must contain a 'geometry' column.")

def _to_metric(
        gdf: gpd.GeoDataFrame,
        target_epsg: Optional[int] = None,
        fallback_epsg: int = 3857
) -> gpd.GeoDataFrame:
    """
    Project to a metric CRS for area/distance calculations.
    If target_epsg is None, use fallback_epsg (3857).
    Always returns a GeoDataFrame (never None).
    """
    if gdf.crs is None:
        warnings.warn("GeoDataFrame has no CRS; metrics will be unitless (no reprojection).")
        return gdf
    epsg = target_epsg if target_epsg is not None else fallback_epsg
    try:
        return gdf.to_crs(epsg=epsg)
    except Exception as e:
        warnings.warn(f"Reprojection to EPSG:{epsg} failed ({e}); using original CRS.")
        return gdf

def _allowed_crs_to_str_set(allowed: Iterable[CRSType]) -> Set[str]:
    out = set()
    for a in allowed:
        if isinstance(a, int):
            out.add(f"EPSG:{a}")
        else:
            s = str(a).upper().replace("EPSG:", "").strip()
            out.add(f"EPSG:{s}")
    return out

def _geom_iter_polygon_parts(geom: base.BaseGeometry) -> Iterable[Polygon]:
    """Yield Polygon parts for both Polygon and MultiPolygon."""
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            yield p


# -------------------------
# Common rules
# -------------------------

class GeometryNotEmpty(Rule):
    category = "geometry"
    name = "GeometryNotEmpty"
    applies_to = frozenset({"common"})
    description = "Geometry must not be empty or NA."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        _ensure_geometry(gdf)
        mask = gdf.geometry.isna() | gdf.geometry.is_empty
        return [Failure(idx, self.name, "Empty geometry") for idx in gdf.index[mask]]


class GeometryValid(Rule):
    category = "geometry"
    name = "GeometryValid"
    applies_to = frozenset({"common"})
    description = "Geometry must be valid; provides Shapely explain_validity message."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        _ensure_geometry(gdf)
        mask = ~gdf.geometry.is_valid
        failures: List[Failure] = []
        if mask.any():
            for idx, geom in gdf.loc[mask, "geometry"].items():
                failures.append(Failure(idx, self.name, explain_validity(geom)))
        return failures


class ExpectedCRS(Rule):
    category = "crs"
    name = "ExpectedCRS"
    applies_to = frozenset({"common"})
    description = "CRS must be one of the allowed EPSG codes."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        allowed: Iterable[CRSType] = self.params.get(
            "allowed",
            (4326, 7844, "EPSG:4326", "EPSG:7844")  # WGS84 + GDA2020 (commonly seen in AU)
        )
        if gdf.crs is None:
            return [Failure(None, self.name, "CRS is missing (None).")]
        allowed_str = _allowed_crs_to_str_set(allowed)
        current = str(gdf.crs).upper().replace("EPSG:", "").strip()
        current = f"EPSG:{current}"
        if current not in allowed_str:
            return [Failure(None, self.name, f"CRS {current} not in allowed set {sorted(allowed_str)}")]
        return []


class AttributeNotNull(Rule):
    category = "attribute"
    name = "AttributeNotNull"
    applies_to = frozenset({"common"})
    description = "Selected attributes must not be null. Can skip missing fields."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        fields: Sequence[str] = self.params.get("fields", [])
        skip_missing: bool = bool(self.params.get("skip_missing", True))
        failures: List[Failure] = []
        for col in fields:
            if col not in gdf.columns:
                if skip_missing:
                    continue
                failures.append(Failure(None, self.name, f"Field '{col}' not found"))
                continue
            mask = gdf[col].isna()
            failures.extend(
                Failure(idx, self.name, f"Null in field '{col}'", {"field": col}) for idx in gdf.index[mask]
            )
        return failures


class AttributeInDomain(Rule):
    category = "attribute"
    name = "AttributeInDomain"
    applies_to = frozenset({"common"})
    description = "Attribute values must be in a given domain."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        field: str = self.params["field"]
        domain: Set[Any] = set(self.params.get("domain", []))
        skip_missing: bool = bool(self.params.get("skip_missing", True))
        failures: List[Failure] = []
        if field not in gdf.columns:
            if skip_missing:
                return []
            return [Failure(None, self.name, f"Field '{field}' not found")]
        mask = ~gdf[field].isin(domain)
        for idx, val in gdf.loc[mask, field].items():
            failures.append(Failure(idx, self.name, f"Value '{val}' not in domain", {"field": field}))
        return failures


class UniqueField(Rule):
    category = "attribute"
    name = "UniqueField"
    applies_to = frozenset({"common"})
    description = "Attribute must be unique among features."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        field: str = self.params["field"]
        skip_missing: bool = bool(self.params.get("skip_missing", True))
        if field not in gdf.columns:
            if skip_missing:
                return []
            return [Failure(None, self.name, f"Field '{field}' not found")]
        counts = gdf[field].value_counts(dropna=False)
        dup_vals = counts[counts > 1].index
        failures: List[Failure] = []
        if len(dup_vals) == 0:
            return failures
        mask = gdf[field].isin(dup_vals)
        for idx, val in gdf.loc[mask, field].items():
            failures.append(Failure(idx, self.name, f"Duplicate value '{val}'", {"field": field}))
        return failures


class DuplicateGeometries(Rule):
    category = "topology"
    name = "DuplicateGeometries"
    applies_to = frozenset({"common"})
    description = "Flag identical geometries (by WKB)."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        wkb = gdf.geometry.to_wkb()  # bytes
        counts = pd.Series(wkb).value_counts(dropna=False)
        dup_wkbs = counts[counts > 1].index
        if len(dup_wkbs) == 0:
            return []
        mask = pd.Series(wkb).isin(dup_wkbs).values
        failures: List[Failure] = []
        for idx, geom in gdf.loc[mask, "geometry"].items():
            failures.append(Failure(idx, self.name, "Duplicate geometry"))
        return failures


class WithinBoundingPolygon(Rule):
    category = "spatial"
    name = "WithinBoundingPolygon"
    applies_to = frozenset({"common"})
    description = "Feature centroid must lie within the provided boundary polygon."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        boundary: Union[Polygon, MultiPolygon] = self.params["polygon"]
        mask = ~gdf.geometry.centroid.within(boundary)
        return [Failure(idx, self.name, "Feature centroid outside boundary") for idx in gdf.index[mask]]


# -------------------------
# Building-specific rules
# -------------------------

class BuildingPolygonsOnly(Rule):
    category = "geometry"
    name = "BuildingPolygonsOnly"
    applies_to = frozenset({"building"})
    description = "Buildings must be Polygon or MultiPolygon."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        mask = ~gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        return [Failure(idx, self.name, "Geometry must be Polygon/MultiPolygon") for idx in gdf.index[mask]]


class MinimumArea(Rule):
    category = "metric"
    name = "MinimumArea"
    applies_to = frozenset({"building"})
    description = "Polygon area must be >= min_area (m^2) in metric CRS."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        min_area: float = float(self.params.get("min_area", 1.0))
        target_epsg: Optional[int] = self.params.get("target_epsg", 7855)  # GDA2020 / MGA Zone 55 (Melbourne)
        gdf_m = _to_metric(gdf, target_epsg=target_epsg)
        areas = gdf_m.geometry.area
        mask = areas < min_area
        return [
            Failure(idx, self.name, f"Area {areas.loc[idx]:.3f} < {min_area}",
                    {"area_m2": float(areas.loc[idx])})
            for idx in gdf.index[mask]
        ]


class HolesTotalAreaLimit(Rule):
    category = "topology"
    name = "HolesTotalAreaLimit"
    applies_to = frozenset({"building"})
    description = "Sum of interior ring (hole) areas per feature must be <= max_total_hole_area (m^2)."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        max_total: float = float(self.params.get("max_total_hole_area", 2.0))
        target_epsg: Optional[int] = self.params.get("target_epsg", 7855)
        gdf_m = _to_metric(gdf, target_epsg=target_epsg)

        failures: List[Failure] = []
        for idx, geom in gdf_m.geometry.items():
            if geom is None or geom.is_empty:
                continue
            total_holes = 0.0
            for poly in _geom_iter_polygon_parts(geom):
                # Each interior ring is a LinearRing convertible to Polygon for area
                for coords in poly.interiors:
                    hole_poly = Polygon(coords)
                    total_holes += hole_poly.area
            if total_holes > max_total:
                failures.append(Failure(idx, self.name,
                                        f"Total hole area {total_holes:.3f} > {max_total}",
                                        {"holes_area_m2": float(total_holes)}))
        return failures

class HolesAreaRatioLimit(Rule):
    name = "HolesAreaRatioLimit"
    applies_to = frozenset({"building"})
    category = "topology"
    description = "Sum of hole areas must be <= max_ratio * polygon area (in metric CRS)."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        ratio_max: float = float(self.params.get("max_ratio", 0.20))  # e.g., 20%
        target_epsg: Optional[int] = self.params.get("target_epsg", 7855)
        gdf_m = _to_metric(gdf, target_epsg=target_epsg)

        fails: List[Failure] = []
        for idx, geom in gdf_m.geometry.items():
            if geom is None or geom.is_empty:
                continue
            area = float(geom.area)  # includes holes
            if area <= 0:
                continue
            holes = 0.0
            for poly in _geom_iter_polygon_parts(geom):
                for ring in poly.interiors:
                    holes += Polygon(ring).area
            ratio = holes / area
            if ratio > ratio_max:
                fails.append(Failure(
                    idx, self.name,
                    f"Holes/Area ratio {ratio:.3f} > {ratio_max}",
                    {"holes_area_m2": holes, "area_m2": area}
                ))
        return fails



class OverlappingBuildings(Rule):
    category = "topology"
    name = "OverlappingBuildings"
    applies_to = frozenset({"building"})
    description = "Buildings must not overlap each other beyond tolerance (m^2). Touching is allowed."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        tol: float = float(self.params.get("area_tolerance", 0.05))  # m^2
        target_epsg: Optional[int] = self.params.get("target_epsg", 7855)
        gdf_m = _to_metric(gdf, target_epsg=target_epsg)

        geoms: List[base.BaseGeometry] = list(gdf_m.geometry.values)
        idxs:  List[Any] = list(gdf_m.index.values)
        tree = STRtree(geoms)

        pairs: List[Tuple[int, int]] = []

        # 优先用 Shapely 2.x 的 query_bulk（最快）
        try:
            qb = tree.query_bulk(geoms, predicate="intersects")
            for i, j in zip(qb[0], qb[1]):
                i, j = int(i), int(j)
                if i < j:
                    pairs.append((i, j))
        except AttributeError:
            # 1.x 回退：逐个 query
            # 为“几何对象返回”的情况准备 WKB 映射
            wkb_to_idxs: Dict[bytes, List[int]] = {}
            for i, g in enumerate(geoms):
                wkb_to_idxs.setdefault(g.wkb, []).append(i)

            seen: Set[Tuple[int, int]] = set()
            for i, gi in enumerate(geoms):
                nbrs = tree.query(gi)  # 可能是 numpy 索引数组，或“几何对象列表”
                # 情况 1：numpy 索引数组（pygeos 风格）
                if hasattr(nbrs, "dtype") and np.issubdtype(nbrs.dtype, np.integer):
                    for j in map(int, np.asarray(nbrs).tolist()):
                        if i < j:
                            t = (i, j)
                            if t not in seen:
                                seen.add(t)
                                pairs.append(t)
                else:
                    # 情况 2：几何对象列表（纯 shapely 1.x 风格）
                    for gj in nbrs:
                        idx_list = wkb_to_idxs.get(gj.wkb, None)
                        if idx_list:  # 直接用 WKB 命中
                            for j in idx_list:
                                if i < j:
                                    t = (i, j)
                                    if t not in seen:
                                        seen.add(t)
                                        pairs.append(t)
                        else:
                            # 极少数情况退回线性匹配 equals
                            for j, g2 in enumerate(geoms):
                                if gj.equals(g2) and i < j:
                                    t = (i, j)
                                    if t not in seen:
                                        seen.add(t)
                                        pairs.append(t)
                                    break

        overlaps_by_idx: Dict[Any, Dict[str, Any]] = {}
        for i, j in pairs:
            gi, gj = geoms[i], geoms[j]
            if gi.is_empty or gj.is_empty:
                continue
            inter = gi.intersection(gj)
            area = float(getattr(inter, "area", 0.0))
            if area > tol:
                idx_i, idx_j = idxs[i], idxs[j]
                for a, b in ((idx_i, idx_j), (idx_j, idx_i)):
                    rec = overlaps_by_idx.setdefault(a, {"overlaps_with": [], "overlap_area_sum_m2": 0.0})
                    rec["overlaps_with"].append(b)
                    rec["overlap_area_sum_m2"] += area

        return [
            Failure(idx, self.name,
                    f"Overlaps {len(info['overlaps_with'])} feature(s); total overlap area {info['overlap_area_sum_m2']:.3f} m^2",
                    info)
            for idx, info in overlaps_by_idx.items()
        ]



class PartsSelfOverlap(Rule):
    category = "topology"
    name = "PartsSelfOverlap"
    applies_to = frozenset({"building"})
    description = "Within a MultiPolygon, parts must not overlap each other."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        target_epsg: Optional[int] = self.params.get("target_epsg", 7855)
        gdf_m = _to_metric(gdf, target_epsg=target_epsg)

        failures: List[Failure] = []
        for idx, geom in gdf_m.geometry.items():
            if not isinstance(geom, MultiPolygon):
                continue
            parts = list(geom.geoms)
            if len(parts) <= 1:
                continue

            tree = STRtree(parts)
            pairs: List[Tuple[int, int]] = []

            try:
                qb = tree.query_bulk(parts, predicate="intersects")
                for i, j in zip(qb[0], qb[1]):
                    i, j = int(i), int(j)
                    if i < j:
                        pairs.append((i, j))
            except AttributeError:
                # 1.x 回退
                wkb_to_ids: Dict[bytes, List[int]] = {}
                for i, p in enumerate(parts):
                    wkb_to_ids.setdefault(p.wkb, []).append(i)

                seen: Set[Tuple[int, int]] = set()
                for i, pi in enumerate(parts):
                    nbrs = tree.query(pi)
                    if hasattr(nbrs, "dtype") and np.issubdtype(nbrs.dtype, np.integer):
                        for j in map(int, np.asarray(nbrs).tolist()):
                            if i < j:
                                t = (i, j)
                                if t not in seen:
                                    seen.add(t); pairs.append(t)
                    else:
                        for pj in nbrs:
                            idx_list = wkb_to_ids.get(pj.wkb, None)
                            if idx_list:
                                for j in idx_list:
                                    if i < j:
                                        t = (i, j)
                                        if t not in seen:
                                            seen.add(t); pairs.append(t)
                            else:
                                for j, p2 in enumerate(parts):
                                    if pj.equals(p2) and i < j:
                                        t = (i, j)
                                        if t not in seen:
                                            seen.add(t); pairs.append(t)
                                        break

            has_overlap = False
            for i, j in pairs:
                inter = parts[i].intersection(parts[j])
                if float(getattr(inter, "area", 0.0)) > 0.0:
                    has_overlap = True
                    break
            if has_overlap:
                failures.append(Failure(idx, self.name, "MultiPolygon parts overlap"))
        return failures


class SliverPolygons(Rule):
    category = "shape"
    name = "SliverPolygons"
    applies_to = frozenset({"building"})
    description = "Flag polygons with area/perimeter ratio below threshold (sliver-like)."

    def _check(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        ratio_min: float = float(self.params.get("min_area_perimeter_ratio", 0.2))
        target_epsg: Optional[int] = self.params.get("target_epsg", 7855)
        gdf_m = _to_metric(gdf, target_epsg=target_epsg)
        areas = gdf_m.geometry.area
        perims = gdf_m.geometry.length.replace(0, pd.NA)
        ratio = areas / perims
        mask = ratio < ratio_min
        return [
            Failure(idx, self.name, f"Area/Perimeter ratio {ratio.loc[idx]:.3f} < {ratio_min}",
                    {"area_m2": float(areas.loc[idx]), "perimeter_m": float(perims.loc[idx])})
            for idx in gdf.index[mask.fillna(False)]
        ]


# -------------------------
# Validation engine & presets
# -------------------------

class ValidationEngine:
    """Run a set of Rule instances against a GeoDataFrame."""
    def __init__(self, rules: Sequence[Rule]):
        self.rules = list(rules)

    def run(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        out: List[pd.DataFrame] = []
        for rule in self.rules:
            df = rule.check(gdf)
            if not df.empty:
                df["rule"] = rule.name
                df["category"] = getattr(rule, "category", "general")
                out.append(df)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            columns=["index", "error_type", "message", "rule"]
        )

    def run_with_geometry(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Return failures joined back with geometry for mapping."""
        df = self.run(gdf)
        if df.empty:
            return gpd.GeoDataFrame(df, geometry=[])
        # Note: failures may include index=None rows (dataset-level errors like CRS)
        # Keep only row-level failures for join.
        row_df = df.dropna(subset=["index"]).copy()
        row_df = row_df.set_index("index", drop=False)
        joined = row_df.join(gdf[["geometry"]], on="index", how="left")
        return gpd.GeoDataFrame(joined, geometry="geometry", crs=gdf.crs)


def default_building_rules(
        melb_target_epsg: int = 7855,
        min_area_m2: float = 5.0,
        overlap_tol_m2: float = 0.1,
        max_total_hole_area_m2: float = 2.0,
        sliver_ratio_min: float = 0.15,
        id_field: Optional[str] = None,
        required_fields: Optional[Sequence[str]] = None,
        unique_fields: Optional[Sequence[str]] = None
) -> List[Rule]:
    """
    Sensible defaults for Melbourne buildings.
    - EPSG 7855 = GDA2020 / MGA Zone 55 (Melbourne); adjust if needed.
    - All thresholds are adjustable via parameters.
    """
    req_fields = list(required_fields or [])
    uniq_fields = list(unique_fields or ([] if id_field is None else [id_field]))

    rules: List[Rule] = [
        ExpectedCRS(allowed=(4326, 7844, 28355, 7855, "EPSG:4326", "EPSG:7844", "EPSG:28355", "EPSG:7855")),
        GeometryNotEmpty(),
        GeometryValid(),
        BuildingPolygonsOnly(),
        AttributeNotNull(fields=req_fields, skip_missing=True),
        *(UniqueField(field=f, skip_missing=True) for f in uniq_fields),
        DuplicateGeometries(),
        MinimumArea(min_area=min_area_m2, target_epsg=melb_target_epsg),
        HolesTotalAreaLimit(max_total_hole_area=max_total_hole_area_m2, target_epsg=melb_target_epsg),
        OverlappingBuildings(area_tolerance=overlap_tol_m2, target_epsg=melb_target_epsg),
        PartsSelfOverlap(target_epsg=melb_target_epsg),
        SliverPolygons(min_area_perimeter_ratio=sliver_ratio_min, target_epsg=melb_target_epsg),
    ]
    return rules


# -------------------------
# Tiny demo (optional)
# -------------------------
if __name__ == "__main__":
    # Example usage with your sample file path:
    import sys
    path = "/mnt/data/geoscape-geoscape-melbourne-buildings-jun22-na.geojson"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    gdf = gpd.read_file(path)

    # Tip: set id_field / required_fields to columns that actually exist in your dataset.
    # For Geoscape, you may have fields like 'BUILDING_PID', etc. Adjust as needed.
    rules = default_building_rules(
        melb_target_epsg=7855,
        min_area_m2=5.0,
        overlap_tol_m2=0.1,
        max_total_hole_area_m2=2.0,
        sliver_ratio_min=0.15,
        id_field="BUILDING_PID",                 # <-- replace with your real unique ID column
        required_fields=["BUILDING_PID"]         # <-- add more required fields if desired
    )

    engine = ValidationEngine(rules)
    failures = engine.run(gdf)
    print(failures.head())
    failures.to_csv("validation_failures.csv", index=False)
    print("Saved to validation_failures.csv")
