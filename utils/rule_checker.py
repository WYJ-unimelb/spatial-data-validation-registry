from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from shapely.geometry import LineString, MultiLineString, base, Polygon, MultiPolygon, LinearRing, box
from shapely.strtree import STRtree
from shapely.validation import explain_validity
from shapely.ops import unary_union
from difflib import get_close_matches
from shapely.ops import triangulate
from shapely.errors import TopologicalError
try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None
import warnings
import numpy as np
import math
import pandas as pd
import geopandas as gpd


    # === ADD: simple runners for the Streamlit frontend ===
def run_suburb_checks(gdf, min_area_m2: float = 30, min_shared_len_m: float = 5):
    sc = SuburbChecker()
    fails = []
    fails += sc.check_geometries(gdf)                 # InvalidGeometry
    fails += sc.check_overlaps(gdf)                   # Overlap
    fails += sc.check_slivers(gdf, min_area_m2=min_area_m2)  # SliverPolygon
    fails += sc.check_areas(gdf)                      # AreaNonPositive
    fails += sc.check_gaps(gdf)                       # TopologicalGaps
    fails += sc.check_holes(gdf)                      # SmallHole
    fails += sc.check_non_contiguous_parts(gdf)       # NonContiguousParts
    fails += sc.check_misaligned_boundaries(gdf)      # BoundaryMisalignment
    fails += sc.check_shared_boundary_length(gdf, min_shared_len_m=min_shared_len_m)  # VeryShortSharedBoundary
    fails += sc.check_enclaves(gdf)                   # Enclave
    fails += sc.check_crs_and_units(gdf)              # CRSNotProjected
    fails += sc.check_centroid_inside(gdf)            # CentroidOutside
    fails += sc.check_duplicate_geometries(gdf)       # DuplicateGeometry
    try: fails += sc.check_semantics(gdf)             # DuplicateSuburbName / MissingPostcode / InvalidLegalStartDate
    except Exception: pass
    try: fails += sc.check_postcode_geometry_consistency(gdf)  # SuburbToMultiplePostcodes / PostcodeToMultipleSuburbs
    except Exception: pass
    try: fails += sc.check_attribute_consistency_on_borders(gdf, attrs=("state",))
    except Exception: pass
    try: fails += sc.check_name_typos(gdf)
    except Exception: pass

    return failures_to_frame(fails)

def run_road_checks(gdf, start_col="start_node_name", end_col="end_node_name"):
    ric = RoadIsolationChecker(gdf, start_col=start_col, end_col=end_col)
    fails = []
    fails += ric.compute_candidate_isolated()
    fails += ric.compute_final_isolated(strict=False, buffer_eps=0.0)
    return failures_to_frame(fails)

def run_building_checks(
    gdf,
    overlap_tol_m2: float = 0.05,
    holes_ratio_max: float = 0.20,
    min_area_m2: float = 1.0,
    hole_total_area_max_m2: float = 2.0,
    sliver_ratio_min: float = 0.20,
):
    bc = BuildingChecker(work_epsg=7855)
    fails = []

    fails += bc.check_geometry_not_empty(gdf)
    fails += bc.check_geometry_valid(gdf)
    fails += bc.check_building_polygons_only(gdf)

    fails += bc.check_duplicate_geometries(gdf)
    fails += bc.check_expected_crs(gdf)

    fails += bc.check_overlapping_buildings(gdf, area_tolerance=overlap_tol_m2, target_epsg=7855)
    fails += bc.check_holes_area_ratio_limit(gdf, max_ratio=holes_ratio_max, target_epsg=7855)
    fails += bc.check_holes_total_area_limit(gdf, max_total_hole_area=hole_total_area_max_m2, target_epsg=7855)

    fails += bc.check_minimum_area(gdf, min_area=min_area_m2, target_epsg=7855)
    fails += bc.check_parts_self_overlap(gdf, target_epsg=7855)
    fails += bc.check_sliver_polygons(gdf, min_area_perimeter_ratio=sliver_ratio_min, target_epsg=7855)

    return bc.failures_to_frame(fails)

def run_blocks_checks(gdf, min_len_m: float = 2.0):
    chk = BlockChecker()
    fails = []
    # Basic geometry quality (notebook checks both validity and simplicity)
    fails += chk.check_invalid_geometry(gdf)
    fails += chk.check_not_simple(gdf)
    # Pairwise overlaps (any positive area counts in the notebook)
    fails += chk.check_overlaps(gdf, min_area=0.0)
    # Shape-quality checks aligned with notebook semantics
    fails += chk.check_short_edges(gdf, min_len=min_len_m)            # exterior-only
    fails += chk.check_wrong_orientation(gdf)                          # CCW expected
    fails += chk.check_nonmanifold_edges(gdf)                          # exterior-only
    fails += chk.check_min_angle(gdf, min_deg=5.0, include_interiors=False)  # exterior only
    fails += chk.check_excessive_precision(gdf, max_decimals=3)        # exterior-only, string-based
    # Extra checks present in the notebook
    fails += chk.check_triangulation(gdf, tol=1e-1)
    fails += chk.check_outside_bbox(gdf, bbox=((140, -40), (150, -33)))
    return failures_to_frame(fails)

# Failure
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


def failures_to_frame(fails: List[Failure]) -> pd.DataFrame:
    """Optional helper: convert failures to a DataFrame."""
    return pd.DataFrame([f.to_dict() for f in fails]) if fails else pd.DataFrame(
        columns=["index", "error_type", "message"]
    )


# TODO: @ ZhiZhou
class SuburbChecker:
    def __init__(self, work_epsg: int = 3107):
        self.work_epsg = work_epsg

    # ---------- utils ----------
    @staticmethod
    def _clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        g = gdf.copy()
        g["geometry"] = g["geometry"].buffer(0)
        g = g.explode(index_parts=False, ignore_index=True)
        return g

    @staticmethod
    def _project(gdf: gpd.GeoDataFrame, epsg: Optional[int], assume_src_epsg: int = 4326):
        """Safe projection: if CRS is missing, set a reasonable default first."""
        if not epsg:
            return gdf
        if gdf.crs is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gdf = gdf.set_crs(epsg=assume_src_epsg)
        return gdf.to_crs(epsg=epsg)


    # ---------- checks (Failures) ----------
    @staticmethod
    def check_overlaps(gdf: gpd.GeoDataFrame) -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        sidx = g.sindex
        failures: List[Failure] = []
        for i, geom1 in enumerate(g.geometry):
            for j in sidx.intersection(geom1.bounds):
                if j <= i:
                    continue
                geom2 = g.geometry.iloc[j]
                try:
                    if geom1.intersects(geom2):
                        inter = geom1.intersection(geom2)
                        if inter.area > 0:
                            # emit a failure for BOTH features so each row knows who it overlaps with
                            info = {
                                "other_index": j,
                                "suburb": g.iloc[i].get("suburb"),
                                "other_suburb": g.iloc[j].get("suburb"),
                                "intersection_area": float(inter.area),
                            }
                            failures.append(Failure(i, "Overlap", "Polygon overlaps neighbour", info))
                            info2 = {
                                "other_index": i,
                                "suburb": g.iloc[j].get("suburb"),
                                "other_suburb": g.iloc[i].get("suburb"),
                                "intersection_area": float(inter.area),
                            }
                            failures.append(Failure(j, "Overlap", "Polygon overlaps neighbour", info2))
                except TopologicalError:
                    continue
        return failures

    @staticmethod
    def check_areas(gdf: gpd.GeoDataFrame, reproject_epsg: int = 3107) -> List[Failure]:
        g = SuburbChecker._project(gdf, reproject_epsg)
        failures = []
        areas = g.geometry.area
        for idx in g.index[areas <= 0]:
            failures.append(Failure(idx, "AreaNonPositive", "Area <= 0", {"area": float(areas.loc[idx])}))
        return failures

    @staticmethod
    def check_geometries(gdf: gpd.GeoDataFrame) -> List[Failure]:
        failures = []
        mask = ~gdf.geometry.is_valid
        for idx, geom in gdf.loc[mask, "geometry"].items():
            failures.append(Failure(idx, "InvalidGeometry", explain_validity(geom)))
        return failures

    @staticmethod
    def check_gaps(gdf: gpd.GeoDataFrame) -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        dissolved = g.dissolve().geometry.iloc[0]
        merged = unary_union(g.geometry)
        gaps = dissolved.symmetric_difference(merged)
        if gaps.is_empty:
            return []
        # One record with the gap geometry included in extras
        return [Failure(None, "TopologicalGaps", "Potential gaps between polygons", {"geometry": gaps})]

    @staticmethod
    def check_semantics(gdf: gpd.GeoDataFrame) -> List[Failure]:
        failures: List[Failure] = []
        dup_counts = gdf.groupby("suburb").size()
        duplicate_suburbs = dup_counts[dup_counts > 1]
        for suburb, cnt in duplicate_suburbs.items():
            failures.append(Failure(None, "DuplicateSuburbName", f"Suburb name repeated {cnt} times",
                                    {"suburb": suburb, "count": int(cnt)}))
        for idx in gdf.index[gdf["postcode"].isna()]:
            failures.append(Failure(idx, "MissingPostcode", "Postcode is missing"))
        bad_date_mask = pd.to_datetime(gdf["legalstartdate"], errors="coerce").isna()
        for idx in gdf.index[bad_date_mask]:
            failures.append(Failure(idx, "InvalidLegalStartDate", "Cannot parse legalstartdate"))
        return failures

    @staticmethod
    def check_misaligned_boundaries(gdf: gpd.GeoDataFrame, epsg: int = 3107, tolerance: float = 0.5) -> List[Failure]:
        g = SuburbChecker._project(SuburbChecker._clean_geoms(gdf), epsg)
        sidx = g.sindex
        failures: List[Failure] = []
        for idx, geom in g.geometry.items():
            for nidx in sidx.intersection(geom.bounds):
                if nidx <= idx:
                    continue
                geom2 = g.geometry.iloc[nidx]
                if not geom.touches(geom2):
                    continue
                shared = geom.boundary.intersection(geom2.boundary)
                if shared.is_empty or not shared.is_valid or shared.length == 0:
                    continue
                buf = shared.buffer(tolerance)
                if not (geom.intersects(buf) and geom2.intersects(buf)):
                    failures.append(
                        Failure(idx, "BoundaryMisalignment",
                                "Suspicious shared boundary alignment",
                                {"other_index": nidx, "shared_length": float(shared.length), "geometry": shared})
                    )
                    failures.append(
                        Failure(nidx, "BoundaryMisalignment",
                                "Suspicious shared boundary alignment",
                                {"other_index": idx, "shared_length": float(shared.length), "geometry": shared})
                    )
        return failures

    @staticmethod
    def check_slivers(gdf: gpd.GeoDataFrame, min_area_m2: float = 50, epsg: int = 3107) -> List[Failure]:
        g = SuburbChecker._project(SuburbChecker._clean_geoms(gdf), epsg)
        failures: List[Failure] = []
        for idx in g.index[g.area < min_area_m2]:
            failures.append(Failure(idx, "SliverPolygon",
                                    f"Area < {min_area_m2} m^2", {"area": float(g.area.loc[idx])}))
        return failures

    @staticmethod
    def check_holes(gdf: gpd.GeoDataFrame, max_hole_area_m2: float = 25, epsg: int = 3107) -> List[Failure]:
        g = SuburbChecker._project(SuburbChecker._clean_geoms(gdf), epsg)
        failures: List[Failure] = []
        for i, geom in enumerate(g.geometry):
            if isinstance(geom, Polygon):
                interiors = list(geom.interiors)
                for ring in interiors:
                    hole_poly = Polygon(ring)
                    if hole_poly.area <= max_hole_area_m2:
                        failures.append(Failure(i, "SmallHole",
                                                f"Hole area ≤ {max_hole_area_m2} m^2",
                                                {"hole_area": float(hole_poly.area), "geometry": hole_poly}))
            elif isinstance(geom, MultiPolygon):
                for part in geom.geoms:
                    for ring in part.interiors:
                        hole_poly = Polygon(ring)
                        if hole_poly.area <= max_hole_area_m2:
                            failures.append(Failure(i, "SmallHole",
                                                    f"Hole area ≤ {max_hole_area_m2} m^2",
                                                    {"hole_area": float(hole_poly.area), "geometry": hole_poly}))
        return failures

    @staticmethod
    def check_non_contiguous_parts(gdf: gpd.GeoDataFrame, key: str = "suburb") -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        parts = (g.explode(index_parts=False, ignore_index=True)
                 .assign(_ones=1)
                 .dissolve(by=[key], as_index=False, aggfunc={"_ones": "count"}))
        failures: List[Failure] = []
        for _, row in parts[parts["_ones"] > 1].iterrows():
            failures.append(Failure(None, "NonContiguousParts",
                                    "Suburb has multiple disconnected parts",
                                    {"suburb": row[key], "num_parts": int(row["_ones"])})
                           )
        return failures

    @staticmethod
    def check_shared_boundary_length(gdf: gpd.GeoDataFrame, min_shared_len_m: float = 5, epsg: int = 3107) -> List[Failure]:
        g = SuburbChecker._project(SuburbChecker._clean_geoms(gdf), epsg)
        sidx = g.sindex
        failures: List[Failure] = []
        for i, a in enumerate(g.geometry):
            for j in sidx.intersection(a.bounds):
                if j <= i:
                    continue
                b = g.geometry.iloc[j]
                if a.touches(b):
                    shared = a.boundary.intersection(b.boundary)
                    if not shared.is_empty and shared.length < min_shared_len_m:
                        info = {
                            "other_index": j,
                            "suburb": g.iloc[i].get("suburb"),
                            "other_suburb": g.iloc[j].get("suburb"),
                            "shared_length_m": float(shared.length),
                            "geometry": shared,
                        }
                        failures.append(Failure(i, "VeryShortSharedBoundary",
                                                f"Shared boundary < {min_shared_len_m} m", info))
                        failures.append(Failure(j, "VeryShortSharedBoundary",
                                                f"Shared boundary < {min_shared_len_m} m",
                                                {**info, "other_index": i}))
        return failures

    @staticmethod
    def check_enclaves(gdf: gpd.GeoDataFrame) -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        sidx = g.sindex
        failures: List[Failure] = []
        for i, a in enumerate(g.geometry):
            for j in sidx.intersection(a.bounds):
                if j == i:
                    continue
                b = g.geometry.iloc[j]
                if a.within(b) and g.iloc[i].get("suburb") != g.iloc[j].get("suburb"):
                    failures.append(Failure(i, "Enclave",
                                            "Polygon is fully inside a different suburb",
                                            {"outer_index": j,
                                             "inner_suburb": g.iloc[i].get("suburb"),
                                             "outer_suburb": g.iloc[j].get("suburb")}))
        return failures

    @staticmethod
    def check_crs_and_units(gdf: gpd.GeoDataFrame, must_be_projected: bool = True) -> List[Failure]:
        crs = gdf.crs
        ok = crs is not None and getattr(crs, "is_projected", False)
        if must_be_projected and not ok:
            return [Failure(None, "CRSNotProjected",
                            "Data are not in a projected CRS. Reproject before area/length checks.")]
        return []

    @staticmethod
    def check_centroid_inside(gdf: gpd.GeoDataFrame) -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        failures: List[Failure] = []
        for i, geom in enumerate(g.geometry):
            c = geom.centroid
            if not geom.contains(c):
                pos = geom.representative_point()
                failures.append(Failure(i, "CentroidOutside",
                                        "Centroid not contained in polygon",
                                        {"centroid": c, "point_on_surface": pos}))
        return failures

    @staticmethod
    def check_bbox_within_extent(gdf: gpd.GeoDataFrame, state_extent: gpd.GeoSeries) -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        extent_union = unary_union(state_extent)
        failures: List[Failure] = []
        for idx in g.index[~g.geometry.within(extent_union)]:
            failures.append(Failure(idx, "OutsideExtent", "Feature outside provided extent"))
        return failures

    @staticmethod
    def check_attribute_consistency_on_borders(gdf: gpd.GeoDataFrame, attrs: Sequence[str] = ("state",)) -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        sidx = g.sindex
        failures: List[Failure] = []
        for i, a in enumerate(g.geometry):
            for j in sidx.intersection(a.bounds):
                if j <= i:
                    continue
                b = g.geometry.iloc[j]
                if a.touches(b):
                    for attr in attrs:
                        vi, vj = g.iloc[i].get(attr), g.iloc[j].get(attr)
                        if vi != vj:
                            failures.append(Failure(i, "NeighbourAttributeMismatch",
                                                    f"Attribute '{attr}' mismatch with neighbour",
                                                    {"other_index": j, "attr": attr, "val_i": vi, "val_j": vj}))
                            failures.append(Failure(j, "NeighbourAttributeMismatch",
                                                    f"Attribute '{attr}' mismatch with neighbour",
                                                    {"other_index": i, "attr": attr, "val_i": vj, "val_j": vi}))
        return failures

    @staticmethod
    def check_name_typos(gdf: gpd.GeoDataFrame, name_col: str = "suburb", cutoff: float = 0.88) -> List[Failure]:
        names = sorted(set(map(str, gdf[name_col].fillna(""))))
        failures: List[Failure] = []
        for i, n in enumerate(names):
            for m in get_close_matches(n, names[i + 1 :], n=5, cutoff=cutoff):
                failures.append(Failure(None, "NearDuplicateName", f"'{n}' ≈ '{m}'", {"name_a": n, "name_b": m}))
        return failures

    @staticmethod
    def check_spikes_and_bows(gdf: gpd.GeoDataFrame, min_angle_deg: float = 5) -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        failures: List[Failure] = []
        for i, geom in enumerate(g.geometry):
            if isinstance(geom, Polygon):
                coords = list(geom.exterior.coords)
                for k in range(1, len(coords) - 1):
                    a, b, c = coords[k - 1], coords[k], coords[k + 1]
                    v1 = (a[0] - b[0], a[1] - b[1])
                    v2 = (c[0] - b[0], c[1] - b[1])
                    norm = (v1[0] ** 2 + v1[1] ** 2) ** 0.5 * (v2[0] ** 2 + v2[1] ** 2) ** 0.5
                    if norm == 0:
                        continue
                    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / norm))
                    ang = math.degrees(math.acos(cosang))
                    if ang < min_angle_deg:
                        failures.append(Failure(i, "AcuteBoundaryAngle",
                                                f"Angle {ang:.2f}° < {min_angle_deg}°",
                                                {"vertex_idx": k, "angle_deg": float(ang)}))
        return failures

    @staticmethod
    def check_duplicate_geometries(gdf: gpd.GeoDataFrame) -> List[Failure]:
        g = SuburbChecker._clean_geoms(gdf)
        wkb = g.geometry.apply(lambda x: x.wkb)
        dup_idx = wkb[wkb.duplicated(keep=False)].index
        return [Failure(idx, "DuplicateGeometry", "Exact duplicate geometry") for idx in dup_idx]

    @staticmethod
    def check_postcode_geometry_consistency(
        gdf: gpd.GeoDataFrame, key: str = "suburb", code: str = "postcode"
    ) -> List[Failure]:
        failures: List[Failure] = []
        sub_to_pc = gdf.groupby(key)[code].nunique().reset_index()
        for _, r in sub_to_pc[sub_to_pc[code] > 1].iterrows():
            failures.append(Failure(None, "SuburbToMultiplePostcodes",
                                    "Same suburb has multiple postcodes",
                                    {"suburb": r[key], "num_postcodes": int(r[code])}))
        pc_to_sub = gdf.groupby(code)[key].nunique().reset_index()
        for _, r in pc_to_sub[pc_to_sub[key] > 1].iterrows():
            failures.append(Failure(None, "PostcodeToMultipleSuburbs",
                                    "Same postcode covers multiple suburb names",
                                    {"postcode": r[code], "num_suburbs": int(r[key])}))
        return failures

# TODO: @ Yaojin
class RoadIsolationChecker:
    def __init__(self, gdf, start_col: str = "start_node_name", end_col: str = "end_node_name"):
        self.gdf = gdf
        self.start_col = start_col
        self.end_col = end_col
        self.truly_isolated = None
        self.final_isolated = None
        self.problematic = None

    @classmethod
    def from_file(cls, path, start_col: str = "start_node_name", end_col: str = "end_node_name"):
        gdf = gpd.read_file(path)
        return cls(gdf, start_col=start_col, end_col=end_col)

    # ---------- endpoint candidates ----------
    def compute_candidate_isolated(self) -> List[Failure]:
        from collections import Counter

        all_nodes = list(self.gdf[self.start_col]) + list(self.gdf[self.end_col])
        counts = Counter(all_nodes)
        once_nodes = {n for n, c in counts.items() if c == 1}

        def both_ends_once(row):
            return (row[self.start_col] in once_nodes) and (row[self.end_col] in once_nodes)

        once_node_rows = self.gdf[
            self.gdf[self.start_col].isin(once_nodes) | self.gdf[self.end_col].isin(once_nodes)
        ]
        self.truly_isolated = once_node_rows[once_node_rows.apply(both_ends_once, axis=1)]

        failures: List[Failure] = []
        for idx, r in self.truly_isolated.iterrows():
            failures.append(Failure(idx, "EndpointIsolated",
                                    "Segment has endpoints that appear only once in the network",
                                    {"start": r[self.start_col], "end": r[self.end_col]}))
        return failures

    # ---------- geometric confirmation ----------
    def compute_final_isolated(self, strict: bool = False, buffer_eps: float = 0.0) -> List[Failure]:
        if self.truly_isolated is None:
            raise ValueError("Please call compute_candidate_isolated() first.")

        still_isolated_idx = []
        has_sindex = hasattr(self.gdf, "sindex") and (self.gdf.sindex is not None)

        for idx, row in self.truly_isolated.iterrows():
            geom = row.geometry
            if buffer_eps and isinstance(geom, (LineString, MultiLineString)):
                geom = geom.buffer(buffer_eps)

            candidates = self.gdf[~self.gdf.index.isin([idx])]
            if has_sindex:
                possible = candidates.sindex.query(geom, predicate="intersects")
                candidates = candidates.iloc[possible]

            if len(candidates) == 0:
                still_isolated_idx.append(idx)
                continue

            rel = (candidates.geometry.crosses(geom) | candidates.geometry.overlaps(geom)) if strict \
                else candidates.geometry.intersects(geom)

            if not rel.any():
                still_isolated_idx.append(idx)

        self.final_isolated = self.truly_isolated.loc[still_isolated_idx]
        self.problematic = self.truly_isolated.drop(self.final_isolated.index)

        failures: List[Failure] = []
        for idx, r in self.final_isolated.iterrows():
            failures.append(Failure(idx, "IsolatedRoad",
                                    "Road segment is isolated (no geometric connection to others)",
                                    {"start": r[self.start_col], "end": r[self.end_col]}))
        # Optional: emit “false positives” for visibility
        for idx, r in self.problematic.iterrows():
            failures.append(Failure(idx, "EndpointButConnected",
                                    "Endpoint test flagged it, but it intersects other roads",
                                    {"start": r[self.start_col], "end": r[self.end_col]}))
        return failures

    # ---------- misc helpers preserved ----------
    def summary(self, n: int = 10):
        if self.truly_isolated is None:
            return
        print(f"endpoint testing  {len(self.truly_isolated)} pieces of data")
        if self.final_isolated is not None:
            print(f"geometric testing {len(self.final_isolated)} pieces of data")
            print(self.final_isolated[[self.start_col, self.end_col]].head(n))
            print(f"{len(self.problematic)} pieces of different data")
            print(self.problematic[[self.start_col, self.end_col]].head(n))

    def export(self,
               final_geojson: Optional[str] = None,
               problematic_geojson: Optional[str] = None,
               final_csv: Optional[str] = None,
               problematic_csv: Optional[str] = None):
        if self.final_isolated is not None:
            if final_geojson: self.final_isolated.to_file(final_geojson, driver="GeoJSON")
            if final_csv: self.final_isolated.to_csv(final_csv, index=False)
        if self.problematic is not None:
            if problematic_geojson: self.problematic.to_file(problematic_geojson, driver="GeoJSON")
            if problematic_csv: self.problematic.to_csv(problematic_csv, index=False)

    def plot(self, show_background: bool = True, figsize=(10, 10)):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        if show_background:
            self.gdf.plot(ax=ax, color="lightsteelblue", linewidth=0.3)
        if self.truly_isolated is not None:
            self.truly_isolated.plot(ax=ax, color="orange", linewidth=0.8, label="Endpoint candidates")
        if self.final_isolated is not None:
            self.final_isolated.plot(ax=ax, color="red", linewidth=1.2, label="Final isolated")
        ax.set_title("Road Isolation")
        ax.set_axis_off()
        ax.legend()
        plt.show()

# TODO: @ Lei
class BuildingChecker:
    """
    One-stop geometric/attribute validator.

    Each `check_*` method returns `List[Failure]`.
    Use `failures_to_frame(...)` if you want a DataFrame instead.
    """

    # -------------------------
    # Config / construction
    # -------------------------
    def __init__(self,
                 work_epsg: int = 7855,                 # GDA2020 / MGA Zone 55 (Melbourne)
                 allowed_crs: Iterable[Union[int,str]] = (4326, 7844, "EPSG:4326", "EPSG:7844")):
        self.work_epsg = work_epsg
        self.allowed_crs = self._allowed_crs_to_str_set(allowed_crs)

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _ensure_geometry(gdf: gpd.GeoDataFrame) -> None:
        if "geometry" not in gdf:
            raise ValueError("GeoDataFrame must contain a 'geometry' column.")

    @staticmethod
    def _geom_iter_polygon_parts(geom: base.BaseGeometry):
        """Yield Polygon parts for Polygon and MultiPolygon."""
        if isinstance(geom, Polygon):
            yield geom
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                yield p

    @staticmethod
    def _allowed_crs_to_str_set(allowed: Iterable[Union[int,str]]) -> Set[str]:
        out = set()
        for a in allowed:
            if isinstance(a, int):
                out.add(f"EPSG:{a}")
            else:
                s = str(a).upper().replace("EPSG:", "").strip()
                out.add(f"EPSG:{s}")
        return out

    @staticmethod
    def _clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Simple fix for invalids and explode multiparts for per-part checks."""
        g = gdf.copy()
        g["geometry"] = g["geometry"].buffer(0)
        g = g.explode(index_parts=False, ignore_index=True)
        return g

    def _to_metric(self,
                   gdf: gpd.GeoDataFrame,
                   target_epsg: Optional[int] = None,
                   fallback_epsg: int = 3857) -> gpd.GeoDataFrame:
        """
        Project to a metric CRS for area/distance calculations.
        If target_epsg is None, use self.work_epsg (or fallback if that fails).
        """
        if gdf.crs is None:
            warnings.warn("GeoDataFrame has no CRS; metrics will be unitless (no reprojection).")
            return gdf
        epsg = target_epsg if target_epsg is not None else self.work_epsg
        try:
            return gdf.to_crs(epsg=epsg)
        except Exception as e:
            warnings.warn(f"Reprojection to EPSG:{epsg} failed ({e}); trying fallback EPSG:{fallback_epsg}.")
            try:
                return gdf.to_crs(epsg=fallback_epsg)
            except Exception as e2:
                warnings.warn(f"Fallback reprojection failed ({e2}); using original CRS.")
                return gdf

    @staticmethod
    def failures_to_frame(failures: List[Failure]) -> pd.DataFrame:
        if not failures:
            return pd.DataFrame(columns=["index", "error_type", "message"])
        return pd.DataFrame([f.to_dict() for f in failures])

    # -------------------------
    # Common checks
    # -------------------------
    def check_geometry_not_empty(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        """Geometry must not be empty or NA."""
        self._ensure_geometry(gdf)
        mask = gdf.geometry.isna() | gdf.geometry.is_empty
        return [Failure(idx, "GeometryNotEmpty", "Empty geometry") for idx in gdf.index[mask]]

    def check_geometry_valid(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        """Geometry must be valid; attach shapely explain_validity message."""
        self._ensure_geometry(gdf)
        mask = ~gdf.geometry.is_valid
        failures: List[Failure] = []
        if mask.any():
            for idx, geom in gdf.loc[mask, "geometry"].items():
                failures.append(Failure(idx, "GeometryValid", explain_validity(geom)))
        return failures

    def check_expected_crs(self, gdf: gpd.GeoDataFrame,
                           allowed: Optional[Iterable[Union[int,str]]] = None) -> List[Failure]:
        """CRS must be in allowed list."""
        allowed_set = self._allowed_crs_to_str_set(allowed) if allowed is not None else self.allowed_crs
        if gdf.crs is None:
            return [Failure(None, "ExpectedCRS", "CRS is missing (None).")]
        current = str(gdf.crs).upper().replace("EPSG:", "").strip()
        current = f"EPSG:{current}"
        if current not in allowed_set:
            return [Failure(None, "ExpectedCRS", f"CRS {current} not in allowed set {sorted(allowed_set)}")]
        return []

    def check_attribute_not_null(self, gdf: gpd.GeoDataFrame,
                                 fields: Sequence[str], skip_missing: bool = True) -> List[Failure]:
        """Selected attributes must not be null. Can skip missing fields."""
        failures: List[Failure] = []
        for col in fields:
            if col not in gdf.columns:
                if skip_missing:
                    continue
                failures.append(Failure(None, "AttributeNotNull", f"Field '{col}' not found"))
                continue
            mask = gdf[col].isna()
            failures.extend(
                Failure(idx, "AttributeNotNull", f"Null in field '{col}'", {"field": col})
                for idx in gdf.index[mask]
            )
        return failures

    def check_attribute_in_domain(self, gdf: gpd.GeoDataFrame, field: str,
                                  domain: Iterable[Any], skip_missing: bool = True) -> List[Failure]:
        """Attribute values must be in a given domain."""
        dom: Set[Any] = set(domain)
        if field not in gdf.columns:
            if skip_missing:
                return []
            return [Failure(None, "AttributeInDomain", f"Field '{field}' not found")]
        mask = ~gdf[field].isin(dom)
        return [
            Failure(idx, "AttributeInDomain", f"Value '{val}' not in domain", {"field": field})
            for idx, val in gdf.loc[mask, field].items()
        ]

    def check_unique_field(self, gdf: gpd.GeoDataFrame, field: str,
                           skip_missing: bool = True) -> List[Failure]:
        """Attribute must be unique among features."""
        if field not in gdf.columns:
            if skip_missing:
                return []
            return [Failure(None, "UniqueField", f"Field '{field}' not found")]
        counts = gdf[field].value_counts(dropna=False)
        dup_vals = counts[counts > 1].index
        if len(dup_vals) == 0:
            return []
        mask = gdf[field].isin(dup_vals)
        return [
            Failure(idx, "UniqueField", f"Duplicate value '{val}'", {"field": field})
            for idx, val in gdf.loc[mask, field].items()
        ]

    def check_duplicate_geometries(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        """Flag identical geometries (by WKB)."""
        wkb = gdf.geometry.to_wkb()  # bytes
        counts = pd.Series(wkb).value_counts(dropna=False)
        dup_wkbs = counts[counts > 1].index
        if len(dup_wkbs) == 0:
            return []
        mask = pd.Series(wkb).isin(dup_wkbs).values
        return [Failure(idx, "DuplicateGeometries", "Duplicate geometry")
                for idx, _ in gdf.loc[mask, "geometry"].items()]

    def check_within_bounding_polygon(self, gdf: gpd.GeoDataFrame,
                                      boundary: Union[Polygon, MultiPolygon]) -> List[Failure]:
        """Feature centroid must lie within the provided boundary polygon."""
        mask = ~gdf.geometry.centroid.within(boundary)
        return [Failure(idx, "WithinBoundingPolygon", "Feature centroid outside boundary")
                for idx in gdf.index[mask]]

    # -------------------------
    # Building-specific checks
    # -------------------------
    def check_building_polygons_only(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        """Buildings must be Polygon or MultiPolygon."""
        mask = ~gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        return [Failure(idx, "BuildingPolygonsOnly", "Geometry must be Polygon/MultiPolygon")
                for idx in gdf.index[mask]]

    def check_minimum_area(self, gdf: gpd.GeoDataFrame, min_area: float = 1.0,
                           target_epsg: Optional[int] = None) -> List[Failure]:
        """Polygon area must be >= min_area (m^2) in metric CRS."""
        gdf_m = self._to_metric(gdf, target_epsg=target_epsg or self.work_epsg)
        areas = gdf_m.geometry.area
        mask = areas < float(min_area)
        return [
            Failure(idx, "MinimumArea", f"Area {areas.loc[idx]:.3f} < {min_area}",
                         {"area_m2": float(areas.loc[idx])})
            for idx in gdf.index[mask]
        ]

    def check_holes_total_area_limit(self, gdf: gpd.GeoDataFrame, max_total_hole_area: float = 2.0,
                                     target_epsg: Optional[int] = None) -> List[Failure]:
        """Sum of interior ring areas per feature must be <= max_total_hole_area (m^2)."""
        gdf_m = self._to_metric(gdf, target_epsg=target_epsg or self.work_epsg)

        failures: List[Failure] = []
        for idx, geom in gdf_m.geometry.items():
            if geom is None or geom.is_empty:
                continue
            total_holes = 0.0
            for poly in self._geom_iter_polygon_parts(geom):
                for coords in poly.interiors:
                    total_holes += Polygon(coords).area
            if total_holes > max_total_hole_area:
                failures.append(Failure(
                    idx, "HolesTotalAreaLimit",
                    f"Total hole area {total_holes:.3f} > {max_total_hole_area}",
                    {"holes_area_m2": float(total_holes)}
                ))
        return failures

    def check_holes_area_ratio_limit(self, gdf: gpd.GeoDataFrame, max_ratio: float = 0.20,
                                     target_epsg: Optional[int] = None) -> List[Failure]:
        """Sum of hole areas must be <= max_ratio * polygon area (in metric CRS)."""
        gdf_m = self._to_metric(gdf, target_epsg=target_epsg or self.work_epsg)

        fails: List[Failure] = []
        for idx, geom in gdf_m.geometry.items():
            if geom is None or geom.is_empty:
                continue
            area = float(geom.area)
            if area <= 0:
                continue
            holes = 0.0
            for poly in self._geom_iter_polygon_parts(geom):
                for ring in poly.interiors:
                    holes += Polygon(ring).area
            ratio = holes / area
            if ratio > max_ratio:
                fails.append(Failure(
                    idx, "HolesAreaRatioLimit",
                    f"Holes/Area ratio {ratio:.3f} > {max_ratio}",
                    {"holes_area_m2": holes, "area_m2": area}
                ))
        return fails

    def check_overlapping_buildings(self, gdf: gpd.GeoDataFrame, area_tolerance: float = 0.05,
                                    target_epsg: Optional[int] = None) -> List[Failure]:
        """
        Buildings must not overlap each other beyond tolerance (m^2). Touching is allowed.
        Uses STRtree with query_bulk if available; falls back for older Shapely.
        """
        gdf_m = self._to_metric(gdf, target_epsg=target_epsg or self.work_epsg)

        geoms: List[base.BaseGeometry] = list(gdf_m.geometry.values)
        idxs:  List[Any] = list(gdf_m.index.values)
        tree = STRtree(geoms)

        pairs: List[Tuple[int, int]] = []

        # Try Shapely 2.x (fast)
        try:
            qb = tree.query_bulk(geoms, predicate="intersects")
            for i, j in zip(qb[0], qb[1]):
                i, j = int(i), int(j)
                if i < j:
                    pairs.append((i, j))
        except AttributeError:
            # Shapely 1.x fallback
            wkb_to_idxs: Dict[bytes, List[int]] = {}
            for i, g in enumerate(geoms):
                wkb_to_idxs.setdefault(g.wkb, []).append(i)

            seen: Set[Tuple[int, int]] = set()
            for i, gi in enumerate(geoms):
                nbrs = tree.query(gi)
                if hasattr(nbrs, "dtype") and np.issubdtype(nbrs.dtype, np.integer):
                    for j in map(int, np.asarray(nbrs).tolist()):
                        if i < j:
                            t = (i, j)
                            if t not in seen:
                                seen.add(t); pairs.append(t)
                else:
                    for gj in nbrs:
                        idx_list = wkb_to_idxs.get(gj.wkb, None)
                        if idx_list:
                            for j in idx_list:
                                if i < j:
                                    t = (i, j)
                                    if t not in seen:
                                        seen.add(t); pairs.append(t)
                        else:
                            for j, g2 in enumerate(geoms):
                                if gj.equals(g2) and i < j:
                                    t = (i, j)
                                    if t not in seen:
                                        seen.add(t); pairs.append(t)
                                    break

        overlaps_by_idx: Dict[Any, Dict[str, Any]] = {}
        for i, j in pairs:
            gi, gj = geoms[i], geoms[j]
            if gi.is_empty or gj.is_empty:
                continue
            inter = gi.intersection(gj)
            area = float(getattr(inter, "area", 0.0))
            if area > area_tolerance:
                idx_i, idx_j = idxs[i], idxs[j]
                for a, b in ((idx_i, idx_j), (idx_j, idx_i)):
                    rec = overlaps_by_idx.setdefault(a, {"overlaps_with": [], "overlap_area_sum_m2": 0.0})
                    rec["overlaps_with"].append(b)
                    rec["overlap_area_sum_m2"] += area

        return [
            Failure(idx, "OverlappingBuildings",
                         f"Overlaps {len(info['overlaps_with'])} feature(s); "
                         f"total overlap area {info['overlap_area_sum_m2']:.3f} m^2",
                         info)
            for idx, info in overlaps_by_idx.items()
        ]

    def check_parts_self_overlap(self, gdf: gpd.GeoDataFrame,
                                 target_epsg: Optional[int] = None) -> List[Failure]:
        """Within a MultiPolygon, parts must not overlap each other."""
        gdf_m = self._to_metric(gdf, target_epsg=target_epsg or self.work_epsg)

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
                failures.append(Failure(idx, "PartsSelfOverlap", "MultiPolygon parts overlap"))
        return failures

    def check_sliver_polygons(self, gdf: gpd.GeoDataFrame, min_area_perimeter_ratio: float = 0.2,
                              target_epsg: Optional[int] = None) -> List[Failure]:
        """Flag polygons with area/perimeter ratio below threshold (sliver-like)."""
        gdf_m = self._to_metric(gdf, target_epsg=target_epsg or self.work_epsg)
        areas = gdf_m.geometry.area
        perims = gdf_m.geometry.length.replace(0, pd.NA)
        ratio = areas / perims
        mask = ratio < float(min_area_perimeter_ratio)
        return [
            Failure(idx, "SliverPolygons",
                         f"Area/Perimeter ratio {ratio.loc[idx]:.3f} < {min_area_perimeter_ratio}",
                         {"area_m2": float(areas.loc[idx]), "perimeter_m": float(perims.loc[idx])})
            for idx in gdf.index[mask.fillna(False)]
        ]

# TODO: @ Xiaohan Deng
class FloorPlanChecker:
    def __init__(self, work_epsg: int = 3107):
        self.work_epsg = work_epsg

    # --------------- utils ----------------
    @staticmethod
    def _clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        g = gdf.copy()
        g["geometry"] = g["geometry"].buffer(0)
        return g

    @staticmethod
    def _project(gdf: gpd.GeoDataFrame, epsg: Optional[int], assume_src_epsg: int = 4326):
        """Safe projection: if CRS is missing, set a reasonable default first."""
        if not epsg:
            return gdf
        if gdf.crs is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gdf = gdf.set_crs(epsg=assume_src_epsg)
        return gdf.to_crs(epsg=epsg)


    # --------------- checks (each returns List[Failure]) ----------------
    @staticmethod
    def check_invalid_geometry(gdf: gpd.GeoDataFrame) -> List[Failure]:
        mask = ~gdf.geometry.is_valid
        return [Failure(idx, "GeometryValidity", "Invalid geometry") for idx in gdf.index[mask]]

    @staticmethod
    def check_overlaps(gdf: gpd.GeoDataFrame) -> List[Failure]:
        g = FloorPlanChecker._clean_geoms(gdf)
        sidx = g.sindex
        fails: List[Failure] = []
        for i, geom1 in g.geometry.items():
            for j in sidx.intersection(geom1.bounds):
                if j <= i:
                    continue
                geom2 = g.geometry.iloc[j]
                try:
                    if geom1.intersects(geom2):
                        inter = geom1.intersection(geom2)
                        if hasattr(inter, "area") and inter.area > 0:
                            # create a failure for both features so each row gets flagged
                            info1 = {
                                "other_index": j,
                                "intersection_area": float(inter.area),
                                "block_id": g.iloc[i].get("block_id"),
                                "other_block_id": g.iloc[j].get("block_id"),
                                "block_use": g.iloc[i].get("space_use"),
                                "other_block_use": g.iloc[j].get("space_use"),
                            }
                            info2 = {**info1, "other_index": i}
                            fails.append(Failure(i, "GeometryOverlap", "Overlaps neighbour", info1))
                            fails.append(Failure(j, "GeometryOverlap", "Overlaps neighbour", info2))
                except TopologicalError:
                    continue
        return fails

    @staticmethod
    def check_nulls(gdf: gpd.GeoDataFrame) -> List[Failure]:
        counts = gdf.isnull().sum()
        fails: List[Failure] = []
        for col, n in counts[counts > 0].items():
            fails.append(Failure(None, "NullValues", f"{int(n)} NULLs in column '{col}'",
                                 {"column": col, "null_count": int(n)}))
        return fails

    @staticmethod
    def check_negative_floor_space(gdf: gpd.GeoDataFrame, col: str = "floor_space") -> List[Failure]:
        if col not in gdf.columns:
            return []
        mask = gdf[col] < 0
        return [Failure(idx, "NegativeFloorSpace", f"{col} is negative", {col: float(gdf.loc[idx, col])})
                for idx in gdf.index[mask]]

    @staticmethod
    def check_total_consistency(gdf: gpd.GeoDataFrame, total_col: str = "total") -> List[Failure]:
        if total_col not in gdf.columns:
            return []
        tmp = gdf.copy()
        comp_cols = [c for c in tmp.columns if c not in ["geometry", total_col]]
        tmp["__component_sum__"] = tmp[comp_cols].sum(axis=1, skipna=True, numeric_only=True)
        mask = tmp["__component_sum__"] > tmp[total_col]
        return [
            Failure(idx, "TotalConsistency",
                    "Sum of components exceeds total",
                    {"component_sum": float(tmp.loc[idx, "__component_sum__"]),
                     "total": float(tmp.loc[idx, total_col])})
            for idx in tmp.index[mask]
        ]

    @staticmethod
    def check_crs(gdf: gpd.GeoDataFrame) -> List[Failure]:
        if gdf.crs is None:
            return [Failure(None, "CRSCheck", "CRS is missing (None)")]
        return []

    @staticmethod
    def check_conflict_blocks(
        gdf: gpd.GeoDataFrame, block_col: str = "block_id", use_col: str = "space_use", threshold: int = 3
    ) -> List[Failure]:
        if block_col not in gdf.columns or use_col not in gdf.columns:
            return []
        counts = gdf.groupby(block_col)[use_col].nunique()
        conflicts = counts[counts > threshold]
        return [
            Failure(bid, "ConflictBlockUses",
                    f"{int(k)} distinct uses (> {threshold})", {"num_uses": int(k)})
            for bid, k in conflicts.items()
        ]

    # --------------- runner & helpers ----------------
    def run_all(
        self,
        gdf: gpd.GeoDataFrame,
        floor_space_col: str = "floor_space",
        total_col: str = "total",
        block_col: str = "block_id",
        use_col: str = "space_use",
        conflict_threshold: int = 3,
    ) -> List[Failure]:
        fails: List[Failure] = []
        fails += self.check_invalid_geometry(gdf)
        fails += self.check_overlaps(gdf)
        fails += self.check_nulls(gdf)
        fails += self.check_negative_floor_space(gdf, col=floor_space_col)
        fails += self.check_total_consistency(gdf, total_col=total_col)
        fails += self.check_crs(gdf)
        fails += self.check_conflict_blocks(
            gdf, block_col=block_col, use_col=use_col, threshold=conflict_threshold
        )
        return fails

    @staticmethod
    def inspect_overlaps(gdf: gpd.GeoDataFrame, overlap_failures: List[Failure]) -> pd.DataFrame:
        """
        Convert 'GeometryOverlap' failures into a readable table:
        row indices + block_id/space_use for both sides of the pair.
        """
        pairs: List[Tuple[int, int]] = []
        for f in overlap_failures:
            if f.error_type != "GeometryOverlap":
                continue
            i = f.index
            j = f.extras.get("other_index")
            if isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)):
                a, b = (int(i), int(j))
                if a < b:
                    pairs.append((a, b))
        # deduplicate pairs
        pairs = sorted(set(pairs))
        if not pairs:
            return pd.DataFrame(columns=["row1", "row2", "block1_id", "block1_use", "block2_id", "block2_use"])

        df = pd.DataFrame(pairs, columns=["row1", "row2"])
        df["block1_id"] = gdf.loc[df["row1"], "block_id"].values
        df["block1_use"] = gdf.loc[df["row1"], "space_use"].values
        df["block2_id"] = gdf.loc[df["row2"], "block_id"].values
        df["block2_use"] = gdf.loc[df["row2"], "space_use"].values
        return df

    @staticmethod
    def get_null_counts(gdf: gpd.GeoDataFrame) -> pd.Series:
        """Series of NULL counts per column (for plotting)."""
        return gdf.isnull().sum()

# TODO: @ Xuanyi
def _fix_geom(geom):
    if geom is None:
        return geom
    try:
        if not geom.is_valid:
            if _make_valid is not None:
                geom = _make_valid(geom)
            else:
                geom = geom.buffer(0)
    except Exception:
        pass
    return geom


@dataclass
class BlockChecker:
    """Geometry-quality checks over a GeoDataFrame; each check returns List[Failure]."""
    work_epsg: int = 3107  # projected CRS for metric ops

    # ------------------------------
    # Core utilities (GeoDataFrame in / out)
    # ------------------------------
    @staticmethod
    def _clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Copy, fix simple invalids, explode multipart features."""
        g = gdf.copy()
        g["geometry"] = g["geometry"].apply(_fix_geom)
        g = g.explode(index_parts=False, ignore_index=True)
        return g

    @staticmethod
    def _project(gdf: gpd.GeoDataFrame, epsg: Optional[int], assume_src_epsg: int = 4326):
        """Safe projection: if CRS is missing, set a reasonable default first."""
        if not epsg:
            return gdf
        if gdf.crs is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gdf = gdf.set_crs(epsg=assume_src_epsg)
        return gdf.to_crs(epsg=epsg)

    # ------------------------------
    # 1) Triangulation closure
    # ------------------------------
    @staticmethod
    def _triang_area_diff(geom) -> float:
        try:
            tris = triangulate(geom)
            tri_area = sum(t.area for t in tris)
            return float(abs(tri_area - geom.area))
        except Exception:
            return float("nan")

    def check_triangulation(self, gdf: gpd.GeoDataFrame, tol: float = 1e-1) -> List[Failure]:
        """Fail if sum of triangulated areas differs from polygon area by more than tol."""
        g = self._clean_geoms(gdf)
        fails: List[Failure] = []
        for i, geom in enumerate(g.geometry):
            diff = self._triang_area_diff(geom)
            if not (abs(diff) < float(tol)):
                fails.append(Failure(
                    index=i,
                    error_type="TriangulationNotClosed",
                    message=f"Triangulated area differs by {diff:.6f} (tol {tol}).",
                    extras={"triang_error": float(diff), "tolerance": float(tol)}
                ))
        return fails

    # ------------------------------
    # 2) Short edges
    # ------------------------------
    @staticmethod
    def _iter_segments(geom):
        def segs(coords):
            pts = list(coords)
            for k in range(1, len(pts)):
                yield (pts[k-1], pts[k])

        if isinstance(geom, Polygon):
            for s in segs(list(geom.exterior.coords)):
                yield s
            for r in geom.interiors:
                for s in segs(list(r.coords)):
                    yield s
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                for s in BlockChecker._iter_segments(p):
                    yield s

    @staticmethod
    def _iter_segments_exterior(geom):
        """Yield only EXTERIOR segments (to match the notebook)."""
        def segs(coords):
            pts = list(coords)
            for k in range(1, len(pts)):
                yield (pts[k-1], pts[k])
        if isinstance(geom, Polygon):
            for s in segs(list(geom.exterior.coords)):
                yield s
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                for s in BlockChecker._iter_segments_exterior(p):
                    yield s

    @staticmethod
    def _extract_exterior_edges(geom):
        """Undirected EXTERIOR edges only (normalised so AB==BA)."""
        edges = []
        for a, b in BlockChecker._iter_segments_exterior(geom):
            edges.append(BlockChecker._norm_edge(a, b))
        return edges

    @staticmethod
    def check_not_simple(gdf: gpd.GeoDataFrame) -> List[Failure]:
        mask = ~gdf.geometry.is_simple
        return [Failure(i, "NotSimple", "Geometry is not simple.") for i in gdf.index[mask]]

    @staticmethod
    def _min_edge_length(geom) -> float:
        mins = []
        try:
            for (x1, y1), (x2, y2) in BlockChecker._iter_segments(geom):
                mins.append(math.hypot(x2 - x1, y2 - y1))
        except Exception:
            return float("nan")
        return float(min(mins)) if mins else float("nan")

    def check_short_edges(self, gdf: gpd.GeoDataFrame, min_len: float) -> List[Failure]:
        need_project = (gdf.crs is None) or getattr(gdf.crs, "is_geographic", False)
        g = self._clean_geoms(gdf)
        g = self._project(g, self.work_epsg if need_project else None)

        fails: List[Failure] = []
        for i, geom in enumerate(g.geometry):
            mins = []
            try:
                for (x1, y1), (x2, y2) in BlockChecker._iter_segments_exterior(geom):
                    mins.append(math.hypot(x2 - x1, y2 - y1))
            except Exception:
                continue
            if mins:
                m = float(min(mins))  
                if m < float(min_len):
                    fails.append(Failure(i, "ShortEdge", f"Shortest edge {m:.6f} < {min_len}.",
                                        {"shortest_edge": m, "threshold": float(min_len)}))
        return fails

    # ------------------------------
    # 3) Exterior ring orientation (expect CCW)
    # ------------------------------
    @staticmethod
    def _outer_is_clockwise(geom) -> bool:
        try:
            if isinstance(geom, (Polygon, MultiPolygon)):
                poly = geom if isinstance(geom, Polygon) else list(geom.geoms)[0]
                ring = LinearRing(poly.exterior.coords)
                return not ring.is_ccw
        except Exception:
            pass
        return False

    from shapely.geometry.polygon import orient

    def check_wrong_orientation(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        g = self._clean_geoms(gdf)
        fails: List[Failure] = []
        for i, geom in enumerate(g.geometry):
            try:
                if isinstance(geom, (Polygon, MultiPolygon)):
                    # Re-orient exterior CCW; if geometry changes, the original was "wrong"
                    fixed = orient(geom, sign=1.0)
                    if not geom.equals_exact(fixed, 1e-8):
                        fails.append(Failure(i, "WrongOrientation", "Outer ring is clockwise (expected CCW)."))
            except Exception:
                continue
        return fails


    # ------------------------------
    # 4) Non-manifold edges (>2 polygons share an undirected edge)
    # ------------------------------
    @staticmethod
    def _norm_edge(p1, p2):
        return (p1, p2) if p1 <= p2 else (p2, p1)

    @staticmethod
    def _extract_edges(geom):
        edges = []
        for a, b in BlockChecker._iter_segments(geom):
            edges.append(BlockChecker._norm_edge(a, b))
        return edges

    @staticmethod
    def _has_nonmanifold(local_edges, edge_counter: Dict) -> bool:
        for e in local_edges:
            if edge_counter.get(e, 0) > 2:
                return True
        return False

    def check_nonmanifold_edges(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        g = self._clean_geoms(gdf)

        all_edges: List[Tuple] = []
        per_geom: List[List[Tuple]] = []
        for geom in g.geometry:
            e = BlockChecker._extract_exterior_edges(geom)
            per_geom.append(e)
            all_edges.extend(e)

        counts: Dict = {}
        for e in all_edges:
            counts[e] = counts.get(e, 0) + 1

        fails: List[Failure] = []
        for i, edges in enumerate(per_geom):
            if any(counts.get(e, 0) > 2 for e in edges):
                fails.append(Failure(i, "NonManifoldEdge", "Edge shared by more than two polygons."))
        return fails


    # ------------------------------
    # 5) Minimum interior angle
    # ------------------------------
    @staticmethod
    def _ring_min_angle(coords) -> float:
        def ang(a, b, c):
            v1 = (a[0] - b[0], a[1] - b[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 == 0 or n2 == 0:
                return float("inf")
            cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
            d = math.degrees(math.acos(cosang))
            return 360 - d if d > 180 else d

        m = float("inf")
        pts = list(coords)
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        for k in range(1, len(pts) - 1):
            a, b, c = pts[k - 1], pts[k], pts[k + 1]
            m = min(m, ang(a, b, c))
        return m

    @staticmethod
    def _compute_min_angle(geom) -> float:
        try:
            if isinstance(geom, Polygon):
                mins = [BlockChecker._ring_min_angle(list(geom.exterior.coords))]
                for r in geom.interiors:
                    mins.append(BlockChecker._ring_min_angle(list(r.coords)))
                return float(min(mins))
            elif isinstance(geom, MultiPolygon):
                vals = [BlockChecker._compute_min_angle(p) for p in geom.geoms]
                return float(min(vals)) if vals else float("nan")
        except Exception:
            pass
        return float("nan")

    def check_min_angle(self, gdf: gpd.GeoDataFrame, min_deg: float, include_interiors: bool = False) -> List[Failure]:
        g = self._clean_geoms(gdf)

        def ring_min_angle(coords) -> float:
            def ang(a, b, c):
                v1 = (a[0] - b[0], a[1] - b[1])
                v2 = (c[0] - b[0], c[1] - b[1])
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                n1 = math.hypot(*v1); n2 = math.hypot(*v2)
                if n1 == 0 or n2 == 0: 
                    return float("inf")
                cosang = max(-1.0, min(1.0, dot/(n1*n2)))
                d = math.degrees(math.acos(cosang))
                return 360 - d if d > 180 else d

            m = float("inf")
            pts = list(coords)
            if pts and pts[0] != pts[-1]:
                pts.append(pts[0])
            for k in range(1, len(pts)-1):
                m = min(m, ang(pts[k-1], pts[k], pts[k+1]))
            return m

        fails: List[Failure] = []
        for i, geom in enumerate(g.geometry):
            try:
                if isinstance(geom, Polygon):
                    mins = [ring_min_angle(list(geom.exterior.coords))]
                    if include_interiors:
                        for r in geom.interiors:
                            mins.append(ring_min_angle(list(r.coords)))
                elif isinstance(geom, MultiPolygon):
                    vals = []
                    for p in geom.geoms:
                        vals.append(ring_min_angle(list(p.exterior.coords)))
                        if include_interiors:
                            for r in p.interiors:
                                vals.append(ring_min_angle(list(r.coords)))
                    mins = [min(vals)] if vals else []
                else:
                    mins = []
                if mins:
                    val = float(min(mins))
                    if val < float(min_deg):
                        fails.append(Failure(i, "AcuteAngle", f"Minimum interior angle {val:.4f}° < {min_deg}°.",
                                            {"min_angle_deg": val, "threshold_deg": float(min_deg)}))
            except Exception:
                continue
        return fails


    # ------------------------------
    # 6) Coordinate precision
    # ------------------------------
    @staticmethod
    def _max_decimals_in_coords(geom) -> int:
        def count_dec(x: float) -> int:
            s = f"{x:.12f}".rstrip("0").rstrip(".")
            return len(s.split(".")[1]) if "." in s else 0

        mx = 0
        def scan(coords):
            nonlocal mx
            for x, y in coords:
                mx = max(mx, count_dec(x), count_dec(y))

        try:
            if isinstance(geom, Polygon):
                scan(list(geom.exterior.coords))
                for r in geom.interiors:
                    scan(list(r.coords))
            elif isinstance(geom, MultiPolygon):
                for p in geom.geoms:
                    mx = max(mx, BlockChecker._max_decimals_in_coords(p))
        except Exception:
            pass
        return mx

    def check_excessive_precision(self, gdf: gpd.GeoDataFrame, max_decimals: int = 3) -> List[Failure]:
        if (gdf.crs is None) or getattr(gdf.crs, "is_geographic", False):
            return []

        g = self._clean_geoms(gdf)  
        tol = 1e-9  
        def quant_ok(v: float) -> bool:
            return abs(v - round(v, int(max_decimals))) < tol

        fails: List[Failure] = []
        for i, geom in enumerate(g.geometry):
            try:
                polys = [geom] if isinstance(geom, Polygon) else (list(geom.geoms) if isinstance(geom, MultiPolygon) else [])
                bad = total = 0
                for poly in polys:
                    for x, y in poly.exterior.coords:
                        total += 1
                        if not (quant_ok(x) and quant_ok(y)):
                            bad += 1
                if total > 0 and (bad / total) > 0.10:
                    fails.append(Failure(
                        i, "ExcessivePrecision",
                        f"> {max_decimals} decimals for {bad}/{total} exterior vertices",
                        {"limit": int(max_decimals), "bad_ratio": bad / total}
                    ))
            except Exception:
                continue
        return fails

    # ------------------------------
    # 7) Outside bounding box
    # ------------------------------
    @staticmethod
    def _outside_bbox_geom(geom, bbox: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        (minx, miny), (maxx, maxy) = bbox
        try:
            return not geom.within(box(minx, miny, maxx, maxy))
        except Exception:
            return False

    def check_outside_bbox(self, gdf: gpd.GeoDataFrame, bbox: Tuple[Tuple[float, float], Tuple[float, float]]) -> List[Failure]:
        """Fail features that lie outside the supplied bounding box ((minx,miny),(maxx,maxy))."""
        g = self._clean_geoms(gdf)
        return [
            Failure(i, "OutsideBBox", "Geometry lies outside the bounding box.", {"bbox": bbox})
            for i, geom in enumerate(g.geometry) if self._outside_bbox_geom(geom, bbox)
        ]

    # ------------------------------
    # 8) Pairwise overlaps / invalids / touches
    # ------------------------------
    def check_overlaps(self, gdf: gpd.GeoDataFrame, min_area: float = 0.0) -> List[Failure]:
        """Fail for each overlapping pair; include overlap area/ratio on each side."""
        g = self._clean_geoms(gdf)
        sidx = g.sindex
        fails: List[Failure] = []
        for i, geom1 in enumerate(g.geometry):
            for j in sidx.intersection(geom1.bounds):
                if j <= i:
                    continue
                geom2 = g.geometry.iloc[j]
                try:
                    if geom1.intersects(geom2):
                        inter = geom1.intersection(geom2)
                        area = float(getattr(inter, "area", 0.0))
                        if area > float(min_area):
                            a1 = float(geom1.area) if geom1.area else float("nan")
                            a2 = float(geom2.area) if geom2.area else float("nan")
                            info1 = {"other_index": j, "overlap_area": area,
                                     "share_of_row": area / a1 if a1 and a1 > 0 else float("nan"),
                                     "geometry": inter}
                            info2 = {"other_index": i, "overlap_area": area,
                                     "share_of_row": area / a2 if a2 and a2 > 0 else float("nan"),
                                     "geometry": inter}
                            fails.append(Failure(i, "Overlap", "Overlaps neighbour.", info1))
                            fails.append(Failure(j, "Overlap", "Overlaps neighbour.", info2))
                except TopologicalError:
                    continue
        return fails

    @staticmethod
    def check_invalid_geometry(gdf: gpd.GeoDataFrame) -> List[Failure]:
        """Fail each invalid geometry (reason can be added if desired)."""
        mask = ~gdf.geometry.is_valid
        return [Failure(i, "InvalidGeometry", "Geometry is invalid.") for i in gdf.index[mask]]

    def check_touches(self, gdf: gpd.GeoDataFrame) -> List[Failure]:
        """Info-level: report touching neighbours (shared boundary)."""
        g = self._clean_geoms(gdf)
        sidx = g.sindex
        fails: List[Failure] = []
        for i, a in enumerate(g.geometry):
            for j in sidx.intersection(a.bounds):
                if j <= i:
                    continue
                b = g.geometry.iloc[j]
                if a.touches(b):
                    shared = a.boundary.intersection(b.boundary)
                    if not shared.is_empty:
                        infoA = {
                            "other_index": j,
                            "shared_length": float(getattr(shared, "length", 0.0)),
                            "geometry": shared
                        }
                        infoB = {
                            "other_index": i,
                            "shared_length": float(getattr(shared, "length", 0.0)),
                            "geometry": shared
                        }
                        fails.append(Failure(i, "Touches", "Shares boundary with neighbour.", infoA))
                        fails.append(Failure(j, "Touches", "Shares boundary with neighbour.", infoB))
        return fails

if __name__ == "__main__":
    # Suburbs
    suburbs = gpd.read_file("../../data/sa-govt-dpti-sa-suburb-boundaries-aug-2018-na.json")
    sc = SuburbChecker()
    suburb_fails = []
    suburb_fails += sc.check_geometries(suburbs)
    suburb_fails += sc.check_overlaps(suburbs)
    suburb_fails += sc.check_slivers(suburbs, min_area_m2=30)
    report = failures_to_frame(suburb_fails)
    print(report)

    # Roads TODO @ Yaojin
    roads = ...
    ric = RoadIsolationChecker(roads)
    fails_roads = []
    fails_roads += ric.compute_candidate_isolated()
    fails_roads += ric.compute_final_isolated(strict=False, buffer_eps=0.0)
    roads_report = failures_to_frame(fails_roads)

    # building TODO @ lei
    gdf = gpd.read_file("../../case_study/melbourne_building/geoscape-geoscape-melbourne-buildings-jun22-na.geojson")
    validator = BuildingChecker(work_epsg=7855)
    # Example: run a few checks and turn into a table
    fails = []
    fails += validator.check_geometry_not_empty(gdf)
    fails += validator.check_geometry_valid(gdf)
    fails += validator.check_building_polygons_only(gdf)
    fails += validator.check_overlapping_buildings(gdf, area_tolerance=0.05)
    fails += validator.check_holes_area_ratio_limit(gdf, max_ratio=0.20, target_epsg=7855)
    df_report = validator.failures_to_frame(fails)
    print(df_report)

    # floor plan TODO @ han
    gdf = gpd.read_file("case_study/melbourne_city/geoscape-geoscape-melbourne-buildings-jun22-na.geojson")
    checker = FloorPlanChecker()
    all_fails = checker.run_all(gdf)
    report = failures_to_frame(all_fails)
    # Overlap details table
    overlap_table = FloorPlanChecker.inspect_overlaps(gdf,
                                                  [f for f in all_fails if f.error_type == "GeometryOverlap"])
    # Plot missing fields
    null_counts = FloorPlanChecker.get_null_counts(gdf)

    # City Block TODO @ Xuanyi
    gdf = gpd.read_file(r"D:\unimelb_MSc_course\Data Science Project MAST901067\project\Unimelb-DS-Project-G19\data\vic-govt-det-vic-det-school-zone-secondary-year9-2020-na.xml")
    chk = BlockChecker()
    fails: List[Failure] = []
    fails += chk.check_invalid_geometry(gdf)      
    fails += chk.check_overlaps(gdf, min_area=0.01) 
    fails += chk.check_short_edges(gdf, min_len=1.0)
    fails += chk.check_wrong_orientation(gdf)    
    fails += chk.check_nonmanifold_edges(gdf)      
    fails += chk.check_min_angle(gdf, min_deg=5.0)  
    fails += chk.check_excessive_precision(gdf, max_decimals=3)  
    minx, miny, maxx, maxy = (140.9, -39.2, 150.0, -33.9)
    fails += chk.check_outside_bbox(gdf, bbox=((minx, miny), (maxx, maxy)))
    fails += chk.check_touches(gdf)  
    report = failures_to_frame(fails)
    print(report)

