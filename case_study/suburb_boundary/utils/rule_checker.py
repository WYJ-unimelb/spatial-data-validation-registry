import geopandas as gpd
import pandas as pd
import math
from shapely.geometry import Polygon, MultiPolygon
from shapely.errors import TopologicalError
from shapely.ops import unary_union
from difflib import get_close_matches


class SuburbChecker:
    def __init__(self, work_epsg=3107):
        self.work_epsg = work_epsg

    @staticmethod
    def _clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # Make a copy, fix simple invalids, explode multiparts for per-part checks
        g = gdf.copy()
        g['geometry'] = g['geometry'].buffer(0)
        g = g.explode(index_parts=False, ignore_index=True)
        return g

    @staticmethod
    def _project(gdf: gpd.GeoDataFrame, epsg: int | None):
        return gdf if not epsg else gdf.to_crs(epsg=epsg)

    @staticmethod
    def check_overlaps(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        g = SuburbChecker._clean_geoms(gdf)
        sidx = g.sindex
        overlaps = []
        for i, geom1 in enumerate(g.geometry):
            for j in sidx.intersection(geom1.bounds):
                if j <= i:
                    continue
                geom2 = g.geometry.iloc[j]
                try:
                    if geom1.intersects(geom2):
                        inter = geom1.intersection(geom2)
                        if inter.area > 0:
                            overlaps.append({
                                'i': i,
                                'j': j,
                                'suburb_1': g.iloc[i].get('suburb'),
                                'suburb_2': g.iloc[j].get('suburb'),
                                'intersection_area': inter.area
                            })
                except TopologicalError:
                    continue
        print(f"{len(overlaps)} overlaps found.")
        return pd.DataFrame(overlaps)

    @staticmethod
    def check_areas(gdf: gpd.GeoDataFrame, reproject_epsg=3107):
        g = SuburbChecker._project(gdf, reproject_epsg)
        invalid = g[g.geometry.area <= 0]
        print(f"{len(invalid)} suburbs with area <= 0 found.")
        return invalid

    @staticmethod
    def check_geometries(gdf: gpd.GeoDataFrame):
        invalid = gdf[~gdf.geometry.is_valid]
        print(f"{len(invalid)} invalid geometries found.")
        return invalid

    @staticmethod
    def check_gaps(gdf: gpd.GeoDataFrame):
        g = SuburbChecker._clean_geoms(gdf)
        dissolved = g.dissolve().geometry.iloc[0]
        merged = unary_union(g.geometry)
        gaps = dissolved.symmetric_difference(merged)
        if gaps.is_empty:
            print("✅ No topological gaps found.")
            return None
        print("❗ Potential topological gaps detected.")
        return gpd.GeoDataFrame(geometry=[gaps], crs=g.crs)

    @staticmethod
    def check_semantics(gdf: gpd.GeoDataFrame):
        dup_counts = gdf.groupby("suburb").size()
        duplicate_suburbs = dup_counts[dup_counts > 1]
        missing_postcode = gdf[gdf['postcode'].isna()]
        invalid_dates = gdf[pd.to_datetime(gdf['legalstartdate'], errors='coerce').isna()]
        if len(duplicate_suburbs):
            print(f"{len(duplicate_suburbs)} duplicate suburb names found.")
        if len(missing_postcode):
            print(f"{len(missing_postcode)} suburbs with missing postcode.")
        if len(invalid_dates):
            print(f"{len(invalid_dates)} suburbs with invalid legal start date.")
        return duplicate_suburbs, missing_postcode, invalid_dates

    @staticmethod
    def check_misaligned_boundaries(gdf: gpd.GeoDataFrame, epsg=3107, tolerance=0.5):
        g = SuburbChecker._project(SuburbChecker._clean_geoms(gdf), epsg)
        sidx = g.sindex
        inconsistent = []
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
                    inconsistent.append(shared)
        print(f"{len(inconsistent)} inconsistent boundaries detected.")
        return gpd.GeoDataFrame(geometry=inconsistent, crs=g.crs)

    @staticmethod
    def check_slivers(gdf: gpd.GeoDataFrame, min_area_m2=50, epsg=3107):
        """Flag tiny 'sliver' polygons likely caused by digitising artefacts."""
        g = SuburbChecker._project(SuburbChecker._clean_geoms(gdf), epsg)
        slivers = g[g.area < min_area_m2]
        print(f"{len(slivers)} sliver polygons (<{min_area_m2} m²).")
        return slivers

    @staticmethod
    def check_holes(gdf: gpd.GeoDataFrame, max_hole_area_m2=25, epsg=3107):
        """Find small interior holes; often unintended."""
        g = SuburbChecker._project(SuburbChecker._clean_geoms(gdf), epsg)
        records = []
        for i, geom in enumerate(g.geometry):
            if isinstance(geom, (Polygon,)):
                holes = list(geom.interiors)
                for ring in holes:
                    hole_poly = Polygon(ring)
                    if hole_poly.area <= max_hole_area_m2:
                        records.append({'index': i, 'hole_area': hole_poly.area, 'geometry': hole_poly})
            elif isinstance(geom, MultiPolygon):
                for part in geom.geoms:
                    for ring in part.interiors:
                        hole_poly = Polygon(ring)
                        if hole_poly.area <= max_hole_area_m2:
                            records.append({'index': i, 'hole_area': hole_poly.area, 'geometry': hole_poly})
        print(f"{len(records)} small holes (≤{max_hole_area_m2} m²).")
        return gpd.GeoDataFrame(records, geometry='geometry', crs=g.crs)

    @staticmethod
    def check_non_contiguous_parts(gdf: gpd.GeoDataFrame, key='suburb'):
        """Detect suburbs represented by multiple disconnected parts."""
        g = SuburbChecker._clean_geoms(gdf)
        parts = (g
                 .explode(index_parts=False, ignore_index=True)
                 .assign(_ones=1)
                 .dissolve(by=[key], as_index=False, aggfunc={'_ones':'count'}))
        multi = parts[parts['_ones'] > 1][[key, '_ones']]
        print(f"{len(multi)} suburbs are non-contiguous (multiple parts).")
        return multi.rename(columns={'_ones': 'num_parts'})

    @staticmethod
    def check_shared_boundary_length(gdf: gpd.GeoDataFrame, min_shared_len_m=5, epsg=3107):
        """Where neighbours touch, flag pairs whose shared boundary is suspiciously short."""
        g = SuburbChecker._project(SuburbChecker._clean_geoms(gdf), epsg)
        sidx = g.sindex
        rows = []
        for i, a in enumerate(g.geometry):
            for j in sidx.intersection(a.bounds):
                if j <= i:
                    continue
                b = g.geometry.iloc[j]
                if a.touches(b):
                    shared = a.boundary.intersection(b.boundary)
                    if not shared.is_empty and shared.length < min_shared_len_m:
                        rows.append({
                            'i': i, 'j': j,
                            'suburb_1': g.iloc[i].get('suburb'),
                            'suburb_2': g.iloc[j].get('suburb'),
                            'shared_length_m': shared.length,
                            'geometry': shared
                        })
        print(f"{len(rows)} neighbour pairs with very short shared boundary (<{min_shared_len_m} m).")
        return gpd.GeoDataFrame(rows, geometry='geometry', crs=g.crs)

    @staticmethod
    def check_enclaves(gdf: gpd.GeoDataFrame):
        """Detect polygons that are entirely inside others with a different suburb (islands)."""
        g = SuburbChecker._clean_geoms(gdf)
        sidx = g.sindex
        rows = []
        for i, a in enumerate(g.geometry):
            for j in sidx.intersection(a.bounds):
                if j == i:
                    continue
                b = g.geometry.iloc[j]
                if a.within(b):
                    if g.iloc[i].get('suburb') != g.iloc[j].get('suburb'):
                        rows.append({'inner_idx': i, 'outer_idx': j,
                                     'inner_suburb': g.iloc[i].get('suburb'),
                                     'outer_suburb': g.iloc[j].get('suburb')})
        print(f"{len(rows)} enclave/exclave situations.")
        return pd.DataFrame(rows)

    @staticmethod
    def check_crs_and_units(gdf: gpd.GeoDataFrame, must_be_projected=True):
        """Ensure a projected CRS for metric operations."""
        crs = gdf.crs
        ok = crs is not None and getattr(crs, "is_projected", False)
        if must_be_projected and not ok:
            print("❗ Data are not in a projected CRS. Reproject before area/length checks.")
        else:
            print("✅ CRS is projected." if ok else "ℹ️ No CRS set.")
        return crs

    @staticmethod
    def check_centroid_inside(gdf: gpd.GeoDataFrame):
        """Centroid can be outside in weird shapes/multiparts; point_on_surface is guaranteed inside."""
        g = SuburbChecker._clean_geoms(gdf)
        outside = []
        for i, geom in enumerate(g.geometry):
            c = geom.centroid
            if not geom.contains(c):
                pos = geom.representative_point()
                outside.append({'index': i, 'centroid_inside': False,
                                'centroid': c, 'point_on_surface': pos})
        print(f"{len(outside)} features with centroid outside; consider using point_on_surface.")
        return gpd.GeoDataFrame(outside, geometry='point_on_surface', crs=g.crs) if outside else None

    @staticmethod
    def check_bbox_within_extent(gdf: gpd.GeoDataFrame, state_extent: gpd.GeoSeries):
        """Ensure each suburb lies within an official state/territory extent (e.g., admin boundary)."""
        g = SuburbChecker._clean_geoms(gdf)
        extent_union = unary_union(state_extent)
        outside = g[~g.geometry.within(extent_union)]
        print(f"{len(outside)} features outside provided extent.")
        return outside

    @staticmethod
    def check_attribute_consistency_on_borders(gdf: gpd.GeoDataFrame, attrs=('state',)):
        """
        For neighbours that touch, ensure certain attributes match (e.g., both in same 'state').
        Useful when suburbs should not cross administrative boundaries.
        """
        g = SuburbChecker._clean_geoms(gdf)
        sidx = g.sindex
        rows = []
        for i, a in enumerate(g.geometry):
            for j in sidx.intersection(a.bounds):
                if j <= i:
                    continue
                b = g.geometry.iloc[j]
                if a.touches(b):
                    for attr in attrs:
                        if g.iloc[i].get(attr) != g.iloc[j].get(attr):
                            rows.append({'i': i, 'j': j, 'attr': attr,
                                         'val_i': g.iloc[i].get(attr),
                                         'val_j': g.iloc[j].get(attr)})
        print(f"{len(rows)} neighbour attribute mismatches on {attrs}.")
        return pd.DataFrame(rows)

    @staticmethod
    def check_name_typos(gdf: gpd.GeoDataFrame, name_col='suburb', cutoff=0.88):
        """
        Fuzzy-flag near-duplicate suburb names (e.g., 'St Kilda' vs 'Saint Kilda').
        Uses difflib (no extra deps). Adjust cutoff (0..1).
        """
        names = sorted(set(map(str, gdf[name_col].fillna(''))))
        flagged = []
        for i, n in enumerate(names):
            matches = get_close_matches(n, names[i+1:], n=5, cutoff=cutoff)
            for m in matches:
                flagged.append({'name_a': n, 'name_b': m})
        print(f"{len(flagged)} near-duplicate name pairs (cutoff={cutoff}).")
        return pd.DataFrame(flagged)

    @staticmethod
    def check_spikes_and_bows(gdf: gpd.GeoDataFrame, min_angle_deg=5):
        """
        Heuristic: detect very acute angles along polygon boundaries (spikes/bows).
        This is a geometry-quality smell that often hints at digitising errors.
        """
        g = SuburbChecker._clean_geoms(gdf)
        rows = []
        for i, geom in enumerate(g.geometry):
            if isinstance(geom, Polygon):
                coords = list(geom.exterior.coords)
                for k in range(1, len(coords)-1):
                    a, b, c = coords[k-1], coords[k], coords[k+1]
                    v1 = (a[0]-b[0], a[1]-b[1])
                    v2 = (c[0]-b[0], c[1]-b[1])
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    norm = (v1[0]**2+v1[1]**2)**0.5 * (v2[0]**2+v2[1]**2)**0.5
                    if norm == 0:
                        continue
                    ang = math.degrees(math.acos(max(-1,min(1,dot/norm))))
                    if ang < min_angle_deg:
                        rows.append({'index': i, 'angle_deg': ang, 'vertex_idx': k})
        print(f"{len(rows)} acute boundary angles (<{min_angle_deg}°) flagged.")
        return pd.DataFrame(rows)

    @staticmethod
    def check_duplicate_geometries(gdf: gpd.GeoDataFrame):
        """Exact-duplicate geometries (after cleaning)."""
        g = SuburbChecker._clean_geoms(gdf)
        wkb = g.geometry.apply(lambda x: x.wkb)
        dup_idx = wkb[wkb.duplicated(keep=False)].index
        dups = g.loc[dup_idx]
        print(f"{dups.shape[0]} exact-duplicate geometries.")
        return dups

    @staticmethod
    def check_postcode_geometry_consistency(gdf: gpd.GeoDataFrame, key='suburb', code='postcode'):
        """
        Flag cases where the same suburb name maps to multiple postcodes or
        a postcode maps to multiple non-contiguous suburb parts (data hygiene).
        """
        # one suburb -> many postcodes?
        sub_to_pc = gdf.groupby(key)[code].nunique().reset_index()
        s_multi = sub_to_pc[sub_to_pc[code] > 1]
        # one postcode -> many suburb names?
        pc_to_sub = gdf.groupby(code)[key].nunique().reset_index()
        p_multi = pc_to_sub[pc_to_sub[key] > 1]
        print(f"{len(s_multi)} suburbs with multiple postcodes; {len(p_multi)} postcodes spanning multiple suburbs.")
        return s_multi, p_multi
