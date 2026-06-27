import geopandas as gpd
import pandas as pd
from shapely.strtree import STRtree

def check_geometry_validity(gdf):
    failed = gdf[~gdf.is_valid]
    return failed, "Geometry is not valid"

def check_overlap_buildings(gdf):
    geoms = gdf.geometry
    tree = STRtree(geoms)
    overlaps = []
    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            if i != j and geom.intersects(geoms[j]) and not geom.touches(geoms[j]):
                overlaps.append(i)
                break
    failed = gdf.iloc[overlaps]
    return failed, "Overlapping buildings detected"

def validate(gdf, dtype):
    failed_list = []
    if dtype == "building":
        rules = [check_geometry_validity, check_overlap_buildings]
    else:
        rules = [check_geometry_validity]

    for rule_func in rules:
        failed, reason = rule_func(gdf)
        failed["error_type"] = reason
        failed_list.append(failed)

    return pd.concat(failed_list)
