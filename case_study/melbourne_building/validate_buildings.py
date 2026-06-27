import geopandas as gpd
from rules_building_modular import (
    default_building_rules, ValidationEngine,
    HolesAreaRatioLimit
)

gdf = gpd.read_file("sample_data/geoscape-geoscape-melbourne-buildings-jun22-na.geojson")

base = default_building_rules(
    melb_target_epsg=7855,
    min_area_m2=5.0,
    overlap_tol_m2=0.1,
    max_total_hole_area_m2=1e12,      
    sliver_ratio_min=0.15,
    id_field="bld_pid",
    required_fields=["bld_pid", "state", "area", "mb_code"],
    unique_fields=["ogc_fid"]
)


rules = base + [
    HolesAreaRatioLimit(max_ratio=0.20, target_epsg=7855)
]


engine = ValidationEngine(rules)
fail_df = engine.run(gdf)
fail_df.to_csv("validation_failures.csv", index=False)


print("By category:\n", fail_df.groupby("category").size())
print("\nBy rule:\n", fail_df.groupby("rule").size())
print("\nSaved to validation_failures.csv")
