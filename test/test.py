from demo import rule_checker
import geopandas as gpd

resu = rule_checker.run_suburb_checks(gpd.read_file("test/toy_suburb.geojson").to_crs(3107))
print(resu)
# suburb check

