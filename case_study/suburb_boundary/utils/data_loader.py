import json
import geopandas as gpd


def json_loader(file_path: str):
    # Use this function to load GeoJson dataset
    data = json.load(open(file_path, encoding='utf-8'))
    return data


def gpd_loader(file_path: str) -> gpd.GeoDataFrame:
    data = gpd.read_file(file_path)
    if data.geometry.empty:
        raise AttributeError("No geometry found in file")
    return data
