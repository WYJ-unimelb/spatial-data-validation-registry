import streamlit as st
import geopandas as gpd
import sqlite3
from datetime import datetime
import hashlib

from validation_rules import validate

st.title("Spatial Data Validation Demo")

st.markdown("Upload a GeoJSON file (e.g., roads.geojson or buildings.geojson)")
uploaded_file = st.file_uploader("Choose a GeoJSON file", type=["geojson"])

dtype = st.selectbox("Select dataset type", ["building", "road"])


if uploaded_file is not None:
    gdf = gpd.read_file(uploaded_file)
    st.success("File loaded successfully!")

    # Add lat/lon columns for visualization
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x
    st.map(gdf.rename(columns={"lat": "latitude", "lon": "longitude"}))

    if st.button("Run Validation"):
        result_df = validate(gdf, dtype)

        st.subheader("Validation Report")
        st.dataframe(result_df)

        # save as  CSV
        result_df.to_csv("validation_result.csv", index=False)
        st.download_button("Download CSV Report", data=open("validation_result.csv", "rb"), file_name="validation_result.csv")

        # Extract safe-to-write fields
        safe_cols = [col for col in result_df.columns if result_df[col].apply(lambda x: isinstance(x, (str, int, float, bool))).all()]
        clean_df = result_df[safe_cols].copy()

        # status, timestamp
        clean_df["status"] = "invalid"
        clean_df["validated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def compute_feature_hash(row):
            key_string = str(row.get("id", "")) + str(row.get("lat", "")) + str(row.get("lon", ""))
            return hashlib.md5(key_string.encode()).hexdigest()

        clean_df["feature_hash"] = clean_df.apply(compute_feature_hash, axis=1)

        # write into SQLite database
        conn = sqlite3.connect("validation_registry.db")
        clean_df.to_sql("failed_validations", conn, if_exists="append", index=False)
        conn.close()

        st.success("Validation results stored in registry database!")
