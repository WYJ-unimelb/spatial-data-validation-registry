import streamlit as st
import sqlite3
import pandas as pd

st.title("Validation Registry Viewer")


# connect to databse
conn = sqlite3.connect("validation_registry.db")

# get all tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
table_names = tables['name'].tolist()

if not table_names:
    st.warning("No tables found in the database.")
else:
    selected_table = st.selectbox("Select a table", table_names)

    # read and show data
    df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)

    # filter status
    if "status" in df.columns:
        status_filter = st.selectbox("Filter by status", options=["All", "invalid", "valid", "resolved"])
        if status_filter != "All":
            df = df[df["status"] == status_filter]

    # order data
    if "validated_at" in df.columns:
        df = df.sort_values("validated_at", ascending=False)

    st.subheader("Validation Records")
    st.dataframe(df)

    # visualize
    if 'lat' in df.columns and 'lon' in df.columns:
        st.subheader("Map of Failed Records")
        st.map(df.rename(columns={"lat": "latitude", "lon": "longitude"}))
    else:
        st.info("No 'lat' and 'lon' columns found for mapping.")

    # export data
    st.download_button("Download as CSV", df.to_csv(index=False), "validation_registry_export.csv", "text/csv")

conn.close()
