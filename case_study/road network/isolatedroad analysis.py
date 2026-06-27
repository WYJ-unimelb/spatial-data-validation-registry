import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

class RoadIsolationAnalyzer:
    def __init__(self, gdf, start_col="start_node_name", end_col="end_node_name"):
        self.gdf = gdf
        self.start_col = start_col
        self.end_col = end_col
        self.truly_isolated = None
        self.final_isolated = None
        self.problematic = None

    @classmethod
    def from_file(cls, path, start_col="start_node_name", end_col="end_node_name"):
        gdf = gpd.read_file(path)
        return cls(gdf, start_col=start_col, end_col=end_col)

    def compute_candidate_isolated(self):
        #using endpoint testing
        from collections import Counter
        all_nodes = list(self.gdf[self.start_col]) + list(self.gdf[self.end_col])
        counts = Counter(all_nodes)
        once_nodes = {n for n,c in counts.items() if c == 1}

        # Contains candidates that appear only once at either end.
        once_node_rows = self.gdf[
            self.gdf[self.start_col].isin(once_nodes) |
            self.gdf[self.end_col].isin(once_nodes)
        ]

        # Appears only once at both ends
        def both_ends_once(row):
            return (row[self.start_col] in once_nodes) and (row[self.end_col] in once_nodes)

        self.truly_isolated = once_node_rows[once_node_rows.apply(both_ends_once, axis=1)]
        return self.truly_isolated

    def compute_final_isolated(self, strict=False, buffer_eps=0.0):
        # geometric test, strict=False means any geometric intersection (intersects) with other roads is considered ‘connected.’
        if self.truly_isolated is None:
            raise ValueError("请先调用 compute_candidate_isolated()")

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

            if strict:
                rel = (candidates.geometry.crosses(geom) | candidates.geometry.overlaps(geom))
            else:
                rel = candidates.geometry.intersects(geom)

            if not rel.any():
                still_isolated_idx.append(idx)

        self.final_isolated = self.truly_isolated.loc[still_isolated_idx]
        # not really isolated 
        self.problematic = self.truly_isolated.drop(self.final_isolated.index)
        return self.final_isolated

    def summary(self, n=10):
        if self.truly_isolated is None:
            return
        print(f"endpoint testing  {len(self.truly_isolated)} pieces of data")
        if self.final_isolated is not None:
            print(f"geometric testing {len(self.final_isolated)} pieces of data")
            print(self.final_isolated[["road", "road_name", self.start_col, self.end_col]].head(n))
            print(f"{len(self.problematic)} pieces of different data")
            print(self.problematic[["road", "road_name", self.start_col, self.end_col]].head(n))

    def export(self,
               final_geojson=None,
               problematic_geojson=None,
               final_csv=None,
               problematic_csv=None):
        
        if self.final_isolated is not None:
            if final_geojson: self.final_isolated.to_file(final_geojson, driver="GeoJSON")
            if final_csv: self.final_isolated.to_csv(final_csv, index=False)
        if self.problematic is not None:
            if problematic_geojson: self.problematic.to_file(problematic_geojson, driver="GeoJSON")
            if problematic_csv: self.problematic.to_csv(problematic_csv, index=False)

    def plot(self, show_background=True, figsize=(10,10)):
        #visualisation 
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



if __name__ == "__main__":
    import geopandas as gpd

    gdf = gpd.read_file(r"C:\Users\86183\Desktop\master\25s1\PROJECT\wa-govt-mrwa-mrwa-road-network-2018-na.json")
    test = RoadIsolationAnalyzer(gdf)
    test.compute_candidate_isolated()
    test.compute_final_isolated(strict=False)
    test.summary(n=10)
    test.plot(show_background=True)

