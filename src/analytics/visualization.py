import matplotlib.pyplot as plt
import contextily as ctx
from src.data.make_dataset import h3_to_polygon
import geopandas as gpd
import numpy as np


def plot_map(gdf, column=None, bins=None, alpha=0.7, colors=None, colorscheme=None):
    try:
        gdf = gpd.GeoDataFrame(gdf, crs="EPSG:4326")
    except:
        pass
    fig_size = (15, 15)
    map_source = ctx.providers.CartoDB.Positron
    fig, ax = plt.subplots(figsize=fig_size)
    if bins is not None:
        gdf.to_crs(epsg=3857).plot(
            ax=ax,
            column=column,
            alpha=alpha,
            cmap=colors,
            scheme="UserDefined",
            classification_kwds={"bins": bins},
            legend=True,
        )
    else:
        gdf.to_crs(epsg=3857).plot(
            ax=ax,
            column=column,
            alpha=alpha,
            cmap=colors,
            legend=True,
            color=colorscheme,
        )
    ctx.add_basemap(ax, source=map_source)
