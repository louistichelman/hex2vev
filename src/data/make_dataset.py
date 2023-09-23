from geopandas import GeoDataFrame
from pyproj import crs
from src.data.download import get_bounding_gdf
import h3
from typing import Callable, List, Union
from shapely.geometry import Point, Polygon, geo, MultiPolygon
import pandas as pd
from shapely.geometry import mapping
from tqdm import tqdm
import geopandas as gpd
from src.data.load_data import load_city_tag
from src.data.utils import TOP_LEVEL_OSM_TAGS
from src.settings import DATA_INTERIM_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_BRW
from pathlib import Path
from src.data.load_data import load_city_tag_h3, load_grouped_city
import numpy as np
from typing import Dict


def h3_to_polygon(hex: int) -> Polygon:
    boundary = h3.h3_to_geo_boundary(hex)
    boundary = [[y, x] for [x, y] in boundary]
    h3_polygon = Polygon(boundary)
    return h3_polygon


def get_hexes_for_place(
    area_gdf: GeoDataFrame, resolution: int, return_gdf=False
) -> Union[List[int], GeoDataFrame]:
    original = area_gdf
    buffered = get_buffered_place_for_h3(original, resolution)
    geojson = mapping(buffered)

    all_hexes = []
    all_hex_polygons = []
    for feature in geojson["features"]:
        geom = feature["geometry"]
        geom["coordinates"] = [[[j[1], j[0]] for j in i] for i in geom["coordinates"]]
        hexes = list(h3.polyfill(geom, resolution))
        all_hexes.extend(hexes)
        hex_polygons = list(map(h3_to_polygon, hexes))
        all_hex_polygons.extend(hex_polygons)

    hexes_gdf = GeoDataFrame(
        pd.DataFrame({"h3": all_hexes, "geometry": all_hex_polygons}), crs="EPSG:4326"
    )

    intersecting_hexes_gdf = gpd.sjoin(hexes_gdf, original)
    intersecting_hexes_gdf = intersecting_hexes_gdf[["h3", "geometry"]]
    intersecting_hexes_gdf.drop_duplicates(inplace=True)

    if return_gdf:
        return intersecting_hexes_gdf
    else:
        intersecting_hexes = intersecting_hexes_gdf.h3.tolist()
        return intersecting_hexes


def get_buffered_place_for_h3(place: GeoDataFrame, resolution: int) -> GeoDataFrame:
    twice_edge_length = 2 * h3.edge_length(resolution=resolution, unit="m")
    buffered = place.copy()
    buffered["geometry"] = (
        place.to_crs(epsg=3395).buffer(int(twice_edge_length)).to_crs(epsg=4326)
    )
    return buffered


def prepare_city_path(base_path: Path, city: str, year: str = None) -> Path:
    if year is None:
        city_path = base_path.joinpath(city)
    else:
        city_path = base_path.joinpath(city, year)
    city_path.mkdir(parents=True, exist_ok=True)
    return city_path


def get_hexes_polygons_for_city(
    city: Union[str, List[str]], resolution: int, use_cache=False
) -> GeoDataFrame:
    if type(city) == str:
        city_name = city
    else:
        city_name = city[0]
    city_path = prepare_city_path(DATA_RAW_DIR, city_name)
    cache_file = city_path.joinpath(f"h3_{resolution}.geojson")

    if use_cache and cache_file.exists() and cache_file.is_file():
        return gpd.read_file(cache_file)
    bounding_gdf = get_bounding_gdf(city)

    if type(bounding_gdf.geometry[0]) == MultiPolygon:
        bounding_gdf = bounding_gdf.explode(index_parts=False).reset_index(drop=True)

    hexes_gdf = get_hexes_for_place(bounding_gdf, resolution, return_gdf=True)
    hexes_gdf.to_file(cache_file, driver="GeoJSON")
    return hexes_gdf


def add_h3_indices(
    gdf: GeoDataFrame, city: Union[str, List[str]], resolution: int
) -> GeoDataFrame:
    hexes_polygons_gdf = get_hexes_polygons_for_city(city, resolution, use_cache=True)
    h3_added = gpd.sjoin(gdf, hexes_polygons_gdf, how="inner", predicate="intersects")
    return h3_added


def add_h3_indices_to_city(
    city: Union[str, List[str]],
    year: str,
    resolution: int,
    force=False,
    filter_values: Dict[str, List[str]] = None,
):
    if type(city) == str:
        city_name = city
    else:
        city_name = city[0]
    city_destination_path = prepare_city_path(DATA_INTERIM_DIR, city_name, year)
    for tag in TOP_LEVEL_OSM_TAGS:
        if filter_values is not None and tag not in filter_values.keys():
            continue
        # check if file already exists
        result_path = city_destination_path.joinpath(f"{tag}_{resolution}.feather")
        if result_path.exists() and not force:
            print(f"Skipping {tag} for {city_name} and year {year}, already exists")
            continue
        tag_gdf = load_city_tag(city_name, year, tag, filter_values=filter_values)
        if tag_gdf is not None:
            tag_gdf = tag_gdf[["osmid", tag, "geometry"]]
            h3_gdf = add_h3_indices(tag_gdf, city, resolution)[
                ["osmid", tag, "h3"]
            ].reset_index(drop=True)
            h3_gdf.to_feather(result_path)
        else:
            print(
                f"Tag {tag} doesn't exist for city {city} and year {year}, skipping..."
            )


def merge_all_tags_for_city(city: str, resolution: int):
    city_path = prepare_city_path(DATA_INTERIM_DIR, city)
    dfs = []
    for tag in TOP_LEVEL_OSM_TAGS:
        try:
            gdf = load_city_tag(city, tag)
            if gdf is not None:
                gdf = add_h3_indices(gdf, city, resolution)
                dfs.append(gdf)
        except FileNotFoundError:
            print(f"Tag {tag} file not found for {city}")
    city_df = pd.concat(dfs, ignore_index=True)
    city_file_path = city_path.joinpath(f"all_{resolution}.pkl")
    city_df.to_pickle(city_file_path)
    return city_df


def group_df_by_tag_values(df, tag: str):
    tags = df.reset_index(drop=True)[["h3", tag]]
    indicators = tags[[tag]].pivot(columns=tag, values=tag)
    indicators[indicators.notnull()] = 1
    indicators.fillna(0, inplace=True)
    indicators = indicators.add_prefix(f"{tag}_")
    result = (
        pd.concat([tags[["h3"]], indicators], axis=1).groupby("h3").sum().reset_index()
    )
    return result


def group_city_tags(
    city: str,
    year: str,
    resolution: int,
    tags=TOP_LEVEL_OSM_TAGS,
    filter_values: Dict[str, str] = None,
    fill_missing=True,
) -> pd.DataFrame:
    dfs = []
    for tag in tags:
        df = load_city_tag_h3(city, year, tag, resolution, filter_values)
        if df is not None and not df.empty:
            tag_grouped = group_df_by_tag_values(df, tag)
        else:
            tag_grouped = pd.DataFrame()
        if fill_missing and filter_values is not None:
            missing_columns = pd.Index(
                [f"{tag}_{value}" for value in filter_values[tag]]
            ).difference(tag_grouped.columns)
            tag_grouped = tag_grouped.reindex(
                columns=tag_grouped.columns.union(missing_columns), fill_value=0
            )
        dfs.append(tag_grouped)

    results = pd.concat(dfs, axis=0)
    results = results.fillna(0).groupby("h3").sum().reset_index()

    city_destination_path = prepare_city_path(DATA_PROCESSED_DIR, city, year)
    file_path = city_destination_path.joinpath(f"{resolution}.feather")
    results.to_feather(file_path)
    return results


def group_city_top_level_tags(
    city: str, resolution: int, tags=TOP_LEVEL_OSM_TAGS
) -> pd.DataFrame:
    dfs = []
    for tag in tags:
        df = load_city_tag_h3(city, tag, resolution)
        if df is not None:
            df = df[["h3", tag]]
            dfs.append(df)

    results = pd.concat(dfs, axis=0)
    for tag in tags:
        if tag not in results.columns:
            results[tag] = np.nan
    df_tags = results[tags]
    df_tags = df_tags.notna().astype(float)
    results[tags] = df_tags
    df_top_level = results.groupby("h3").sum().reset_index()[["h3", *tags]]
    return df_top_level


def add_geometry_to_df(df: pd.DataFrame) -> GeoDataFrame:
    df["geometry"] = df["h3"].apply(h3_to_polygon)
    gdf = GeoDataFrame(df, crs="EPSG:4326")
    return gdf


def group_cities(
    cities: str, years: str, resolution: int, add_city_column=True
) -> pd.DataFrame:
    dfs = []
    for city in cities:
        for year in years:
            df = load_grouped_city(city, year, resolution)
            if add_city_column:
                df["city"] = city
                df["year"] = year
            dfs.append(df)
    all_cities = pd.concat(dfs, axis=0, ignore_index=True).set_index("h3")
    return all_cities


def polygon_to_h3_old(df, city, resolution, label="BRW", with_empty_hexes=False):
    hex_df = get_hexes_polygons_for_city(city, resolution)
    hex_df[["x", "y"]] = hex_df.h3.apply(h3.h3_to_geo).tolist()
    hex_df["centroid"] = gpd.points_from_xy(hex_df.y, hex_df.x)
    hex_df.index = hex_df.h3

    hex_df[label] = np.nan
    df = df.to_crs(epsg=4326)

    for idx in hex_df.index:
        try:
            contained = df.geometry.contains(hex_df.loc[idx, "centroid"])
        except:
            print(f"There was a problem with the geometries.")
            continue
        if contained.sum() > 0:
            hex_df.loc[idx, label] = df[contained][label].iloc[0]

    hex_df[label] = pd.to_numeric(hex_df[label])

    if with_empty_hexes:
        return hex_df
    else:
        return hex_df[~np.isnan(hex_df[label])]


def polygon_to_h3(df, city, resolution, label="BRW", with_empty_hexes=False):
    hex_df = get_hexes_polygons_for_city(city, resolution)
    hex_df.index = hex_df.h3

    hex_df[label] = np.nan
    df = df.to_crs(epsg=4326)

    for idx in hex_df.index:
        if df.geometry.intersects(hex_df.loc[idx, "geometry"]).sum() > 0:
            hex_df.loc[idx, label] = df.loc[
                df.geometry.intersection(hex_df.loc[idx, "geometry"]).area.idxmax(),
                label,
            ]

    hex_df[label] = pd.to_numeric(hex_df[label])

    if with_empty_hexes:
        return hex_df
    else:
        return hex_df[~np.isnan(hex_df[label])]


def merge_years(dic, years, label: str = "BRW", save: bool = False):
    hex_intersect = set(dic[years[0]].h3)
    for year in years:
        hex_intersect = set(dic[year].h3) & hex_intersect

    hex_intersect = list(hex_intersect)

    df_merged = dic[years[0]].loc[hex_intersect]
    df_merged[f"{label}_{years[0]}"] = df_merged[label]
    df_merged.drop(columns=label, inplace=True)
    for year in years:
        df_merged[f"{label}_{year}"] = dic[year].loc[hex_intersect][label]

    if save:
        df_merged.to_feather(DATA_BRW.joinpath("merged.feather"))

    return df_merged


def concat(dic):
    df_concat = pd.concat(list(dic.values()))
    df_concat = df_concat[~df_concat.index.duplicated(keep="first")]
    df_concat.to_feather(DATA_BRW.joinpath("concat.feather"))
    return df_concat


def merge_tags(df, tag_filter, tags_no_merge, values_no_merge):
    all_values = set()
    doubled = set()
    for tag in tag_filter:
        for value in tag_filter[tag]:
            if value in all_values:
                doubled.add(value)
            all_values.add(value)

    doubled = doubled - values_no_merge

    tags_to_sum_up = {}
    for value_doubled in doubled:
        tags_to_sum_up[value_doubled] = []
        for tag in tag_filter:
            if tag in tags_no_merge:
                continue
            for poi in tag_filter[tag]:
                if poi == value_doubled:
                    tags_to_sum_up[value_doubled].append(tag + "_" + poi)

    for tag in tags_to_sum_up:
        df[tag] = df[tags_to_sum_up[tag]].sum(axis=1)
        df.drop(columns=tags_to_sum_up[tag], inplace=True)

    return df
