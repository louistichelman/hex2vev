from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch


def find_important_tags_hexagon(model, tags, hexagon: pd.Series, median, nr_top_tags):
    hexagon_emb = model.encoder(torch.Tensor(hexagon.to_numpy())).detach().numpy()
    similarities = pd.Series(1, index=hexagon.index)
    for tag in tags[median != hexagon]:
        changed_hex = hexagon.copy()
        changed_hex[tag] = median[tag]
        changed_emb = (
            model.encoder(torch.Tensor(changed_hex.to_numpy())).detach().numpy()
        )
        similarities[tag] = float(
            cosine_similarity(changed_emb.reshape(1, -1), hexagon_emb.reshape(1, -1))
        )
    return list(similarities.sort_values()[:nr_top_tags].index)


def find_important_tags(df, model, tags, median, nr_top_tags):
    return pd.DataFrame(
        df.apply(
            lambda row: find_important_tags_hexagon(
                model, tags, row, median, nr_top_tags
            ),
            axis=1,
        )
        .apply(lambda row: list(row))
        .tolist(),
        index=df.index,
    )


def tag_importance_df(all_tags, top_tags):
    df = pd.DataFrame(index=top_tags.index, columns=all_tags)
    for tag in all_tags:
        df.loc[:, tag] = None
        for nr, column in enumerate(top_tags.columns):
            df.loc[top_tags[column] == tag, tag] = nr
    return df


def find_sim_group_hexagon(model, tags, hexagon: pd.Series, median):
    hexagon_emb = model.encoder(torch.Tensor(hexagon.to_numpy())).detach().numpy()
    changed_hex = hexagon.copy()
    changed_hex[tags] = median[tags]
    changed_emb = model.encoder(torch.Tensor(changed_hex.to_numpy())).detach().numpy()
    return float(
        cosine_similarity(changed_emb.reshape(1, -1), hexagon_emb.reshape(1, -1))
    )


def find_sim_group(df, model, tags, median):
    return df.apply(
        lambda row: find_sim_group_hexagon(model, tags, row, median), axis=1
    )


def cluster_difference(df, first_cluster, second_cluster, label, all_tags):
    means = df.groupby(label).mean(numeric_only=True)
    important_tags = set(
        means.columns[
            (means.loc[first_cluster] > 0.25) | (means.loc[second_cluster] > 0.25)
        ]
    )
    return (means.loc[first_cluster] / means.loc[second_cluster])[
        list(set(all_tags).intersection(important_tags))
    ].sort_values(ascending=False)


def explain_cluster(df, cluster, label, ascending, all_tags):
    means_cluster = df[df[label] == cluster].mean(numeric_only=True)
    means_other = df[df[label] != cluster].mean(numeric_only=True)
    important_tags = list(
        set(means_cluster.index[means_cluster > 0.2]).intersection(all_tags)
    )
    return (
        means_cluster.loc[important_tags] / means_other.loc[important_tags]
    ).sort_values(ascending=ascending)


def cluster_sizes(df, label):
    return df.groupby(label).size()
