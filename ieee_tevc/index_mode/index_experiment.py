from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import hawks

NAME = "index-mode"
SAVE_FOLDER = Path.cwd() / NAME
FILENAME = "suite_analysis" # Needed? For plots I think
SEED_NUM = 0
ANALYSIS_PATH = SAVE_FOLDER / (FILENAME+".csv")
LOAD_ANALYSIS = ANALYSIS_PATH.is_file()
LOAD_SUITE = SAVE_FOLDER.is_dir()

FEATURES_ONLY = True

# Functions to process the raw data from different generators/collections
def process_HK(files):
    filenames = []
    datasets = []
    label_sets = []

    for file in files:
        filenames.append(file.stem)
        data = np.loadtxt(file, skiprows=4)
        label_sets.append(data[:, -1].astype(int))
        datasets.append(data[:, :-1])
    return filenames, datasets, label_sets

def process_QJ(files):
    filenames = []
    datasets = []
    label_sets = []

    for file in files:
        filenames.append(file.stem)
        data = np.genfromtxt(file, skip_header=1, delimiter=" ")
        datasets.append(data)
        labels = np.loadtxt(f"{file.parent / file.stem}.mem", delimiter="\n").astype(int)
        label_sets.append(labels)
    return filenames, datasets, label_sets

def process_UCI(files):
    filenames = []
    datasets = []
    label_sets = []

    for file in files:
        filenames.append(file.stem)
        data = np.loadtxt(file, skiprows=1, delimiter=",")
        # Add the datasets and labels
        label_sets.append(data[:, -1].astype(int))
        datasets.append(data[:, :-1])
    return filenames, datasets, label_sets

def process_UKC(files):
    filenames = []
    datasets = []
    label_sets = []

    for file in files:
        filenames.append(file.stem)
        data = np.loadtxt(file, skiprows=4)
        label_sets.append(data[:, -1].astype(int))
        datasets.append(data[:, :-1])
    return filenames, datasets, label_sets

# Load previous analysis if present
if LOAD_ANALYSIS:
    print("Loading previous analysis")
    df = pd.read_csv(ANALYSIS_PATH, index_col=False)
    gen = None
else:
    if not LOAD_SUITE:
        print("Generating datasets")
        # All combinations of parameter lists used
        config = {
            "hawks": {
                "folder_name": NAME,
                "save_best_data": True,
                "save_stats": True,
                "save_config": True,
                "seed_num": SEED_NUM,
                "num_runs": 7,
                "comparison": "ranking"
            },
            "dataset": {
                "num_examples": [500, 2500],
                "num_dims": [2, 50],
                "num_clusters": [5, 30],
                "min_clust_size": 10
            },
            "objectives": {
                "silhouette": {
                    "target": [0.45, 0.9]
                }
            },
            "ga": {
                "num_gens": 100,
                "prob_fitness": 0.5,
                "elites": 0,
                "mut_method_mean": "pso",
                "mut_args_mean": {
                    "pso": {
                        "scalar_lower": 0.0,
                        "scalar_higher": 1.0
                    }
                },
                "initial_mean_upper": 2.0
            },
            "constraints": {
                "overlap": {
                    "threshold": [0.0, 0.1],
                    "limit": "upper"
                },
                "eigenval_ratio": {
                    "threshold": [1, 50],
                    "limit": "lower"
                }
            }
        }
        # Create the gen
        gen = hawks.create_generator(config)
        # Run the gen(s)
        gen.run()
    else:
        print("Loading datasets")
        gen = hawks.load_folder(SAVE_FOLDER)
    ########## Load & analyze datasets ##########
    kws = {
        "clustering": True,
        "feature_space": True,
        "save": True
    }
    if FEATURES_ONLY:
        kws["clustering"] = False
        kws["save"] = False

    print("Analyzing datasets")
    df, _ = hawks.analysis.analyse_datasets(
        generator=gen,
        source="HAWKS",
        seed=SEED_NUM,
        save_folder=SAVE_FOLDER,
        filename=FILENAME+"_incomp",
        **kws
    )
    print("Done HAWKS!")
    # HK data
    filenames, datasets, label_sets = hawks.load_datasets(
        Path("../other_generators/HK_new_data"),
        custom_func=process_HK,
        glob_filter="*.data"
    )
    df, _ = hawks.analysis.analyse_datasets(
        datasets=datasets,
        label_sets=label_sets,
        source="HK",
        seed=SEED_NUM,
        prev_df=df,
        save_folder=SAVE_FOLDER,
        filename=FILENAME+"_incomp",
        **kws
    )
    print("Done HK!")
    # QJ data
    filenames, datasets, label_sets = hawks.load_datasets(
        Path("../other_generators/QJ_data"),
        custom_func=process_QJ,
        glob_filter="*.dat"
    )
    df, _ = hawks.analysis.analyse_datasets(
        datasets=datasets,
        label_sets=label_sets,
        source="QJ",
        seed=SEED_NUM,
        prev_df=df,
        save_folder=SAVE_FOLDER,
        filename=FILENAME+"_incomp",
        **kws
    )
    print("Done QJ!")
    # SIPU data
    filenames, datasets, label_sets = hawks.load_datasets(
        Path("../other_generators/SIPU_data"),
        labels_last_column=True,
        glob_filter="*.csv",
        delimiter=","
    )
    df, _ = hawks.analysis.analyse_datasets(
        datasets=datasets,
        label_sets=label_sets,
        source="SIPU",
        seed=SEED_NUM,
        prev_df=df,
        save_folder=SAVE_FOLDER,
        filename=FILENAME+"_incomp",
        **kws
    )
    print("Done SIPU!")
    # UCI data
    filenames, datasets, label_sets = hawks.load_datasets(
        Path("../other_generators/UCI_data"),
        custom_func=process_UCI,
        glob_filter="*.csv"
    )
    df, _ = hawks.analysis.analyse_datasets(
        datasets=datasets,
        label_sets=label_sets,
        source="UCI",
        seed=SEED_NUM,
        prev_df=df,
        save_folder=SAVE_FOLDER,
        filename=FILENAME+"_incomp",
        **kws
    )
    print("Done UCI!")
    # UKC data
    filenames, datasets, label_sets = hawks.load_datasets(
        Path("../other_generators/UKC_data"),
        custom_func=process_UKC,
        glob_filter="*.txt"
    )
    df, _ = hawks.analysis.analyse_datasets(
        datasets=datasets,
        label_sets=label_sets,
        source="UKC",
        seed=SEED_NUM,
        prev_df=df,
        save_folder=SAVE_FOLDER,
        filename=FILENAME,
        **kws
    )
    print("Done UKC!")

sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
print("Generating plots")

# General args for instance space
instance_kws = dict(
    legend_type="brief",
    cmap="viridis",
    show=False,
    alpha=0.7,
    s=12,
    filename=FILENAME,
    seed=SEED_NUM,
    save_folder=SAVE_FOLDER,
    clean_props={
        "clean_labels": False,
        "clean_legend": True,
        "legend_truncate": True,
        "legend_loc": "center left"
    }
)

hawks.plotting.instance_space(
    df=df,
    color_highlight=None,
    marker_highlight="source",
    show=False,
    save_folder=SAVE_FOLDER,
    legend_type=None,
    filename=FILENAME+"_comps",
    plot_data=True,
    plot_components=True,
    save_red_obj=True
)

color_features = ["source"] + [col for col in df if col.startswith("f_")]

hawks.plotting.instance_space(
    df=df,
    color_highlight="source",
    marker_highlight="source",
    legend_type="brief",
    cmap="viridis",
    show=False,
    alpha=0.7,
    s=12,
    filename=FILENAME+"_pca",
    dim_red_method="pca",
    seed=SEED_NUM,
    save_folder=SAVE_FOLDER,
    clean_props={
        "clean_labels": False,
        "clean_legend": True,
        "legend_truncate": True,
        "legend_loc": "center left"
    }
)

for color_feature in color_features:
    if color_feature == "source":
        # instance_kws["cmap"] = cm.get_cmap("viridis", 6).colors
        instance_kws["cmap"] = sns.color_palette("cubehelix", 6)
    else:
        instance_kws["cmap"] = "viridis"
    hawks.plotting.instance_space(
        df=df,
        color_highlight=color_feature,
        marker_highlight="source",
        **instance_kws
    )

if not FEATURES_ONLY:
    instance_kws["cmap"] = sns.color_palette("cubehelix", 7)
    # hawks.plotting.instance_space(
    #     df=df,
    #     color_highlight="algorithm",
    #     marker_highlight="source",
    #     **instance_kws
    # )
    # hawks.plotting.instance_space(
    #     df=df,
    #     color_highlight="algorithm",
    #     marker_highlight="algorithm",
    #     hue_order=["Tied", "Average-Linkage", "Average-Linkage (2k)", "GMM", "K-Means++", "Single-Linkage", "Single-Linkage (2k)"],
    #     style_order=["Tied", "Average-Linkage", "Average-Linkage (2k)", "GMM", "K-Means++", "Single-Linkage", "Single-Linkage (2k)"],
    #     **instance_kws
    # )
    output_df = df.melt(
        id_vars=[col for col in df if not col.startswith("c_")],
        value_vars=[col for col in df if col.startswith("c_")],
        var_name="Algorithm",
        value_name="ARI"
    )
    # Remove the c_ prefix to algorithm names
    output_df['Algorithm'] = output_df['Algorithm'].map(lambda x: str(x)[2:])
    instance_kws["cmap"] = "viridis"
    hawks.plotting.instance_space(
        df=output_df.loc[output_df.groupby(["source", "dataset_num"])["ARI"].idxmax()].reset_index(drop=True),
        color_highlight="ARI",
        marker_highlight="source",
        **instance_kws
    )

    sns.set_style("whitegrid")
    hawks.plotting.create_boxplot(
        df=output_df,
        x="source",
        y="ARI",
        hue="Algorithm",
        cmap=sns.color_palette("colorblind"),
        xlabel="",
        ylabel="ARI",
        fpath=SAVE_FOLDER / f"{FILENAME}_clustering",
        hatching=True,
        clean_props={
            "clean_labels": False,
            "legend_loc": "center left"
        },
        fliersize=3
    )

    hawks.plotting.cluster_alg_ranking(
        df=df[["source"]+[col for col in df if col.startswith("c_")]],
        save_folder=SAVE_FOLDER,
        filename=FILENAME+"-ranking"
    )
    if gen is None:
        gen = hawks.load_folder(SAVE_FOLDER)
    bplot_df = output_df[output_df["source"] == "HAWKS"]
    bplot_df = bplot_df.merge(gen.stats[gen.stats["best_indiv"] == 1][["overlap_threshold","config_id"]], left_on="config_num", right_on="config_id")

    hawks.plotting.create_boxplot(
        df=bplot_df,
        x="overlap_threshold",
        y="ARI",
        hue="Algorithm",
        cmap=sns.color_palette("colorblind"),
        xlabel="Overlap Threshold",
        ylabel="ARI",
        fpath=SAVE_FOLDER / f"{FILENAME}_clustering_olap",
        hatching=True,
        clean_props={
            "clean_labels": False,
            "legend_loc": "center left"
        },
        fliersize=3
    )

if gen is None:
    gen = hawks.load_folder(SAVE_FOLDER)

del instance_kws["seed"]
instance_kws["filename"] = "suite_analysis_ii_highlight"
instance_kws["legend_type"] = "full"
instance_kws["alpha"] = 0.75
instance_kws["s"] = 25
instance_kws["cmap"] = "magma"
instance_kws["cmap"] = sns.dark_palette((260, 88, 50), input="husl", as_cmap=True)

sns.set_style("ticks")
hawks.plotting.instance_parameters(
    gen=gen,
    df=df,
    color_highlight="overlap_threshold",
    marker_highlight="silhouette_target",
    **instance_kws
)
hawks.plotting.instance_parameters(
    gen=gen,
    df=df,
    color_highlight="eigenval_ratio_threshold",
    marker_highlight="silhouette_target",
    **instance_kws
)
df["num_examples"] = df["num_examples"].round(decimals=-2)
instance_kws["clean_props"]["legend_truncate"] = False
hawks.plotting.instance_parameters(
    gen=gen,
    df=df,
    color_highlight="num_examples",
    marker_highlight="silhouette_target",
    **instance_kws
)