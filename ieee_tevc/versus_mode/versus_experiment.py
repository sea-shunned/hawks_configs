from pathlib import Path
from itertools import permutations

import seaborn as sns

import hawks

ALGS = ["alink", "gmm", "kmeans", "slink"]
SEED_NUM = 500
NUM_RUNS = 30
LOAD_DATASETS = False
SKIP = []

for i, (winner, loser) in enumerate(permutations(ALGS, 2)):
    print(winner, loser)
    folder_name = f"versus_{winner}-v-{loser}"
    # Skip certain experiments if desired
    if i in SKIP:
        continue
    if LOAD_DATASETS:
        gen = hawks.load_folder(folder_name)
    else:
        gen = hawks.create_generator({
            "hawks": {
                "mode": "versus",
                "folder_name": folder_name,
                "num_runs": NUM_RUNS,
                "seed_num": SEED_NUM,
                "save_stats": True,
                "save_best_data": True,
                "save_config": True
            },
            "objectives": {
                "winner": {
                    "algorithm": winner
                },
                "loser": {
                    "algorithm": loser
                }
            },
            "dataset": {
                "num_examples": 2000,
                "num_clusters": 5
            },
            "ga": {
                "num_indivs": 10,
                "num_gens": 100,
                "prob_fitness": 0.75,
                "elites": 0,
                "mut_method_mean": "pso",
                "initial_mean_upper": 10.0,
            },
            "constraints": {
                "overlap": {
                    "threshold": 0.0,
                    "limit": "upper"
                },
                "eigenval_ratio": {
                    "threshold": 20,
                    "limit": "upper"
                }
            }
        })

        gen.run()

    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.15)
    # Plot predictions for both algorithms and truth
    gen.plot_best_indiv_alg_predictions(
        cmap="viridis", remove_axis=False,
        alpha=0.7, s=7, save=True, show=False
    )
    hawks.plotting.alg_comp_parallel(
        df=gen.stats[gen.stats["best_indiv"] == 1],
        color=sns.color_palette("cubehelix", 2)[0],
        linewidth=1.5,
        alpha=0.6,
        fpath=Path.cwd() / folder_name / f"parallel_comp_{winner}-v-{loser}",
        show=False
    )
    hawks.plotting.corr_plot(
        df=gen.stats[gen.stats["best_indiv"] == 1],
        x="overlap",
        y="fitness_alg_diff",
        xlabel="Overlap",
        ylabel="ARI Difference",
        color=sns.color_palette("cubehelix", 2)[0],
        fpath=Path.cwd() / folder_name / f"overlap_corr_{winner}-v-{loser}",
        show=False
    )
    # Compare against other clustering algs
    df, _ = hawks.analysis.analyse_datasets(
        generator=gen,
        feature_space=False,
        save_folder=Path.cwd() / folder_name
    )
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.05)
    output_df = df.melt(
        id_vars=[col for col in df if not col.startswith("c_")],
        value_vars=[col for col in df if col.startswith("c_")],
        var_name="Algorithm",
        value_name="ARI"
    )
    # Remove the c_ prefix to algorithm names
    output_df['Algorithm'] = output_df['Algorithm'].map(lambda x: str(x)[2:])
    hawks.plotting.create_boxplot(
        df=output_df,
        x="Algorithm",
        y="ARI",
        hue="Algorithm",
        cmap=sns.color_palette("colorblind"),
        xlabel="",
        ylabel="ARI",
        hatching=True,
        show=False,
        fpath=Path.cwd() / folder_name / f"clustering_bplot_{winner}-v-{loser}",
        remove_legend=True,
        clean_props={
            "legend_type": None,
            "clean_labels": False,
            "wrap_ticks": True
        },
        dodge=False
    )
