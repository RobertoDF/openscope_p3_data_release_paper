import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import io
    from pathlib import Path
    import pickle
    import re

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import remfile
    import seaborn as sns

    from openscope_p3_publication.optotagging import (
        CONDITIONS,
        CONDITION_DISPLAY_NAMES,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SESSION_PATH,
        SessionSkipped,
        analyze_asset,
        baseline_zscore,
        discover_session_assets,
        render_session_heatmaps,
        render_session_summary,
        write_results,
    )


    return (
        CONDITIONS,
        CONDITION_DISPLAY_NAMES,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SESSION_PATH,
        Path,
        SessionSkipped,
        ThreadPoolExecutor,
        analyze_asset,
        as_completed,
        baseline_zscore,
        discover_session_assets,
        h5py,
        io,
        mo,
        np,
        pd,
        pickle,
        plt,
        re,
        remfile,
        render_session_heatmaps,
        render_session_summary,
        sns,
        write_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OpenScope P3 optotagging analysis

    This notebook streams public Neuropixels NWBs from
    [DANDI:001637](https://dandiarchive.org/dandiset/001637/draft/files), aligns
    unit activity to laser onset, and reproduces the reference optotagging
    metrics for one session or the full session inventory.

    Open it with:

    ```powershell
    uv run --extra optotagging marimo edit optotagging_analysis.py
    ```

    Batch results are saved to
    `C:\Users\Roberto\Data\openscope_p3_data_release_paper`.
    """)
    return


@app.cell(hide_code=True)
def _(DEFAULT_SESSION_PATH, discover_session_assets, pd):
    assets = discover_session_assets()
    asset_by_path = {asset["asset_path"]: asset for asset in assets}
    asset_inventory = pd.DataFrame(assets).drop(columns="content_url")
    assert DEFAULT_SESSION_PATH in asset_by_path, "Default optotagging session is absent from DANDI"
    return asset_by_path, asset_inventory, assets


@app.cell(hide_code=True)
def _(asset_inventory, assets, mo):
    mo.vstack(
        [
            mo.md(
                f"**Discovered {len(assets)} session NWBs.** "
                "Only sessions containing all three canonical optotagging tables are analyzed."
            ),
            mo.ui.table(
                asset_inventory,
                pagination=True,
                page_size=10,
                selection=None,
                show_download=True,
                label="DANDI session assets",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(DEFAULT_SESSION_PATH, assets, mo):
    analysis_mode = mo.ui.dropdown(
        options=["Single session", "All sessions"],
        value="Single session",
        label="Analysis mode",
    )
    session_path = mo.ui.dropdown(
        options=[asset["asset_path"] for asset in assets],
        value=DEFAULT_SESSION_PATH,
        searchable=True,
        label="Single-session NWB",
        full_width=True,
    )
    batch_workers = mo.ui.slider(
        start=1,
        stop=4,
        step=1,
        value=2,
        show_value=True,
        label="Concurrent sessions",
    )
    run_analysis = mo.ui.run_button(
        label="Run optotagging analysis",
        kind="success",
        full_width=True,
    )
    mo.vstack(
        [
            mo.hstack([analysis_mode, batch_workers], justify="start", gap=2),
            session_path,
            run_analysis,
            mo.callout(
                "All-session mode streams every NWB and renders every session heatmap. "
                "It is gated because it is network- and compute-intensive.",
                kind="warn",
            ),
        ]
    )
    return analysis_mode, batch_workers, run_analysis, session_path


@app.cell(hide_code=True)
def _(
    SessionSkipped,
    ThreadPoolExecutor,
    analysis_mode,
    analyze_asset,
    as_completed,
    asset_by_path,
    assets,
    batch_workers,
    mo,
    pd,
    render_session_heatmaps,
    render_session_summary,
    run_analysis,
    session_path,
    write_results,
):
    single_result = None
    single_heatmap_png = None
    single_summary_png = None
    batch_heatmaps = []
    batch_metrics = pd.DataFrame()
    skipped_sessions = []
    failed_sessions = []
    parquet_path = None
    provenance_path = None

    if run_analysis.value and analysis_mode.value == "Single session":
        selected_asset = asset_by_path[session_path.value]
        with mo.status.spinner(title="Streaming and analyzing the selected NWB"):
            single_result = analyze_asset(selected_asset)
            single_heatmap_png = render_session_heatmaps(single_result)
            single_summary_png = render_session_summary(single_result)

    elif run_analysis.value and analysis_mode.value == "All sessions":
        metric_frames = []
        rendered_sessions = {}
        with ThreadPoolExecutor(max_workers=int(batch_workers.value)) as executor:
            futures = {executor.submit(analyze_asset, asset): asset for asset in assets}
            completed_futures = mo.status.progress_bar(
                as_completed(futures),
                total=len(futures),
                title="Analyzing DANDI optotagging sessions",
                completion_title="Session analysis complete",
            )
            for future in completed_futures:
                asset = futures.pop(future)
                try:
                    result = future.result()
                except SessionSkipped as error:
                    skipped_sessions.append(
                        {"asset_path": asset["asset_path"], "reason": str(error)}
                    )
                except (KeyError, OSError, RuntimeError, ValueError) as error:
                    failed_sessions.append(
                        {
                            "asset_path": asset["asset_path"],
                            "error_type": type(error).__name__,
                            "reason": str(error),
                        }
                    )
                else:
                    metric_frames.append(result.metrics)
                    rendered_sessions[result.asset_path] = (
                        result.session_id,
                        result.unit_count,
                        result.trial_counts,
                        render_session_heatmaps(result),
                    )

        batch_heatmaps = [
            (asset_path_value, *rendered_sessions[asset_path_value])
            for asset_path_value in sorted(rendered_sessions)
        ]
        if metric_frames:
            batch_metrics = pd.concat(metric_frames, ignore_index=True)
            parquet_path, provenance_path = write_results(
                batch_metrics,
                assets=assets,
                skipped=skipped_sessions,
                failed=failed_sessions,
            )
    return (
        batch_heatmaps,
        failed_sessions,
        single_heatmap_png,
        single_result,
        single_summary_png,
        skipped_sessions,
    )


@app.cell(hide_code=True)
def _(mo, pd, single_heatmap_png, single_result, single_summary_png):
    single_panel = mo.callout(
        "Choose single-session mode and click **Run optotagging analysis**.",
        kind="info",
    )
    if single_result is not None:
        trial_table = pd.DataFrame(
            [
                {"condition": condition, "presentations": count}
                for condition, count in single_result.trial_counts.items()
            ]
        )
        single_panel = mo.vstack(
            [
                mo.md(f"## {single_result.session_id}"),
                mo.hstack(
                    [
                        mo.stat(single_result.unit_count, label="Non-noise units"),
                        mo.stat(
                            sum(single_result.trial_counts.values()),
                            label="Presentations",
                        ),
                    ]
                ),
                mo.ui.table(trial_table, selection=None, label="Condition coverage"),
                mo.md("### Laser-onset-aligned PSTH heatmaps"),
                mo.image(
                    single_heatmap_png,
                    alt="Laser-aligned optotagging PSTH heatmaps",
                ),
                mo.md("### Mean PSTHs and paired pre/post firing rates"),
                mo.image(single_summary_png, alt="Optotagging response summaries"),
                mo.md("### Per-unit optotagging metrics"),
                mo.ui.table(
                    single_result.metrics,
                    pagination=True,
                    page_size=20,
                    selection=None,
                    show_download=True,
                ),
            ]
        )
    return (single_panel,)


@app.cell(hide_code=True)
def _(
    CONDITIONS,
    DEFAULT_OUTPUT_DIR,
    Path,
    assets,
    batch_heatmaps,
    failed_sessions,
    io,
    mo,
    np,
    pd,
    plt,
    skipped_sessions,
):
    coverage = pd.DataFrame(
        [
            {"status": "analyzed", "sessions": len(batch_heatmaps)},
            {"status": "skipped", "sessions": len(skipped_sessions)},
            {"status": "failed", "sessions": len(failed_sessions)},
        ]
    )
    results_path = DEFAULT_OUTPUT_DIR / "optotagging-results.parquet"
    if results_path.exists():
        result_rows = pd.read_parquet(results_path)
        result_sessions = (
            result_rows.groupby(["session_id", "asset_path"], as_index=False)
            .agg(unit_count=("unit_id", "size"))
            .sort_values("session_id", kind="stable")
        )
        heatmap_options = dict(
            zip(result_sessions["session_id"], result_sessions["asset_path"], strict=True)
        )
        heatmap_session_id_by_path = dict(
            zip(result_sessions["asset_path"], result_sessions["session_id"], strict=True)
        )
    else:
        result_rows = pd.DataFrame()
        heatmap_options = {
            Path(asset["asset_path"]).stem
            .split("_ses-", 1)[1]
            .removesuffix("_ecephys")
            .replace("-", "_"): asset["asset_path"]
            for asset in assets
        }
        heatmap_session_id_by_path = {
            asset_path: session_id
            for session_id, asset_path in heatmap_options.items()
        }

    condition_names = [condition.table_name for condition in CONDITIONS]
    if not result_rows.empty:
        tagged_mask = pd.Series(True, index=result_rows.index)
        for condition_name in condition_names:
            tagged_mask &= (
                result_rows[f"{condition_name}__p_value"].lt(0.05)
                & result_rows[f"{condition_name}__modulation_index"].gt(0.1)
            )
        tagged_rows = result_rows.loc[tagged_mask].copy()
        optotagged_per_session = (
            tagged_rows.groupby("session_id").size()
            .reindex(result_sessions["session_id"], fill_value=0)
            .rename("optotagged_cells")
        )
        mean_optotagged = float(optotagged_per_session.mean())

        summary_figure, summary_axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        summary_axis.bar(
            np.arange(len(optotagged_per_session)),
            optotagged_per_session.to_numpy(),
            color="#4C78A8",
            width=0.8,
        )
        summary_axis.axhline(
            mean_optotagged,
            color="#D62728",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_optotagged:.1f} cells/session",
        )
        summary_axis.set(
            xlabel="Session",
            ylabel="Optotagged cells",
            title="Optotagged cells per session",
        )
        summary_axis.set_xticks([])
        summary_axis.legend(frameon=False)
        summary_buffer = io.BytesIO()
        summary_figure.savefig(summary_buffer, format="png", dpi=140)
        plt.close(summary_figure)
        optotagged_summary_png = summary_buffer.getvalue()
    else:
        tagged_rows = pd.DataFrame()
        optotagged_per_session = pd.Series(dtype=int)
        mean_optotagged = float("nan")
        optotagged_summary_png = None

    heatmap_session = mo.ui.dropdown(
        options=heatmap_options,
        value="ecephys_834687_2026-03-18_15-50-10",
        searchable=True,
        label="Session heatmap",
        full_width=True,
    )
    colorbar_limit = mo.ui.slider(
        start=0.5,
        stop=8.0,
        step=0.5,
        value=3.0,
        show_value=True,
        debounce=True,
        label="Symmetric colorbar limit (z-score)",
    )

    summary_items = [
        mo.md("# All-session optotagging results"),
        mo.ui.table(coverage, selection=None, label="Current run coverage"),
    ]
    if optotagged_summary_png is not None:
        summary_items.extend(
            [
                mo.md(
                    f"## Optotagged-cell yield\n"
                    f"**{len(tagged_rows):,} cells** meet the reference criterion: "
                    "`p < 0.05` and modulation index `> 0.1` in all three conditions. "
                    f"The mean is **{mean_optotagged:.1f} cells per session**."
                ),
            ]
        )
    summary_items.extend(
        [
            mo.md("# Interactive laser-aligned PSTH heatmap explorer"),
            mo.md(
                "Rows are sorted explicitly from strongest to weakest response. "
                "Raised cosine uses modulation index; 5 Hz and 40 Hz use mean firing "
                "rate during the exact 10 ms and 6 ms pulses."
            ),
            heatmap_session,
            colorbar_limit,
        ]
    )
    batch_summary_panel = mo.vstack(summary_items)
    return (
        batch_summary_panel,
        colorbar_limit,
        heatmap_options,
        heatmap_session,
        heatmap_session_id_by_path,
        optotagged_per_session,
        result_rows,
        tagged_rows,
    )


@app.cell(hide_code=True)
def anatomy_helpers(Path, h5py, np, pd, pickle, re, remfile):
    hierarchy_dir = Path.home() / "Documents" / "GitHub" / "psycode" / "utils"
    with (hierarchy_dir / "summary_structures.pkl").open("rb") as hierarchy_file:
        summary_structures = pickle.load(hierarchy_file)
    with (hierarchy_dir / "acronym_structure_path_map.pkl").open("rb") as hierarchy_file:
        acronym_structure_path_map = pickle.load(hierarchy_file)
    summary_id_to_acronym = dict(
        zip(summary_structures["id"], summary_structures["acronym"], strict=True)
    )

    def decode_anatomy_text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    def structure_and_parent(raw_acronym):
        structure_acronym = re.sub(r"(?:1|2/3|4|5|6a|6b)$", "", raw_acronym)
        structure_path = acronym_structure_path_map.get(
            raw_acronym, acronym_structure_path_map.get(structure_acronym, [])
        )
        parent_matches = [
            summary_id_to_acronym[structure_id]
            for structure_id in structure_path
            if structure_id in summary_id_to_acronym
        ]
        return (
            structure_acronym or raw_acronym,
            parent_matches[-1] if parent_matches else "root",
        )

    def read_asset_anatomy(asset):
        remote_file_value = remfile.File(asset["content_url"])
        try:
            with h5py.File(remote_file_value, mode="r") as anatomy_nwb:
                anatomy_units = anatomy_nwb["units"]
                anatomy_electrodes = anatomy_nwb[
                    "general/extracellular_ephys/electrodes"
                ]
                anatomy_labels = [
                    decode_anatomy_text(value)
                    for value in anatomy_units["decoder_label"][:]
                ]
                anatomy_unit_ids = (
                    [
                        decode_anatomy_text(value)
                        for value in anatomy_units["unit_name"][:]
                    ]
                    if "unit_name" in anatomy_units
                    else [
                        decode_anatomy_text(value)
                        for value in anatomy_units["id"][:]
                    ]
                )
                anatomy_devices = [
                    decode_anatomy_text(value)
                    for value in anatomy_units["device_name"][:]
                ]
                anatomy_extrema = np.asarray(
                    anatomy_units["extremum_channel_index"], dtype=np.int64
                )
                electrode_groups = np.asarray(
                    [
                        decode_anatomy_text(value)
                        for value in anatomy_electrodes["group_name"][:]
                    ]
                )
                electrode_areas = np.asarray(
                    [
                        decode_anatomy_text(value)
                        for value in anatomy_electrodes["location"][:]
                    ]
                )
                probe_electrodes = {
                    probe_name: np.flatnonzero(electrode_groups == probe_name)
                    for probe_name in set(anatomy_devices)
                }
                anatomy_session_id = decode_anatomy_text(
                    anatomy_nwb["general/session_id"][()]
                )
                anatomy_rows = []
                for unit_index, decoder_label in enumerate(anatomy_labels):
                    if decoder_label == "noise":
                        continue
                    probe_indices = probe_electrodes[anatomy_devices[unit_index]]
                    extremum_index = anatomy_extrema[unit_index]
                    raw_acronym = (
                        electrode_areas[probe_indices[extremum_index]]
                        if 0 <= extremum_index < len(probe_indices)
                        else "void"
                    )
                    structure_acronym, parent_area_value = structure_and_parent(
                        raw_acronym
                    )
                    anatomy_rows.append(
                        {
                            "session_id": anatomy_session_id,
                            "unit_id": anatomy_unit_ids[unit_index],
                            "parent_area": parent_area_value,
                            "structure_acronym": structure_acronym,
                        }
                    )
                return pd.DataFrame(anatomy_rows)
        finally:
            remote_file_value.close()


    return decode_anatomy_text, read_asset_anatomy, structure_and_parent


@app.cell(hide_code=True)
def anatomy_controls(mo):
    load_anatomy = mo.ui.run_button(
        label="Load anatomy for all-session yield plots",
        kind="success",
        full_width=True,
    )
    mo.vstack([
        mo.md(
            "Anatomy is read into memory without recomputing spike responses "
            "or writing another cache."
        ),
        load_anatomy,
    ])
    return (load_anatomy,)


@app.cell(hide_code=True)
def anatomy_loader(
    ThreadPoolExecutor,
    as_completed,
    assets,
    load_anatomy,
    mo,
    pd,
    read_asset_anatomy,
):
    all_unit_anatomy = pd.DataFrame(
        columns=["session_id", "unit_id", "parent_area", "structure_acronym"]
    )
    anatomy_failures = []
    if load_anatomy.value:
        anatomy_frames = []
        with ThreadPoolExecutor(max_workers=12) as anatomy_executor:
            anatomy_futures = {
                anatomy_executor.submit(read_asset_anatomy, asset): asset
                for asset in assets
            }
            completed_anatomy = mo.status.progress_bar(
                as_completed(anatomy_futures),
                total=len(anatomy_futures),
                title="Reading unit anatomy",
                completion_title="Anatomy ready",
            )
            for anatomy_future in completed_anatomy:
                anatomy_asset = anatomy_futures[anatomy_future]
                try:
                    anatomy_frames.append(anatomy_future.result())
                except (KeyError, OSError, RuntimeError, ValueError) as anatomy_error:
                    anatomy_failures.append({
                        "asset_path": anatomy_asset["asset_path"],
                        "reason": str(anatomy_error),
                    })
        if anatomy_frames:
            all_unit_anatomy = pd.concat(anatomy_frames, ignore_index=True)
    return (all_unit_anatomy,)


@app.cell(hide_code=True)
def anatomy_plot_data(
    all_unit_anatomy,
    optotagged_per_session,
    pd,
    result_rows,
    tagged_rows,
):
    anatomy_results = pd.DataFrame()
    overall_plot_data = pd.DataFrame()
    parent_plot_data = pd.DataFrame()
    structure_plot_data = pd.DataFrame()
    parent_order = []
    structure_order = []
    if not all_unit_anatomy.empty:
        anatomy_results = result_rows.merge(
            all_unit_anatomy,
            on=["session_id", "unit_id"],
            how="inner",
            validate="one_to_one",
        )
        tagged_keys = tagged_rows[["session_id", "unit_id"]].assign(
            is_optotagged=True
        )
        anatomy_results = anatomy_results.merge(
            tagged_keys, on=["session_id", "unit_id"], how="left"
        ).fillna({"is_optotagged": False})
        overall_plot_data = pd.DataFrame({
            "group": "All areas",
            "session_id": optotagged_per_session.index,
            "optotagged_cells": optotagged_per_session.to_numpy(),
        })
        anatomy_tagged = anatomy_results.loc[anatomy_results["is_optotagged"]]
        parent_sampled = anatomy_results[
            ["session_id", "parent_area"]
        ].drop_duplicates()
        parent_counts = (
            anatomy_tagged.groupby(["session_id", "parent_area"])
            .size().rename("optotagged_cells").reset_index()
        )
        parent_plot_data = parent_sampled.merge(
            parent_counts, on=["session_id", "parent_area"], how="left"
        ).fillna({"optotagged_cells": 0})
        structure_sampled = anatomy_results[
            ["session_id", "structure_acronym"]
        ].drop_duplicates()
        structure_counts = (
            anatomy_tagged.groupby(["session_id", "structure_acronym"])
            .size().rename("optotagged_cells").reset_index()
        )
        structure_plot_data = structure_sampled.merge(
            structure_counts,
            on=["session_id", "structure_acronym"],
            how="left",
        ).fillna({"optotagged_cells": 0})
        parent_order = (
            parent_plot_data.groupby("parent_area")["optotagged_cells"]
            .mean().sort_values(ascending=False).index.tolist()
        )
        structure_stats = structure_plot_data.groupby("structure_acronym").agg(
            mean_count=("optotagged_cells", "mean"),
            sessions=("session_id", "nunique"),
            total=("optotagged_cells", "sum"),
        )
        structure_order = (
            structure_stats.query("sessions >= 3 and total > 0")
            .sort_values("mean_count", ascending=False).index.tolist()
        )
        structure_plot_data = structure_plot_data.loc[
            structure_plot_data["structure_acronym"].isin(structure_order)
        ]
    return (
        anatomy_results,
        overall_plot_data,
        parent_order,
        parent_plot_data,
        structure_order,
        structure_plot_data,
    )


@app.cell(hide_code=True)
def anatomy_yield(
    anatomy_results,
    io,
    mo,
    overall_plot_data,
    parent_order,
    parent_plot_data,
    plt,
    sns,
    structure_order,
    structure_plot_data,
):
    if anatomy_results.empty:
        anatomy_yield_panel = mo.callout(
            "Click **Load anatomy for all-session yield plots**.",
            kind="info",
        )
    else:
        from iblatlas.regions import BrainRegions

        iblatlas_region_table = (
            BrainRegions().to_df().drop_duplicates("acronym")
        )
        iblatlas_colors = dict(
            zip(
                iblatlas_region_table["acronym"],
                iblatlas_region_table["hexcolor"],
                strict=True,
            )
        )
        overall_palette = {"All areas": "#8FBBD9"}
        parent_palette = {
            acronym: iblatlas_colors.get(acronym, "#A6A6A6")
            for acronym in parent_order
        }
        structure_palette = {
            acronym: iblatlas_colors.get(acronym, "#A6A6A6")
            for acronym in structure_order
        }
        yield_figure, yield_axes = plt.subplots(
            1, 3, figsize=(24, 6),
            gridspec_kw={"width_ratios": [1, 3, 6]},
            constrained_layout=True,
        )
        plot_specs = [
            (
                yield_axes[0], overall_plot_data, "group", ["All areas"],
                "Overall", overall_palette,
            ),
            (
                yield_axes[1], parent_plot_data, "parent_area",
                parent_order, "Major parent area", parent_palette,
            ),
            (
                yield_axes[2], structure_plot_data, "structure_acronym",
                structure_order, "Structure acronym", structure_palette,
            ),
        ]
        for (
            plot_axis,
            plot_data,
            x_column,
            category_order,
            plot_title,
            category_palette,
        ) in plot_specs:
            sns.barplot(
                data=plot_data,
                x=x_column,
                y="optotagged_cells",
                hue=x_column,
                order=category_order,
                hue_order=category_order,
                palette=category_palette,
                legend=False,
                errorbar="se",
                edgecolor="#2F4B5C",
                ax=plot_axis,
            )
            sns.stripplot(
                data=plot_data,
                x=x_column,
                y="optotagged_cells",
                order=category_order,
                color="#1F2933",
                alpha=0.55,
                jitter=0.22,
                size=3,
                ax=plot_axis,
            )
            plot_axis.set(
                title=plot_title,
                xlabel="",
                ylabel="Optotagged cells per sampled session",
            )
            plot_axis.tick_params(axis="x", rotation=90)
            sns.despine(ax=plot_axis)
        yield_figure.suptitle(
            "Optotagged-cell yield: mean and individual sessions",
            fontsize=15,
        )
        yield_buffer = io.BytesIO()
        yield_figure.savefig(yield_buffer, format="png", dpi=140)
        plt.close(yield_figure)
        anatomy_yield_panel = mo.vstack([
            mo.md(
                "## Optotagged-cell yield by anatomy\n"
                "Bars show mean +/- SEM; points are sampled sessions, "
                "including sessions with zero tagged cells."
            ),
            mo.image(
                yield_buffer.getvalue(),
                alt=(
                    "Bar and strip plots of optotagged yield overall, "
                    "by major parent area, and by structure acronym"
                ),
            ),
        ])
    return (anatomy_yield_panel,)


@app.cell(hide_code=True)
def _(
    analyze_asset,
    asset_by_path,
    decode_anatomy_text,
    h5py,
    heatmap_session,
    heatmap_session_id_by_path,
    mo,
    np,
    remfile,
    static_example_analysis,
    static_example_session_id,
    structure_and_parent,
):
    explorer_asset = asset_by_path[heatmap_session.value]
    explorer_session_id = heatmap_session_id_by_path[heatmap_session.value]

    with mo.status.spinner(title="Loading the selected session for interactive controls"):
        if explorer_session_id == static_example_session_id:
            explorer_analysis = static_example_analysis
        else:
            explorer_analysis = analyze_asset(explorer_asset)

        remote_file = remfile.File(explorer_asset["content_url"])
        try:
            with h5py.File(remote_file, mode="r") as explorer_nwb:
                units_table = explorer_nwb["units"]
                electrodes_table = explorer_nwb[
                    "general/extracellular_ephys/electrodes"
                ]
                decoder_labels = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in units_table["decoder_label"][:]
                ]
                electrode_refs = np.asarray(units_table["electrodes"], dtype=np.int64)
                electrode_ends = np.asarray(
                    units_table["electrodes_index"], dtype=np.int64
                )
                electrode_starts = np.concatenate(([0], electrode_ends[:-1]))
                extremum_channels = np.asarray(
                    units_table["extremum_channel_index"], dtype=np.int64
                )
                electrode_locations = electrodes_table["location"]

                selected_area_labels = []
                selected_structure_labels = []
                for unit_index, decoder_label in enumerate(decoder_labels):
                    if decoder_label == "noise":
                        continue
                    unit_electrodes = electrode_refs[
                        electrode_starts[unit_index] : electrode_ends[unit_index]
                    ]
                    extremum_channel = extremum_channels[unit_index]
                    if 0 <= extremum_channel < len(unit_electrodes):
                        raw_area = electrode_locations[
                            unit_electrodes[extremum_channel]
                        ]
                        raw_area = decode_anatomy_text(raw_area)
                    else:
                        raw_area = "void"
                    structure_label, parent_label = structure_and_parent(raw_area)
                    selected_structure_labels.append(structure_label)
                    selected_area_labels.append(parent_label)
        finally:
            remote_file.close()

    parent_area_options = ["All parent areas"] + sorted(
        {
            area
            for area in selected_area_labels
            if area and area.lower() not in {"void", "nan", "unknown"}
        }
    )
    parent_area = mo.ui.dropdown(
        options=parent_area_options,
        value="All parent areas",
        searchable=True,
        label="Parent area",
        full_width=True,
    )
    return (
        explorer_analysis,
        explorer_session_id,
        parent_area,
        selected_area_labels,
    )


@app.cell(hide_code=True)
def _(
    CONDITIONS,
    CONDITION_DISPLAY_NAMES,
    anatomy_yield_panel,
    baseline_zscore,
    batch_summary_panel,
    colorbar_limit,
    explorer_analysis,
    explorer_session_id,
    io,
    mo,
    np,
    parent_area,
    pd,
    plt,
    selected_area_labels,
    tagged_rows,
):
    if parent_area.value == "All parent areas":
        area_mask = np.ones(explorer_analysis.unit_count, dtype=bool)
    else:
        area_mask = np.asarray(selected_area_labels) == parent_area.value

    display_unit_count = int(area_mask.sum())
    heatmap_figure, heatmap_axes = plt.subplots(
        1,
        len(CONDITIONS),
        figsize=(15, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    heatmap_image = None
    for heatmap_axis, heatmap_condition in zip(
        heatmap_axes, CONDITIONS, strict=True
    ):
        zscored_rates = baseline_zscore(
            explorer_analysis.psths[heatmap_condition.table_name],
            explorer_analysis.time_seconds,
        )[area_mask]
        response_scores = explorer_analysis.pulse_firing_rates[
            heatmap_condition.table_name
        ]
        pulse_width_ms = heatmap_condition.pulse_width_seconds * 1000
        ordering_label = (
            f"mean firing rate during {pulse_width_ms:g} ms laser-on windows"
        )
        filtered_scores = np.nan_to_num(
            response_scores[area_mask],
            nan=-np.inf,
        )
        strongest_first = np.argsort(-filtered_scores, kind="stable")
        heatmap_image = heatmap_axis.imshow(
            zscored_rates[strongest_first],
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-float(colorbar_limit.value),
            vmax=float(colorbar_limit.value),
            extent=(
                explorer_analysis.time_seconds[0],
                explorer_analysis.time_seconds[-1],
                display_unit_count,
                0,
            ),
            origin="upper",
            rasterized=True,
        )
        heatmap_axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        heatmap_axis.set_title(
            f"{CONDITION_DISPLAY_NAMES[heatmap_condition.table_name]}\n"
            f"ordered by {ordering_label}"
        )
        heatmap_axis.set_xlabel("Time from laser onset (s)")
    heatmap_axes[0].set_ylabel("Units (strongest response at top)")
    heatmap_figure.suptitle(
        f"{explorer_session_id}: {parent_area.value} "
        f"({display_unit_count:,} non-noise units)"
    )
    if heatmap_image is not None:
        heatmap_figure.colorbar(
            heatmap_image,
            ax=heatmap_axes,
            label="Baseline z-scored firing rate",
        )
    heatmap_buffer = io.BytesIO()
    heatmap_figure.savefig(heatmap_buffer, format="png", dpi=120)
    plt.close(heatmap_figure)
    explorer_heatmap_png = heatmap_buffer.getvalue()

    selected_session_tagged = tagged_rows.loc[
        tagged_rows["session_id"].eq(explorer_session_id)
    ] if not tagged_rows.empty else tagged_rows
    session_tagged_ids = set(selected_session_tagged["unit_id"].astype(str))
    analysis_unit_ids = explorer_analysis.metrics["unit_id"].astype(str).to_numpy()
    tagged_area_counts = (
        pd.DataFrame(
            {
                "unit_id": analysis_unit_ids,
                "parent_area": selected_area_labels,
            }
        )
        .loc[lambda frame: frame["unit_id"].isin(session_tagged_ids)]
        .groupby("parent_area")
        .size()
        .sort_values(ascending=False)
        .rename("optotagged_cells")
        .reset_index()
    )

    heatmap_explorer_panel = mo.vstack(
        [
            mo.hstack([parent_area], justify="start"),
            mo.md(
                f"## {explorer_session_id}\n"
                f"**{display_unit_count:,} displayed units** / "
                f"**{len(selected_session_tagged):,} optotagged cells**"
            ),
            mo.image(
                explorer_heatmap_png,
                alt=f"{explorer_session_id} strongest-first laser response heatmaps",
            ),
            mo.md("### Optotagged cells by parent area in this session"),
            mo.ui.table(tagged_area_counts, selection=None),
        ]
    )

    batch_panel = mo.vstack([batch_summary_panel, anatomy_yield_panel, heatmap_explorer_panel])
    return (batch_panel,)


@app.cell(hide_code=True)
def _(analysis_mode, batch_panel, mo, single_panel):
    if analysis_mode.value == "Single session":
        visible_panel = mo.vstack(
            [
                batch_panel,
                mo.md("---"),
                mo.md("# Detailed single-session analysis"),
                single_panel,
            ]
        )
    else:
        visible_panel = batch_panel

    visible_panel
    return


@app.cell(hide_code=True)
def static_optotagging_composite(
    CONDITIONS,
    baseline_zscore,
    io,
    mo,
    np,
    overall_plot_data,
    parent_order,
    parent_plot_data,
    plt,
    sns,
    static_example_analysis,
    static_example_session_id,
    structure_order,
    structure_plot_data,
):
    if (
        overall_plot_data.empty
        or parent_plot_data.empty
        or structure_plot_data.empty
    ):
        static_composite_png = None
        static_composite_svg = None
        static_composite_panel = mo.callout(
            "Click **Load anatomy for all-session yield plots** to build the static figure.",
            kind="info",
        )
    else:

        def _build_static_optotagging_composite():
            from iblatlas.regions import BrainRegions

            _region_table = BrainRegions().to_df().drop_duplicates("acronym")
            _allen_colors = dict(
                zip(
                    _region_table["acronym"],
                    _region_table["hexcolor"],
                    strict=True,
                )
            )
            _parent_palette = {
                _area: _allen_colors.get(_area, "#A6A6A6")
                for _area in parent_order
            }
            _structure_palette = {
                _area: _allen_colors.get(_area, "#A6A6A6")
                for _area in structure_order
            }

            _figure = plt.figure(
                figsize=(28, 8.5),
                constrained_layout=True,
            )
            _grid = _figure.add_gridspec(
                2,
                4,
                height_ratios=[0.16, 1.0],
                width_ratios=[3.4, 1.0, 2.5, 5.4],
            )
            _pulse_axis = _figure.add_subplot(_grid[0, 0])
            _heatmap_axis = _figure.add_subplot(_grid[1, 0])
            _overall_axis = _figure.add_subplot(_grid[1, 1])
            _parent_axis = _figure.add_subplot(_grid[1, 2])
            _structure_axis = _figure.add_subplot(_grid[1, 3])

            _plot_specs = [
                (
                    _overall_axis,
                    overall_plot_data,
                    "group",
                    ["All areas"],
                    "Overall",
                    {"All areas": "#8FBBD9"},
                ),
                (
                    _parent_axis,
                    parent_plot_data,
                    "parent_area",
                    parent_order,
                    "Major parent area",
                    _parent_palette,
                ),
                (
                    _structure_axis,
                    structure_plot_data,
                    "structure_acronym",
                    structure_order,
                    "Structure acronym",
                    _structure_palette,
                ),
            ]
            for (
                _axis,
                _plot_data,
                _x_column,
                _category_order,
                _title,
                _palette,
            ) in _plot_specs:
                sns.barplot(
                    data=_plot_data,
                    x=_x_column,
                    y="optotagged_cells",
                    hue=_x_column,
                    order=_category_order,
                    hue_order=_category_order,
                    palette=_palette,
                    legend=False,
                    errorbar="se",
                    edgecolor="#2F4B5C",
                    ax=_axis,
                )
                sns.stripplot(
                    data=_plot_data,
                    x=_x_column,
                    y="optotagged_cells",
                    order=_category_order,
                    color="#1F2933",
                    alpha=0.55,
                    jitter=0.22,
                    size=2.8,
                    ax=_axis,
                )
                _axis.set(
                    title=_title,
                    xlabel="",
                    ylabel="Optotagged cells per session",
                )
                _axis.tick_params(axis="x", rotation=90)
                sns.despine(ax=_axis)

            _condition = next(
                _item
                for _item in CONDITIONS
                if _item.table_name == "5 hz pulse train_presentations"
            )
            _zscored = baseline_zscore(
                static_example_analysis.psths[_condition.table_name],
                static_example_analysis.time_seconds,
            )
            _scores = np.nan_to_num(
                static_example_analysis.pulse_firing_rates[_condition.table_name],
                nan=-np.inf,
            )
            _order = np.argsort(-_scores, kind="stable")
            _image = _heatmap_axis.imshow(
                _zscored[_order],
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-3.0,
                vmax=3.0,
                extent=(
                    static_example_analysis.time_seconds[0],
                    static_example_analysis.time_seconds[-1],
                    static_example_analysis.unit_count,
                    0,
                ),
                origin="upper",
                rasterized=True,
            )
            _heatmap_axis.axvline(
                0,
                color="#111111",
                linewidth=0.9,
                linestyle="--",
            )
            _heatmap_axis.set(
                title=(
                    "Example 5 Hz response\n"
                    "ordered by firing during exact 10 ms laser pulses"
                ),
                xlabel="Time from laser onset (s)",
                ylabel="Units (strongest response at top)",
            )
            _figure.colorbar(
                _image,
                ax=_heatmap_axis,
                orientation="horizontal",
                pad=0.12,
                fraction=0.06,
                label="Baseline z-scored firing rate",
            )

            _pulse_starts = np.arange(0.0, 1.0, 0.2)
            _pulse_axis.broken_barh(
                [(_start, 0.01) for _start in _pulse_starts],
                (0.2, 0.6),
                facecolors="#1696D2",
                edgecolors="#0B4F6C",
                linewidth=0.7,
            )
            _pulse_axis.set_xlim(
                static_example_analysis.time_seconds[0],
                static_example_analysis.time_seconds[-1],
            )
            _pulse_axis.set_ylim(0, 1)
            _pulse_axis.set_yticks([])
            _pulse_axis.set_xticks([])
            _pulse_axis.set_title(
                "5 Hz laser stimulation: five 10 ms pulses",
                fontsize=10,
                pad=2,
            )
            for _spine in _pulse_axis.spines.values():
                _spine.set_visible(False)

            _figure.suptitle(
                "Optotagged-cell yield and example laser-aligned response\n"
                f"{static_example_session_id}",
                fontsize=16,
            )
            _png_buffer = io.BytesIO()
            _svg_buffer = io.BytesIO()
            _figure.savefig(
                _png_buffer,
                format="png",
                dpi=180,
                bbox_inches="tight",
            )
            _figure.savefig(
                _svg_buffer,
                format="svg",
                bbox_inches="tight",
            )
            plt.close(_figure)
            return _png_buffer.getvalue(), _svg_buffer.getvalue()


        static_composite_png, static_composite_svg = (
            _build_static_optotagging_composite()
        )
        static_composite_panel = mo.vstack(
            [
                mo.md(
                    "## Static optotagging summary figure\n"
                    "The center heatmap shows the fixed example session's 5 Hz condition. "
                    "Rows are ordered by mean firing rate strictly inside the five "
                    "10 ms laser pulses shown above."
                ),
                mo.image(
                    static_composite_png,
                    alt=(
                        "Optotagged-cell yield summaries surrounding a 5 Hz "
                        "laser-aligned unit heatmap with five pulses above"
                    ),
                ),
                mo.hstack(
                    [
                        mo.download(
                            data=static_composite_png,
                            filename="optotagging-static-composite.png",
                            label="Download PNG",
                        ),
                        mo.download(
                            data=static_composite_svg,
                            filename="optotagging-static-composite.svg",
                            label="Download SVG",
                        ),
                    ],
                    justify="start",
                ),
            ]
        )

    static_composite_panel

    return


@app.cell(hide_code=True)
def static_example_loader(analyze_asset, asset_by_path, heatmap_options, mo):
    static_example_session_id = "ecephys_834687_2026-03-18_15-50-10"
    static_example_path = heatmap_options[static_example_session_id]
    static_example_asset = asset_by_path[static_example_path]
    with mo.status.spinner(title="Loading the fixed example session"):
        static_example_analysis = analyze_asset(static_example_asset)

    return static_example_analysis, static_example_session_id


if __name__ == "__main__":
    app.run()
