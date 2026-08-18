import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def imports_and_configuration():
    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import remfile
    from dandi.dandiapi import DandiAPIClient

    DANDISET_ID = "001637"
    DANDISET_VERSION = "draft"
    SESSION_TABLE_TO_NAME = {
        "Standard mismatch block_presentations": "Standard mismatch",
        "Sensory-motor mismatch block_presentations": "Sensorimotor mismatch",
        "Sequence mismatch block_presentations": "Sequence mismatch",
        "Duration mismatch block_presentations": "Duration mismatch",
    }
    SESSION_ORDER = list(SESSION_TABLE_TO_NAME.values())
    SESSION_COLORS = {
        "Standard mismatch": "#118ab2",
        "Sensorimotor mismatch": "#ef476f",
        "Sequence mismatch": "#7b2cbf",
        "Duration mismatch": "#06d6a0",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    return (
        DANDISET_ID,
        DANDISET_VERSION,
        DandiAPIClient,
        SESSION_COLORS,
        SESSION_ORDER,
        SESSION_TABLE_TO_NAME,
        h5py,
        mo,
        np,
        pd,
        plt,
        remfile,
    )


@app.cell(hide_code=True)
def notebook_header(mo):
    mo.md("""
    # OpenScope predictive processing dataset summary

    This notebook inventories canonical Neuropixels NWB files in DANDI dandiset
    **001637** and assigns each recording its semantic paradigm name from the
    paradigm-specific NWB interval table.
    """)
    return


@app.cell(hide_code=True)
def collect_session_metadata(
    DANDISET_ID,
    DANDISET_VERSION,
    DandiAPIClient,
    SESSION_TABLE_TO_NAME,
    h5py,
    pd,
    remfile,
):
    records = []
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID, version_id=DANDISET_VERSION)
        assets = [asset for asset in dandiset.get_assets() if asset.path.endswith("_ecephys.nwb")]
        for asset_number, asset in enumerate(assets, start=1):
            metadata = asset.get_metadata()
            subject = metadata.wasAttributedTo[0] if metadata.wasAttributedTo else None
            session = next(
                (activity for activity in metadata.wasGeneratedBy if activity.schemaKey == "Session"),
                None,
            )

            remote_file = remfile.File(asset.get_content_url(follow_redirects=1, strip_query=True))
            h5_file = h5py.File(remote_file, mode="r")
            try:
                interval_names = set(h5_file["intervals"])
            finally:
                h5_file.close()
                remote_file.close()

            matching_types = [
                semantic_name
                for table_name, semantic_name in SESSION_TABLE_TO_NAME.items()
                if table_name in interval_names
            ]
            if len(matching_types) != 1:
                raise ValueError(
                    f"Expected exactly one paradigm table in {asset.path}; found {matching_types}"
                )

            records.append(
                {
                    "subject": getattr(subject, "identifier", None),
                    "session_id": getattr(session, "identifier", None),
                    "session_date": (
                        session.startDate.date().isoformat()
                        if session is not None and session.startDate is not None
                        else None
                    ),
                    "session_type": matching_types[0],
                    "sex": getattr(getattr(subject, "sex", None), "name", None),
                    "genotype": getattr(subject, "genotype", None),
                    "species": getattr(getattr(subject, "species", None), "name", None),
                    "age": getattr(getattr(subject, "age", None), "value", None),
                    "path": asset.path,
                    "size_gb": metadata.contentSize / 1e9,
                }
            )
            if asset_number % 10 == 0 or asset_number == len(assets):
                print(f"Read {asset_number}/{len(assets)} NWB assets")

    meta_df = pd.DataFrame(records).sort_values(["subject", "session_date"]).reset_index(drop=True)
    meta_df
    return (meta_df,)


@app.cell(hide_code=True)
def build_summaries(SESSION_ORDER, meta_df):
    session_counts = meta_df["session_type"].value_counts().reindex(SESSION_ORDER, fill_value=0)
    chronological_sessions = meta_df.sort_values(["subject", "session_date"])
    first_session_by_subject = chronological_sessions.groupby("subject")["session_type"].first()
    cohort_by_subject = first_session_by_subject.map(
        {
            "Sensorimotor mismatch": "Motor cohort",
            "Sequence mismatch": "Sequence cohort",
        }
    )
    if cohort_by_subject.isna().any():
        unknown_subjects = cohort_by_subject[cohort_by_subject.isna()].index.tolist()
        raise ValueError(f"Could not infer training cohort for subjects: {unknown_subjects}")

    meta_df["training_cohort"] = meta_df["subject"].map(cohort_by_subject)
    subject_metadata = meta_df.groupby("subject").agg(
        training_cohort=("training_cohort", "first"),
        sessions=("path", "count"),
        genotype=("genotype", "first"),
        sex=("sex", "first"),
        species=("species", "first"),
    )
    session_dates = meta_df.pivot(index="subject", columns="session_type", values="session_date").reindex(
        columns=SESSION_ORDER
    )
    session_dates.columns.name = None
    animal_summary = subject_metadata.join(session_dates)
    animal_summary["complete_four_session_set"] = animal_summary[SESSION_ORDER].notna().all(axis=1)
    animal_summary = animal_summary.reset_index()

    complete_orders = (
        chronological_sessions.loc[
            chronological_sessions["subject"].isin(
                animal_summary.loc[animal_summary["complete_four_session_set"], "subject"]
            )
        ]
        .groupby("subject")["session_type"]
        .agg(list)
    )
    expected_orders = {
        "Motor cohort": [
            "Sensorimotor mismatch",
            "Standard mismatch",
            "Sequence mismatch",
            "Duration mismatch",
        ],
        "Sequence cohort": [
            "Sequence mismatch",
            "Duration mismatch",
            "Standard mismatch",
            "Sensorimotor mismatch",
        ],
    }
    for _subject, _observed_order in complete_orders.items():
        _cohort = cohort_by_subject.loc[_subject]
        if _observed_order != expected_orders[_cohort]:
            raise ValueError(f"Unexpected session order for {_subject}: {_observed_order}")

    dataset_summary = {
        "canonical_sessions": int(len(meta_df)),
        "animals": int(meta_df["subject"].nunique()),
        "complete_animals": int(animal_summary["complete_four_session_set"].sum()),
        "total_size_tb": float(meta_df["size_gb"].sum() / 1000),
        "genotypes": int(meta_df["genotype"].nunique()),
        "female_animals": int((animal_summary["sex"] == "Female").sum()),
        "male_animals": int((animal_summary["sex"] == "Male").sum()),
        "motor_cohort_animals": int((animal_summary["training_cohort"] == "Motor cohort").sum()),
        "sequence_cohort_animals": int((animal_summary["training_cohort"] == "Sequence cohort").sum()),
    }
    return animal_summary, dataset_summary, session_counts


@app.cell(hide_code=True)
def dataset_overview(dataset_summary, mo):
    mo.md(f"""
    ## Dataset overview

    | Canonical sessions | Animals | Complete 4-session sets | Data volume | Genotypes |
    |---:|---:|---:|---:|---:|
    | **{dataset_summary['canonical_sessions']}** | **{dataset_summary['animals']}** | **{dataset_summary['complete_animals']}** | **{dataset_summary['total_size_tb']:.2f} TB** | **{dataset_summary['genotypes']}** |

    **Training cohorts:** {dataset_summary['motor_cohort_animals']} motor,
    {dataset_summary['sequence_cohort_animals']} sequence.  
    **Animal sex:** {dataset_summary['female_animals']} female,
    {dataset_summary['male_animals']} male.
    """)
    return


@app.cell(hide_code=True)
def session_definitions(mo, pd):
    session_definitions = pd.DataFrame(
        [
            {
                "session_type": "Standard mismatch",
                "prediction": "Identity of a repeated drifting grating",
                "violation": "Orientation change, motion halt, or omission",
            },
            {
                "session_type": "Sensorimotor mismatch",
                "prediction": "Visual flow coupled to locomotion",
                "violation": "Transient visual-flow decoupling",
            },
            {
                "session_type": "Sequence mismatch",
                "prediction": "Third item in a fixed four-element sequence",
                "violation": "Unexpected replacement at sequence position 3",
            },
            {
                "session_type": "Duration mismatch",
                "prediction": "Regular stimulus duration",
                "violation": "Unexpectedly short or long presentation",
            },
        ]
    )
    mo.vstack([
        mo.md("## Semantic session types"),
        mo.ui.table(session_definitions, selection=None, pagination=False, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def cohort_definitions(mo, pd):
    cohort_definitions = pd.DataFrame(
        [
            {
                "training_cohort": "Motor cohort",
                "habituation_context": "Closed-loop visuomotor optic flow controlled by locomotion",
                "recording_order": "Sensorimotor -> Standard -> Sequence -> Duration",
            },
            {
                "training_cohort": "Sequence cohort",
                "habituation_context": "Passive repeating A-B-C-D-grey sequences",
                "recording_order": "Sequence -> Duration -> Standard -> Sensorimotor",
            },
        ]
    )
    mo.vstack([
        mo.md("## Training cohorts"),
        mo.ui.table(cohort_definitions, selection=None, pagination=False, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def coverage_visualization(
    SESSION_COLORS,
    SESSION_ORDER,
    animal_summary,
    np,
    plt,
    session_counts,
):
    availability = animal_summary.set_index("subject")[SESSION_ORDER].notna()
    _figure, (_count_axis, _matrix_axis) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1, 2.2]}, layout="constrained"
    )
    _count_colors = [SESSION_COLORS[name] for name in SESSION_ORDER]
    _count_axis.barh(SESSION_ORDER[::-1], session_counts.reindex(SESSION_ORDER[::-1]), color=_count_colors[::-1])
    _count_axis.set(title="Sessions by paradigm", xlabel="NWB sessions")
    _count_axis.spines[["top", "right"]].set_visible(False)
    for _bar in _count_axis.patches:
        _count_axis.text(
            _bar.get_width() + 0.15,
            _bar.get_y() + _bar.get_height() / 2,
            f"{int(_bar.get_width())}",
            va="center",
            fontweight="bold",
        )

    _matrix_axis.imshow(availability.to_numpy(dtype=int), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    _matrix_axis.set(
        title="Session coverage by animal",
        xlabel="Paradigm",
        ylabel="Subject",
        xticks=np.arange(len(SESSION_ORDER)),
        yticks=np.arange(len(availability)),
        xticklabels=[name.replace(" mismatch", "") for name in SESSION_ORDER],
        yticklabels=availability.index,
    )
    _matrix_axis.tick_params(axis="x", rotation=30)
    for _row in range(len(availability)):
        for _column in range(len(SESSION_ORDER)):
            _matrix_axis.text(
                _column,
                _row,
                "OK" if availability.iloc[_row, _column] else "--",
                ha="center",
                va="center",
                color="white" if availability.iloc[_row, _column] else "#6b7280",
                fontsize=8,
                fontweight="bold",
            )
    _figure
    return


@app.cell(hide_code=True)
def inventory_filters(SESSION_ORDER, animal_summary, mo):
    session_type_picker = mo.ui.dropdown(
        options=["All sessions", *SESSION_ORDER],
        value="All sessions",
        label="Session type",
    )
    cohort_picker = mo.ui.dropdown(
        options=["All cohorts", "Motor cohort", "Sequence cohort"],
        value="All cohorts",
        label="Training cohort",
    )
    subject_picker = mo.ui.dropdown(
        options=["All animals", *animal_summary["subject"].astype(str).tolist()],
        value="All animals",
        label="Animal",
        searchable=True,
    )
    mo.hstack([session_type_picker, cohort_picker, subject_picker], widths=[1, 1, 1])
    return cohort_picker, session_type_picker, subject_picker


@app.cell(hide_code=True)
def session_inventory(
    cohort_picker,
    meta_df,
    mo,
    session_type_picker,
    subject_picker,
):
    filtered_sessions = meta_df.copy()
    if session_type_picker.value != "All sessions":
        filtered_sessions = filtered_sessions.loc[
            filtered_sessions["session_type"] == session_type_picker.value
        ]
    if cohort_picker.value != "All cohorts":
        filtered_sessions = filtered_sessions.loc[
            filtered_sessions["training_cohort"] == cohort_picker.value
        ]
    if subject_picker.value != "All animals":
        filtered_sessions = filtered_sessions.loc[
            filtered_sessions["subject"].astype(str) == subject_picker.value
        ]
    filtered_sessions = filtered_sessions[
        [
            "subject",
            "training_cohort",
            "session_date",
            "session_type",
            "sex",
            "genotype",
            "size_gb",
            "session_id",
            "path",
        ]
    ].copy()
    filtered_sessions["size_gb"] = filtered_sessions["size_gb"].round(1)
    session_table = mo.ui.table(
        filtered_sessions,
        selection=None,
        pagination=True,
        page_size=20,
        show_column_summaries=True,
    )
    mo.vstack([
        mo.md(f"## Session inventory ({len(filtered_sessions)} rows)"),
        session_table,
    ])
    return


@app.cell(hide_code=True)
def animal_inventory(animal_summary, mo):
    animal_table = mo.ui.table(
        animal_summary,
        selection=None,
        pagination=True,
        page_size=20,
        show_column_summaries=True,
    )
    mo.vstack([
        mo.md("## Animal summary"),
        animal_table,
    ])
    return


@app.cell(hide_code=True)
def _(
    DANDISET_ID,
    DANDISET_VERSION,
    DandiAPIClient,
    h5py,
    meta_df,
    np,
    pd,
    remfile,
):
    UNIT_CACHE_PATH = __import__("pathlib").Path(
        r"C:\Users\Roberto\Data\openscope_p3_data_release_paper\unit-metadata-v1.parquet"
    )
    OPTO_RESULTS_PATH = __import__("pathlib").Path(
        r"C:\Users\Roberto\Data\openscope_p3_data_release_paper\optotagging-results.parquet"
    )
    WAVEFORM_DURATION_MS = 0.4
    TH_WAVEFORM_DURATION_MS = 0.28
    SST_P_VALUE_THRESHOLD = 0.05
    SST_MODULATION_THRESHOLD = 0.1


    def _decode_text_array(dataset):
        values = dataset[:]
        return np.array([
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ])


    def _allen_parent_lookup():
        import json
        import urllib.request

        url = (
            "https://api.brain-map.org/api/v2/"
            "structure_graph_download/1.json"
        )
        request = urllib.request.Request(
            url, headers={"User-Agent": "openscope-p3-data-summary"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            roots = json.load(response)["msg"]
        if len(roots) != 1:
            raise RuntimeError("Allen CCF structure graph did not return one root")

        major_areas = {
            "Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH",
            "HY", "MB", "P", "MY", "CB",
        }
        lookup = {}

        def visit(node, parent="root"):
            acronym = node["acronym"]
            current_parent = acronym if acronym in major_areas else parent
            lookup[acronym] = current_parent
            for child in node.get("children", []):
                visit(child, current_parent)

        visit(roots[0])
        return lookup


    def _extract_unit_metadata():
        parent_lookup = _allen_parent_lookup()
        session_rows = []
        with DandiAPIClient() as client:
            dandiset = client.get_dandiset(DANDISET_ID, version_id=DANDISET_VERSION)
            asset_by_path = {
                asset.path: asset
                for asset in dandiset.get_assets()
                if asset.path in set(meta_df["path"])
            }
            for session_number, row in enumerate(meta_df.itertuples(index=False), start=1):
                asset = asset_by_path[row.path]
                url = asset.get_content_url(follow_redirects=1, strip_query=True)
                remote_file = remfile.File(url)
                h5_file = h5py.File(remote_file, mode="r")
                try:
                    units = h5_file["units"]
                    electrodes = h5_file[
                        "general/extracellular_ephys/electrodes"
                    ]
                    group_names = _decode_text_array(electrodes["group_name"])
                    locations = _decode_text_array(electrodes["location"])
                    device_names = _decode_text_array(units["device_name"])
                    extremum_indices = units["extremum_channel_index"][:].astype(int)
                    rows_by_device = {
                        device: np.flatnonzero(group_names == device)
                        for device in np.unique(group_names)
                    }
                    peak_rows = np.array([
                        rows_by_device[device][channel_index]
                        if device in rows_by_device
                        and 0 <= channel_index < len(rows_by_device[device])
                        else -1
                        for device, channel_index in zip(
                            device_names, extremum_indices, strict=True
                        )
                    ])
                    if np.any(peak_rows < 0):
                        raise ValueError(
                            f"Could not map all extremum channels in {row.session_id}"
                        )
                    structure_acronyms = locations[peak_rows]
                    unit_count = len(units["id"])
                    session_rows.append(pd.DataFrame({
                        "subject": np.repeat(str(row.subject), unit_count),
                        "session_id": np.repeat(str(row.session_id), unit_count),
                        "session_type": np.repeat(row.session_type, unit_count),
                        "unit_id": _decode_text_array(units["unit_name"]),
                        "decoder_label": _decode_text_array(units["decoder_label"]),
                        "structure_acronym": structure_acronyms,
                        "parent_area": np.array([
                            parent_lookup.get(acronym, "root")
                            for acronym in structure_acronyms
                        ]),
                        "peak_to_valley_ms": units["peak_to_valley"][:] * 1000,
                        "firing_rate": units["firing_rate"][:],
                        "amplitude_cutoff": units["amplitude_cutoff"][:],
                        "isi_violations_ratio": units["isi_violations_ratio"][:],
                        "presence_ratio": units["presence_ratio"][:],
                    }))
                finally:
                    h5_file.close()
                    remote_file.close()
                if session_number % 10 == 0 or session_number == len(meta_df):
                    print(f"Read unit metadata for {session_number}/{len(meta_df)} sessions")
        metadata = pd.concat(session_rows, ignore_index=True)
        UNIT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(UNIT_CACHE_PATH, index=False)
        return metadata


    if UNIT_CACHE_PATH.exists():
        unit_metadata = pd.read_parquet(UNIT_CACHE_PATH)
    else:
        unit_metadata = _extract_unit_metadata()

    unit_metadata["structure_acronym_detailed"] = unit_metadata["structure_acronym"]
    _vis_mask = unit_metadata["structure_acronym"].str.startswith("VIS", na=False)
    unit_metadata.loc[_vis_mask, "parent_area"] = "Isocortex"
    unit_metadata.loc[
        unit_metadata["structure_acronym"] == "VL", "parent_area"
    ] = "TH"
    _cortex_mask = unit_metadata["parent_area"] == "Isocortex"
    unit_metadata.loc[_cortex_mask, "structure_acronym"] = (
        unit_metadata.loc[_cortex_mask, "structure_acronym"]
        .str.replace(r"(?:2/3|6a|6b|1|4|5|6)$", "", regex=True)
    )

    if not OPTO_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"OpenScope P3 optotagging results not found: {OPTO_RESULTS_PATH}"
        )
    optotagging_results = pd.read_parquet(OPTO_RESULTS_PATH)
    _sst_mask = (
        optotagging_results[
            "5 hz pulse train_presentations__p_value"
        ] < SST_P_VALUE_THRESHOLD
    ) & (
        optotagging_results[
            "5 hz pulse train_presentations__modulation_index"
        ] > SST_MODULATION_THRESHOLD
    )
    _sst_keys = pd.MultiIndex.from_frame(
        optotagging_results.loc[_sst_mask, ["session_id", "unit_id"]].astype(str)
    )
    _unit_keys = pd.MultiIndex.from_frame(
        unit_metadata[["session_id", "unit_id"]].astype(str)
    )

    unit_df = unit_metadata.loc[
        (unit_metadata["decoder_label"] != "noise")
        & (unit_metadata["firing_rate"] > 1)
        & (unit_metadata["amplitude_cutoff"] < 0.1)
        & (unit_metadata["isi_violations_ratio"] < 0.5)
        & (unit_metadata["presence_ratio"] > 0.9)
    ].copy()
    unit_df["neuron_type"] = np.where(
        unit_df["peak_to_valley_ms"] > WAVEFORM_DURATION_MS, "RS", "FS"
    )
    _th_mask = unit_df["parent_area"] == "TH"
    unit_df.loc[_th_mask, "neuron_type"] = np.where(
        unit_df.loc[_th_mask, "peak_to_valley_ms"] <= TH_WAVEFORM_DURATION_MS,
        "FS",
        "RS",
    )
    unit_df.loc[unit_df["parent_area"] == "STR", "neuron_type"] = "RS"
    unit_df.loc[_unit_keys.isin(_sst_keys)[unit_df.index], "neuron_type"] = "SST"
    plot_unit_df = unit_df.loc[unit_df["parent_area"] != "root"].copy()
    unit_df
    return plot_unit_df, unit_df


@app.cell(hide_code=True)
def _(mo, plot_unit_df, unit_df):
    mo.md(f"""
    ## Recorded neurons by anatomical area and cell class

    The dataset contains **{len(unit_df):,} QC-passing unit-session observations** from
    {unit_df['session_id'].nunique()} sessions. Units are mapped to the electrode at
    their waveform extremum and grouped using the Allen CCF hierarchy. The plots show **{len(plot_unit_df):,} anatomically assigned units**; **{len(unit_df) - len(plot_unit_df):,}** unresolved, void, or fiber-tract labels are omitted. Because units
    cannot be tracked across recording days, these are recording-session unit counts,
    not unique cells followed longitudinally.

    QC requires firing rate > 1 Hz, amplitude cutoff < 0.1, ISI violations ratio < 0.5,
    and presence ratio > 0.9. RS/FS labels use peak-to-valley duration (0.4 ms; 0.28 ms
    in thalamus, with striatal units assigned RS). SST overrides come from the saved
    OpenScope P3 optotagging analysis: 5 Hz pulse-train p < 0.05 and modulation index > 0.1.
    """)
    return


@app.cell(hide_code=True)
def _(plot_unit_df, plt):
    parent_counts = (
        plot_unit_df.groupby("parent_area", observed=True)
        .size()
        .sort_values()
    )
    structure_counts = (
        plot_unit_df.groupby(["parent_area", "structure_acronym"], observed=True)
        .size()
        .rename("unit_count")
        .reset_index()
        .sort_values(["parent_area", "unit_count", "structure_acronym"])
    )
    _parent_order = parent_counts.sort_values(ascending=False).index.tolist()
    _parent_colors = {
        parent: plt.get_cmap("tab20")(_index % 20)
        for _index, parent in enumerate(_parent_order)
    }
    _structure_order = structure_counts["structure_acronym"].tolist()
    _fig_height = max(8, 0.22 * len(_structure_order))
    area_count_figure, _axes = plt.subplots(
        1, 2,
        figsize=(15, _fig_height),
        gridspec_kw={"width_ratios": [0.7, 2.3]},
        layout="constrained",
    )
    _axes[0].barh(
        parent_counts.index,
        parent_counts.values,
        color=[_parent_colors[parent] for parent in parent_counts.index],
    )
    _axes[0].set(title="Parent area", xlabel="QC-passing units", ylabel="")
    _axes[0].bar_label(_axes[0].containers[0], padding=3, fontsize=8, fmt="{:.0f}")
    _axes[1].barh(
        structure_counts["structure_acronym"],
        structure_counts["unit_count"],
        color=[_parent_colors[parent] for parent in structure_counts["parent_area"]],
    )
    _axes[1].set(title="Structure acronym", xlabel="QC-passing units", ylabel="")
    for _axis in _axes:
        _axis.spines[["top", "right"]].set_visible(False)
    area_count_figure
    return (parent_counts,)


@app.cell(hide_code=True)
def _(np, parent_counts, plot_unit_df, plt):
    cell_type_counts = (
        plot_unit_df.groupby(["parent_area", "neuron_type"], observed=True)
        .size()
        .rename("unit_count")
        .reset_index()
    )
    _cell_order = ["RS", "FS", "SST"]
    _cell_colors = {"RS": "#4c78a8", "FS": "#f58518", "SST": "#54a24b"}
    _plot_parents = parent_counts.sort_values(ascending=False).index.tolist()
    _x = np.arange(len(_plot_parents))
    _width = 0.25
    cell_type_figure, _axis = plt.subplots(
        figsize=(max(10, 0.9 * len(_plot_parents)), 5.2), layout="constrained"
    )
    for _offset, _cell_type in enumerate(_cell_order):
        _values = (
            cell_type_counts.loc[cell_type_counts["neuron_type"] == _cell_type]
            .set_index("parent_area")["unit_count"]
            .reindex(_plot_parents, fill_value=0)
        )
        _bars = _axis.bar(
            _x + (_offset - 1) * _width,
            _values,
            width=_width,
            label=_cell_type,
            color=_cell_colors[_cell_type],
        )
        _axis.bar_label(_bars, padding=2, fontsize=7, rotation=90, fmt="{:.0f}")
    _axis.set_xticks(_x, _plot_parents, rotation=45, ha="right")
    _axis.set(
        title="Waveform and optotagging cell classes by parent area",
        xlabel="Parent area",
        ylabel="QC-passing units",
    )
    _axis.legend(title="Neuron type", frameon=False)
    _axis.spines[["top", "right"]].set_visible(False)
    cell_type_figure
    return


if __name__ == "__main__":
    app.run()
