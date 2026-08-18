import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def imports_and_configuration():
    import json
    from functools import lru_cache
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from dandi.dandiapi import DandiAPIClient

    DATA_ROOT = Path(r"C:\Users\Roberto\Data\sensorimotor_mismatch_psths")
    DANDISET_ID = "001637"
    DANDISET_VERSION = "draft"
    CONDITION_LABELS = {
        "motor_halt": "Motor halt",
        "motor_omission": "Motor omission",
        "motor_orientation": "Motor orientation (45 + 90 deg)",
    }
    CONDITION_COLORS = {
        "motor_halt": "#ef476f",
        "motor_omission": "#118ab2",
        "motor_orientation": "#7b2cbf",
    }
    MISMATCH_DURATION_S = 0.343
    plt.style.use("seaborn-v0_8-whitegrid")
    return (
        CONDITION_COLORS,
        CONDITION_LABELS,
        DANDISET_ID,
        DANDISET_VERSION,
        DATA_ROOT,
        DandiAPIClient,
        MISMATCH_DURATION_S,
        Path,
        json,
        lru_cache,
        mo,
        np,
        pd,
        plt,
        pq,
    )


@app.cell(hide_code=True)
def data_helpers(MISMATCH_DURATION_S, Path, lru_cache, np, pd, pq):
    def _table_to_frame(table):
        return pd.DataFrame(table.to_pydict())


    @lru_cache(maxsize=32)
    def load_unit_metadata(session_dir):
        return _table_to_frame(pq.read_table(Path(session_dir) / "units.parquet"))


    @lru_cache(maxsize=128)
    def load_population_summary(session_dir, condition):
        parquet_file = pq.ParquetFile(Path(session_dir) / f"{condition}.parquet")
        trace_sum = None
        trace_sum_sq = None
        unit_count = 0
        time_s = None
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group, columns=["unit_id", "time_s", "sdf_hz"])
            payload = table.to_pydict()
            unit_ids = np.asarray(payload["unit_id"], dtype=np.int64)
            values = np.asarray(payload["sdf_hz"], dtype=np.float64)
            group_units = np.unique(unit_ids)
            samples_per_unit = len(values) // len(group_units)
            traces = values.reshape(len(group_units), samples_per_unit)
            if time_s is None:
                time_s = np.asarray(payload["time_s"], dtype=np.float64)[:samples_per_unit]
                trace_sum = np.zeros(samples_per_unit, dtype=np.float64)
                trace_sum_sq = np.zeros(samples_per_unit, dtype=np.float64)
            trace_sum += traces.sum(axis=0)
            trace_sum_sq += np.square(traces).sum(axis=0)
            unit_count += len(group_units)
        mean = trace_sum / unit_count
        variance = np.maximum(trace_sum_sq / unit_count - np.square(mean), 0)
        sem = np.sqrt(variance) / np.sqrt(unit_count)
        return time_s, mean, sem, unit_count


    @lru_cache(maxsize=6)
    def load_condition_matrix(session_dir, condition):
        table = pq.read_table(
            Path(session_dir) / f"{condition}.parquet",
            columns=["unit_id", "time_s", "sdf_hz"],
        )
        payload = table.to_pydict()
        unit_ids = np.asarray(payload["unit_id"], dtype=np.int64)
        unique_ids = pd.unique(unit_ids)
        samples_per_unit = len(unit_ids) // len(unique_ids)
        id_matrix = unit_ids.reshape(len(unique_ids), samples_per_unit)
        if not np.all(id_matrix == id_matrix[:, :1]):
            raise ValueError(f"SDF rows are not grouped by unit: {session_dir} / {condition}")
        time_matrix = np.asarray(payload["time_s"], dtype=float).reshape(
            len(unique_ids), samples_per_unit
        )
        traces = np.asarray(payload["sdf_hz"], dtype=float).reshape(
            len(unique_ids), samples_per_unit
        )
        return id_matrix[:, 0], time_matrix[0], traces


    @lru_cache(maxsize=256)
    def load_unit_trace(session_dir, condition, unit_id):
        table = pq.read_table(
            Path(session_dir) / f"{condition}.parquet",
            columns=["time_s", "sdf_hz"],
            filters=[("unit_id", "=", int(unit_id))],
        )
        payload = table.to_pydict()
        return np.asarray(payload["time_s"], dtype=float), np.asarray(payload["sdf_hz"], dtype=float)


    @lru_cache(maxsize=64)
    def load_heatmap(session_dir, condition, unit_count):
        units = load_unit_metadata(session_dir).nlargest(unit_count, "firing_rate")
        selected_ids = units["unit_id"].astype(int).tolist()
        table = pq.read_table(
            Path(session_dir) / f"{condition}.parquet",
            columns=["unit_id", "time_s", "sdf_hz"],
            filters=[("unit_id", "in", selected_ids)],
        )
        payload = table.to_pydict()
        row_ids = np.asarray(payload["unit_id"], dtype=np.int64)
        unique_ids, first_indices = np.unique(row_ids, return_index=True)
        unique_ids = unique_ids[np.argsort(first_indices)]
        samples_per_unit = len(row_ids) // len(unique_ids)
        time_s = np.asarray(payload["time_s"], dtype=float)[:samples_per_unit]
        traces = np.asarray(payload["sdf_hz"], dtype=float).reshape(len(unique_ids), samples_per_unit)
        return unique_ids, time_s, traces


    def baseline_correct(time_s, values):
        baseline_mask = time_s < 0
        if values.ndim == 1:
            return values - values[baseline_mask].mean()
        return values - values[:, baseline_mask].mean(axis=1, keepdims=True)


    def style_response_axis(axis, title, ylabel):
        axis.axvspan(0, MISMATCH_DURATION_S, color="#ffd166", alpha=0.22, lw=0)
        axis.axvline(0, color="#1f2937", lw=1.1, ls="--", alpha=0.8)
        axis.axvline(MISMATCH_DURATION_S, color="#6b7280", lw=0.9, ls=":", alpha=0.8)
        axis.set(xlabel="Time from mismatch onset (s)", ylabel=ylabel, title=title)
        axis.spines[["top", "right"]].set_visible(False)

    return (
        baseline_correct,
        load_condition_matrix,
        load_heatmap,
        load_population_summary,
        load_unit_metadata,
        load_unit_trace,
        style_response_axis,
    )


@app.cell(hide_code=True)
def load_manifest(DATA_ROOT, json, pd):
    _batch_manifest_path = DATA_ROOT / "batch_manifest.json"
    if not _batch_manifest_path.is_file():
        raise FileNotFoundError(f"Missing SDF batch manifest: {_batch_manifest_path}")
    with _batch_manifest_path.open(encoding="utf-8") as _stream:
        batch_manifest = json.load(_stream)

    sessions_df = pd.DataFrame(batch_manifest["sessions"])
    sessions_df["label"] = sessions_df.apply(
        lambda row: f"Mouse {row['subject']} | {row['session_date']} | {row['qc_units']:,} units",
        axis=1,
    )
    SESSION_PATHS = dict(zip(sessions_df["label"], sessions_df["output"], strict=True))
    SESSION_ROWS = sessions_df.set_index("label").to_dict(orient="index")
    return SESSION_PATHS, SESSION_ROWS, sessions_df


@app.cell(hide_code=True)
def dashboard_header(mo):
    mo.md("""
    # Sensorimotor mismatch SDF explorer

    Explore trial-averaged spike-density functions for **motor halt**, **motor omission**,
    and pooled **45/90 degree motor orientation** mismatch events. The shaded interval
    marks the 343 ms mismatch period; dashed lines mark onset and offset.
    """)
    return


@app.cell(hide_code=True)
def dashboard_controls(CONDITION_LABELS, SDF_PARENT_AREAS, SESSION_PATHS, mo):
    session_picker = mo.ui.dropdown(
        options=list(SESSION_PATHS), value=list(SESSION_PATHS)[0],
        label="Recording session", full_width=True,
    )
    view_picker = mo.ui.radio(
        options=["Population comparison", "Single unit", "Population heatmap"],
        value="Population comparison", label="View", inline=True,
    )
    condition_picker = mo.ui.dropdown(
        options=list(CONDITION_LABELS.values()), value=CONDITION_LABELS["motor_halt"],
        label="Heatmap condition",
    )
    baseline_picker = mo.ui.switch(value=True, label="Subtract pre-event baseline")
    sem_picker = mo.ui.switch(value=True, label="Show population SEM")
    heatmap_size_picker = mo.ui.slider(
        start=50, stop=400, step=25, value=200,
        label="Heatmap units (highest firing rate)", show_value=True,
    )
    trace_stratify_toggle = mo.ui.switch(
        value=False,
        label="Facet population SDFs into separate axes",
    )
    trace_stratify_picker = mo.ui.radio(
        options=["Parent area", "Structure acronym", "Neuron type"],
        value="Parent area",
        label="Facet by",
        inline=True,
    )
    trace_parent_areas_picker = mo.ui.multiselect(
        options=SDF_PARENT_AREAS,
        value=SDF_PARENT_AREAS,
        label="Include parent areas",
        full_width=True,
    )

    return (
        baseline_picker,
        condition_picker,
        heatmap_size_picker,
        sem_picker,
        session_picker,
        trace_parent_areas_picker,
        trace_stratify_picker,
        trace_stratify_toggle,
        view_picker,
    )


@app.cell(hide_code=True)
def unit_selector(
    SESSION_PATHS,
    SESSION_ROWS,
    annotated_units,
    load_unit_metadata,
    mo,
    session_picker,
):
    selected_session_dir = SESSION_PATHS[session_picker.value]
    selected_session = SESSION_ROWS[session_picker.value]
    selected_units = (
        load_unit_metadata(selected_session_dir)
        .merge(
            annotated_units.loc[
                annotated_units["session_id"] == selected_session["session_id"],
                ["unit_id", "structure_acronym", "parent_area", "neuron_type"],
            ],
            on="unit_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("firing_rate", ascending=False)
    )
    unit_labels = [
        f"Unit {int(row.unit_id)} | {row.firing_rate:.1f} Hz"
        for row in selected_units.itertuples(index=False)
    ]
    UNIT_IDS = {
        label: int(row.unit_id)
        for label, row in zip(unit_labels, selected_units.itertuples(index=False), strict=True)
    }
    unit_picker = mo.ui.dropdown(
        options=unit_labels, value=unit_labels[0],
        label="Unit (sorted by firing rate)", searchable=True, full_width=True,
    )
    mo.vstack([
        mo.md(
            f"**Mouse {selected_session['subject']}** | {selected_session['session_date']} | "
            f"{selected_session['qc_units']:,} QC units"
        ),
        unit_picker,
    ], gap=0.5)
    return (
        UNIT_IDS,
        selected_session,
        selected_session_dir,
        selected_units,
        unit_picker,
    )


@app.cell(hide_code=True)
def interactive_plot(
    CONDITION_COLORS,
    CONDITION_LABELS,
    MIN_CELLS_PER_SESSION,
    MISMATCH_DURATION_S,
    UNIT_IDS,
    baseline_correct,
    baseline_picker,
    condition_picker,
    heatmap_size_picker,
    load_condition_matrix,
    load_heatmap,
    load_population_summary,
    load_unit_trace,
    mo,
    np,
    plt,
    selected_session,
    selected_session_dir,
    selected_units,
    sem_picker,
    session_picker,
    style_response_axis,
    trace_parent_areas_picker,
    trace_stratify_picker,
    trace_stratify_toggle,
    unit_picker,
    view_picker,
):
    _view = view_picker.value
    _baseline = baseline_picker.value
    _selected_condition = next(
        key for key, label in CONDITION_LABELS.items() if label == condition_picker.value
    )

    if _view == "Population comparison":
        if trace_stratify_toggle.value:
            _stratify_column = {
                "Parent area": "parent_area",
                "Structure acronym": "structure_acronym",
                "Neuron type": "neuron_type",
            }[trace_stratify_picker.value]
            _facet_units = selected_units.loc[
                selected_units["parent_area"].isin(
                    trace_parent_areas_picker.value
                )
            ].dropna(subset=[_stratify_column])
            _facet_counts = _facet_units[_stratify_column].value_counts()
            _facet_levels = _facet_counts.loc[
                _facet_counts >= MIN_CELLS_PER_SESSION
            ].index.tolist()
            _facet_levels = sorted(
                _facet_levels,
                key=lambda level: (-int(_facet_counts[level]), str(level)),
            )

            if _facet_levels:
                _column_count = 2 if len(_facet_levels) > 4 else 1
                _row_count = int(np.ceil(len(_facet_levels) / _column_count))
                _figure, _axes = plt.subplots(
                    _row_count,
                    _column_count,
                    figsize=(14 if _column_count == 2 else 12, 3.2 * _row_count),
                    sharex=True,
                    sharey=True,
                    squeeze=False,
                    layout="constrained",
                )
                _axes_flat = _axes.ravel()
                for _facet_index, _facet_level in enumerate(_facet_levels):
                    _axis = _axes_flat[_facet_index]
                    _facet_ids = _facet_units.loc[
                        _facet_units[_stratify_column] == _facet_level,
                        "unit_id",
                    ].astype(int).to_numpy()
                    for _condition, _label in CONDITION_LABELS.items():
                        _all_ids, _time, _all_traces = load_condition_matrix(
                            selected_session_dir, _condition
                        )
                        _traces = _all_traces[np.isin(_all_ids, _facet_ids)]
                        if _baseline:
                            _traces = baseline_correct(_time, _traces)
                        _mean = _traces.mean(axis=0)
                        _sem = (
                            _traces.std(axis=0, ddof=1) / np.sqrt(len(_traces))
                            if len(_traces) > 1
                            else np.zeros_like(_mean)
                        )
                        _color = CONDITION_COLORS[_condition]
                        _axis.plot(
                            _time, _mean, color=_color, lw=1.8, label=_label
                        )
                        if sem_picker.value:
                            _axis.fill_between(
                                _time,
                                _mean - _sem,
                                _mean + _sem,
                                color=_color,
                                alpha=0.14,
                                linewidth=0,
                            )
                    style_response_axis(
                        _axis,
                        f"{_facet_level} | n={len(_facet_ids):,}",
                        "Baseline-corrected SDF (Hz)" if _baseline else "SDF (Hz)",
                    )
                    _axis.set_xlim(-0.5, 1.0)
                    if _facet_index == 0:
                        _axis.legend(frameon=False, fontsize=8, loc="upper right")
                for _unused_axis in _axes_flat[len(_facet_levels):]:
                    _unused_axis.set_visible(False)
                _figure.suptitle(
                    f"Population responses faceted by {trace_stratify_picker.value.lower()} | "
                    f"Mouse {selected_session['subject']}",
                    fontsize=14,
                )
            else:
                _figure, _axis = plt.subplots(
                    figsize=(12, 4), layout="constrained"
                )
                _axis.text(
                    0.5,
                    0.5,
                    "No selected strata contain at least 5 units in this session",
                    ha="center",
                    va="center",
                    transform=_axis.transAxes,
                )
                _axis.set_axis_off()
        else:
            _figure, _axis = plt.subplots(figsize=(12, 6.2), layout="constrained")
            for _condition, _label in CONDITION_LABELS.items():
                _time, _mean, _sem, _unit_count = load_population_summary(
                    selected_session_dir, _condition
                )
                if _baseline:
                    _mean = baseline_correct(_time, _mean)
                _color = CONDITION_COLORS[_condition]
                _axis.plot(_time, _mean, color=_color, lw=2.2, label=_label)
                if sem_picker.value:
                    _axis.fill_between(
                        _time,
                        _mean - _sem,
                        _mean + _sem,
                        color=_color,
                        alpha=0.16,
                        linewidth=0,
                    )
            style_response_axis(
                _axis,
                f"Population response | Mouse {selected_session['subject']}",
                "Baseline-corrected SDF (Hz)" if _baseline else "SDF (Hz)",
            )
            _axis.legend(frameon=False, ncol=3, loc="upper right")
            _axis.set_xlim(-0.5, 1.0)

    elif _view == "Single unit":
        _unit_id = UNIT_IDS[unit_picker.value]
        _figure, _axis = plt.subplots(figsize=(12, 6.2), layout="constrained")
        for _condition, _label in CONDITION_LABELS.items():
            _time, _trace = load_unit_trace(selected_session_dir, _condition, _unit_id)
            if _baseline:
                _trace = baseline_correct(_time, _trace)
            _axis.plot(_time, _trace, color=CONDITION_COLORS[_condition], lw=2, label=_label)
        style_response_axis(
            _axis, f"Unit {_unit_id} | Mouse {selected_session['subject']}",
            "Baseline-corrected SDF (Hz)" if _baseline else "SDF (Hz)",
        )
        _axis.legend(frameon=False, ncol=3, loc="upper right")
        _axis.set_xlim(-0.5, 1.0)

    else:
        _heat_ids, _time, _matrix = load_heatmap(
            selected_session_dir, _selected_condition, int(heatmap_size_picker.value)
        )
        if _baseline:
            _matrix = baseline_correct(_time, _matrix)
        _response_mask = (_time >= 0) & (_time <= 0.5)
        _baseline_mask = _time < 0
        _scores = _matrix[:, _response_mask].mean(axis=1)
        if not _baseline:
            _scores -= _matrix[:, _baseline_mask].mean(axis=1)
        _order = np.argsort(_scores)[::-1]
        _matrix = _matrix[_order]
        _heat_ids = _heat_ids[_order]
        _figure, _axis = plt.subplots(figsize=(12, 7.2), layout="constrained")
        if _baseline:
            _limit = max(float(np.nanpercentile(np.abs(_matrix), 98)), 1e-6)
            _image = _axis.imshow(
                _matrix, aspect="auto", interpolation="nearest",
                extent=[_time[0], _time[-1], len(_matrix), 0],
                cmap="RdBu_r", vmin=-_limit, vmax=_limit,
            )
            _colorbar_label = "Baseline-corrected SDF (Hz)"
        else:
            _upper = max(float(np.nanpercentile(_matrix, 99)), 1e-6)
            _image = _axis.imshow(
                _matrix, aspect="auto", interpolation="nearest",
                extent=[_time[0], _time[-1], len(_matrix), 0],
                cmap="magma", vmin=0, vmax=_upper,
            )
            _colorbar_label = "SDF (Hz)"
        _axis.axvline(0, color="white", lw=1.1, ls="--")
        _axis.axvline(MISMATCH_DURATION_S, color="white", lw=0.9, ls=":")
        _axis.set(
            xlim=(-0.5, 1.0), xlabel="Time from mismatch onset (s)",
            ylabel="Units ranked by 0-500 ms response",
            title=f"{CONDITION_LABELS[_selected_condition]} heatmap | Mouse {selected_session['subject']}",
        )
        _figure.colorbar(_image, ax=_axis, label=_colorbar_label, shrink=0.86)

    mo.vstack([
        mo.hstack([session_picker, condition_picker], widths=[2, 1]),
        view_picker,
        mo.hstack([baseline_picker, sem_picker, heatmap_size_picker]),
        mo.md("**Population SDF faceting**"),
        mo.hstack([trace_stratify_toggle, trace_stratify_picker]),
        trace_parent_areas_picker,_figure
    ], gap=1.2)

    return


@app.cell(hide_code=True)
def selection_details(
    UNIT_IDS,
    mo,
    selected_session,
    selected_units,
    unit_picker,
):
    _selected_unit_id = UNIT_IDS[unit_picker.value]
    _selected_unit_row = selected_units.loc[selected_units["unit_id"] == _selected_unit_id].iloc[0]
    mo.md(
        f"""
        ### Selection details

        | Session | Unit | Firing rate | Amplitude cutoff | ISI violations | Presence ratio |
        |---|---:|---:|---:|---:|---:|
        | Mouse {selected_session['subject']} / {selected_session['session_date']} |
        {_selected_unit_id} | {_selected_unit_row['firing_rate']:.2f} Hz |
        {_selected_unit_row['amplitude_cutoff']:.3f} |
        {_selected_unit_row['isi_violations_ratio']:.3f} |
        {_selected_unit_row['presence_ratio']:.3f} |

        **SDF parameters:** -1 to +1.5 s | 1 ms sampling | 5 ms causal exponential filter |
        35 halt trials | 35 omission trials | 70 pooled orientation trials.
        """
    )
    return


@app.cell(hide_code=True)
def _(
    CONDITION_LABELS,
    DANDISET_ID,
    DANDISET_VERSION,
    DATA_ROOT,
    DandiAPIClient,
    MISMATCH_DURATION_S,
    Path,
    SESSION_PATHS,
    SESSION_ROWS,
    np,
    pd,
    pq,
    sessions_df,
):
    UNIT_METADATA_CACHE_PATH = Path(
        r"C:\Users\Roberto\Data\openscope_p3_data_release_paper\unit-metadata-v1.parquet"
    )
    OPTO_RESULTS_PATH = Path(
        r"C:\Users\Roberto\Data\openscope_p3_data_release_paper\optotagging-results.parquet"
    )
    RESPONSE_SUMMARY_PATH = DATA_ROOT / "baseline-response-summary-v1.parquet"
    SESSION_AGE_PATH = DATA_ROOT / "session-age-metadata-v1.parquet"
    SUMMARY_BASELINE_WINDOW = (-1.0, 0.0)
    SUMMARY_RESPONSE_WINDOW = (0.0, MISMATCH_DURATION_S)
    MIN_CELLS_PER_SESSION = 5
    MIN_SESSIONS_PER_AREA = 3


    def build_response_summary():
        rows = []
        for session_number, (label, session_dir) in enumerate(SESSION_PATHS.items(), start=1):
            session = SESSION_ROWS[label]
            for condition in CONDITION_LABELS:
                table = pq.read_table(
                    Path(session_dir) / f"{condition}.parquet",
                    columns=["unit_id", "time_s", "sdf_hz"],
                )
                payload = table.to_pydict()
                unit_ids = np.asarray(payload["unit_id"], dtype=np.int64)
                values = np.asarray(payload["sdf_hz"], dtype=np.float64)
                unique_ids = pd.unique(unit_ids)
                samples_per_unit = len(values) // len(unique_ids)
                if len(values) != len(unique_ids) * samples_per_unit:
                    raise ValueError(f"Unequal SDF lengths in {session_dir} / {condition}")
                id_matrix = unit_ids.reshape(len(unique_ids), samples_per_unit)
                if not np.all(id_matrix == id_matrix[:, :1]):
                    raise ValueError(f"SDF rows are not grouped by unit in {session_dir} / {condition}")
                time_matrix = np.asarray(payload["time_s"], dtype=np.float64).reshape(
                    len(unique_ids), samples_per_unit
                )
                time_s = time_matrix[0]
                traces = values.reshape(len(unique_ids), samples_per_unit)
                baseline_mask = (
                    (time_s >= SUMMARY_BASELINE_WINDOW[0])
                    & (time_s < SUMMARY_BASELINE_WINDOW[1])
                )
                response_mask = (
                    (time_s >= SUMMARY_RESPONSE_WINDOW[0])
                    & (time_s < SUMMARY_RESPONSE_WINDOW[1])
                )
                delta_hz = (
                    traces[:, response_mask].mean(axis=1)
                    - traces[:, baseline_mask].mean(axis=1)
                )
                rows.append(pd.DataFrame({
                    "subject": str(session["subject"]),
                    "session_id": str(session["session_id"]),
                    "session_date": str(session["session_date"]),
                    "condition": condition,
                    "unit_id": id_matrix[:, 0],
                    "baseline_corrected_hz": delta_hz,
                }))
            if session_number % 4 == 0 or session_number == len(SESSION_PATHS):
                print(f"Summarized {session_number}/{len(SESSION_PATHS)} sensorimotor sessions")
        summary = pd.concat(rows, ignore_index=True)
        summary.to_parquet(RESPONSE_SUMMARY_PATH, index=False)
        return summary


    if RESPONSE_SUMMARY_PATH.exists():
        response_summary = pd.read_parquet(RESPONSE_SUMMARY_PATH)
    else:
        response_summary = build_response_summary()

    if not UNIT_METADATA_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing unit metadata cache: {UNIT_METADATA_CACHE_PATH}")
    if not OPTO_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing optotagging results: {OPTO_RESULTS_PATH}")

    all_unit_metadata = pd.read_parquet(UNIT_METADATA_CACHE_PATH).rename(
        columns={"unit_id": "unit_uuid"}
    )
    all_unit_metadata["unit_id"] = all_unit_metadata.groupby(
        "session_id", sort=False
    ).cumcount()
    all_unit_metadata["structure_acronym_detailed"] = all_unit_metadata[
        "structure_acronym"
    ]
    _vis_mask = all_unit_metadata["structure_acronym"].str.startswith("VIS", na=False)
    all_unit_metadata.loc[_vis_mask, "parent_area"] = "Isocortex"
    all_unit_metadata.loc[
        all_unit_metadata["structure_acronym"] == "VL", "parent_area"
    ] = "TH"
    _cortex_mask = all_unit_metadata["parent_area"] == "Isocortex"
    all_unit_metadata.loc[_cortex_mask, "structure_acronym"] = (
        all_unit_metadata.loc[_cortex_mask, "structure_acronym"]
        .str.replace(r"(?:2/3|6a|6b|1|4|5|6)$", "", regex=True)
    )
    all_unit_metadata["neuron_type"] = np.where(
        all_unit_metadata["peak_to_valley_ms"] > 0.4, "RS", "FS"
    )
    _th_mask = all_unit_metadata["parent_area"] == "TH"
    all_unit_metadata.loc[_th_mask, "neuron_type"] = np.where(
        all_unit_metadata.loc[_th_mask, "peak_to_valley_ms"] <= 0.28,
        "FS",
        "RS",
    )
    all_unit_metadata.loc[
        all_unit_metadata["parent_area"] == "STR", "neuron_type"
    ] = "RS"

    opto_results = pd.read_parquet(OPTO_RESULTS_PATH)
    _opto_sessions = (
        opto_results[["asset_path", "session_id"]]
        .drop_duplicates()
        .assign(
            subject=lambda frame: frame["asset_path"].str.extract(
                r"sub-(\d+)", expand=False
            ),
            session_date=lambda frame: pd.to_datetime(
                frame["session_id"].str.extract(
                    r"(\d{4}-\d{2}-\d{2})", expand=False
                )
            ),
        )
        .sort_values(["subject", "session_date", "session_id"])
    )



    def build_session_age_metadata():
        target_sessions = _opto_sessions.loc[
            _opto_sessions["session_id"].isin(response_summary["session_id"]),
            ["asset_path", "session_id"],
        ].drop_duplicates()
        target_paths = set(target_sessions["asset_path"])
        rows = []
        with DandiAPIClient() as client:
            dandiset = client.get_dandiset(
                DANDISET_ID, version_id=DANDISET_VERSION
            )
            assets = {
                asset.path: asset
                for asset in dandiset.get_assets()
                if asset.path in target_paths
            }
            for row in target_sessions.itertuples(index=False):
                metadata = assets[row.asset_path].get_metadata()
                subject = metadata.wasAttributedTo[0]
                age_iso = getattr(getattr(subject, "age", None), "value", None)
                if age_iso is None:
                    raise ValueError(f"Missing age for {row.session_id}")
                sex = getattr(getattr(subject, "sex", None), "name", None)
                if sex is None:
                    raise ValueError(f"Missing sex for {row.session_id}")
                rows.append({
                    "session_id": row.session_id,
                    "age_iso": age_iso,
                    "age_days": pd.Timedelta(age_iso).total_seconds() / 86400,
                    "sex": sex,
                })
        ages = pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)
        ages.to_parquet(SESSION_AGE_PATH, index=False)
        return ages


    if SESSION_AGE_PATH.exists():
        session_age_df = pd.read_parquet(SESSION_AGE_PATH)
        if "sex" not in session_age_df.columns:
            session_age_df = build_session_age_metadata()
    else:
        session_age_df = build_session_age_metadata()

    _cohort_by_session = {}
    for _subject, _subject_sessions in _opto_sessions.groupby("subject"):
        _ordered_ids = _subject_sessions["session_id"].tolist()
        for _sensorimotor_id in set(response_summary["session_id"]) & set(_ordered_ids):
            if _sensorimotor_id == _ordered_ids[0]:
                _cohort_by_session[_sensorimotor_id] = "Motor cohort"
            elif _sensorimotor_id == _ordered_ids[-1]:
                _cohort_by_session[_sensorimotor_id] = "Sequence cohort"
            else:
                raise ValueError(
                    f"Sensorimotor session is neither first nor last for {_subject}"
                )
    _sst_positive = (
        opto_results["5 hz pulse train_presentations__p_value"] < 0.05
    ) & (
        opto_results["5 hz pulse train_presentations__modulation_index"] > 0.1
    )
    _sst_keys = pd.MultiIndex.from_frame(
        opto_results.loc[_sst_positive, ["session_id", "unit_id"]].astype(str)
    )
    _unit_keys = pd.MultiIndex.from_frame(
        all_unit_metadata[["session_id", "unit_uuid"]].astype(str)
    )
    all_unit_metadata.loc[_unit_keys.isin(_sst_keys), "neuron_type"] = "SST"

    annotated_units = all_unit_metadata.loc[
        all_unit_metadata["session_id"].isin(sessions_df["session_id"]),
        [
            "session_id", "unit_id", "structure_acronym",
            "parent_area", "neuron_type",
        ],
    ].copy()
    annotated_units["parent_area"] = annotated_units["parent_area"].replace(
        "root", "Unresolved"
    )
    summary_df = response_summary.merge(
        annotated_units,
        on=["session_id", "unit_id"],
        how="inner",
        validate="many_to_one",
    )
    summary_df["training_cohort"] = summary_df["session_id"].map(
        _cohort_by_session
    )
    summary_df = summary_df.merge(
        session_age_df,
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    if summary_df["training_cohort"].isna().any():
        raise ValueError("Missing training cohort assignments")
    if summary_df["age_days"].isna().any():
        raise ValueError("Missing session ages")
    if summary_df["sex"].isna().any():
        raise ValueError("Missing session sex metadata")
    TRAINING_COHORTS = ["Motor cohort", "Sequence cohort"]
    SDF_PARENT_AREAS = sorted(
        area for area in summary_df["parent_area"].dropna().unique()
        if area != "Unresolved"
    ) + (["Unresolved"] if "Unresolved" in set(summary_df["parent_area"]) else [])
    SDF_STRUCTURE_ACRONYMS = sorted(
        summary_df["structure_acronym"].dropna().unique().tolist()
    )
    SDF_NEURON_TYPES = [
        neuron_type
        for neuron_type in ["RS", "FS", "SST"]
        if neuron_type in set(summary_df["neuron_type"])
    ]
    summary_df
    return (
        MIN_CELLS_PER_SESSION,
        MIN_SESSIONS_PER_AREA,
        SDF_NEURON_TYPES,
        SDF_PARENT_AREAS,
        SDF_STRUCTURE_ACRONYMS,
        annotated_units,
        summary_df,
    )


@app.cell(hide_code=True)
def _(SDF_NEURON_TYPES, SDF_PARENT_AREAS, SDF_STRUCTURE_ACRONYMS, mo):
    summary_group_picker = mo.ui.radio(
        options=["All together", "Parent area", "Structure acronym", "Neuron type"],
        value="All together",
        label="Stratify by",
        inline=True,
    )
    parent_area_picker = mo.ui.dropdown(
        options=["All areas", *SDF_PARENT_AREAS],
        value="All areas",
        label="Parent area",
    )
    structure_picker = mo.ui.dropdown(
        options=["All structures", *SDF_STRUCTURE_ACRONYMS],
        value="All structures",
        label="Structure acronym",
        searchable=True,
    )
    neuron_type_picker = mo.ui.dropdown(
        options=["All neuron types", *SDF_NEURON_TYPES],
        value="All neuron types",
        label="Neuron type",
    )
    cohort_toggle = mo.ui.switch(
        value=False,
        label="Split by training cohort (filled Motor / open Sequence diamonds)",
    )
    mo.vstack([
        mo.md(
            """
            ## Cross-session response summary

            Mean baseline-corrected firing-rate change during each 343 ms mismatch.
            Use the selectors to pool all units or stratify by anatomy or cell class.
            Anatomical strata require at least 5 units per session in at least 3 sessions.
            """
        ),
        summary_group_picker,
        cohort_toggle,
        mo.hstack([parent_area_picker, structure_picker, neuron_type_picker]),
    ], gap=0.8)
    return (
        cohort_toggle,
        neuron_type_picker,
        parent_area_picker,
        structure_picker,
        summary_group_picker,
    )


@app.cell(hide_code=True)
def summary(
    CONDITION_COLORS,
    CONDITION_LABELS,
    MIN_CELLS_PER_SESSION,
    MIN_SESSIONS_PER_AREA,
    cohort_toggle,
    neuron_type_picker,
    np,
    parent_area_picker,
    plt,
    structure_picker,
    summary_df,
    summary_group_picker,
):
    import seaborn as sns
    from scipy.stats import ttest_ind as _ttest_annot

    def _sig_stars(_p):
        if _p < 0.001:
            return "***"
        if _p < 0.01:
            return "**"
        if _p < 0.05:
            return "*"
        return "ns"

    def _cohort_pvalues(_frame):
        _out = {}
        for (_st, _lab), _g in _frame.groupby(
            ["stratum", "condition_label"], observed=True
        ):
            _m = _g.loc[_g["training_cohort"] == "Motor cohort", "session_mean_hz"].to_numpy()
            _s = _g.loc[_g["training_cohort"] == "Sequence cohort", "session_mean_hz"].to_numpy()
            if len(_m) >= 2 and len(_s) >= 2:
                _out[(_st, _lab)] = float(_ttest_annot(_m, _s, equal_var=False).pvalue)
        return _out

    _summary = summary_df.copy()
    if parent_area_picker.value != "All areas":
        _summary = _summary.loc[
            _summary["parent_area"] == parent_area_picker.value
        ]
    if structure_picker.value != "All structures":
        _summary = _summary.loc[
            _summary["structure_acronym"] == structure_picker.value
        ]
    if neuron_type_picker.value != "All neuron types":
        _summary = _summary.loc[
            _summary["neuron_type"] == neuron_type_picker.value
        ]

    _group_mode = summary_group_picker.value
    if _group_mode == "Parent area":
        _summary["stratum"] = _summary["parent_area"]
    elif _group_mode == "Structure acronym":
        _summary["stratum"] = _summary["structure_acronym"]
    elif _group_mode == "Neuron type":
        _summary["stratum"] = _summary["neuron_type"]
    else:
        _summary["stratum"] = "All selected units"

    _anatomy_column = None
    if _group_mode == "Parent area":
        _anatomy_column = "parent_area"
    elif _group_mode == "Structure acronym":
        _anatomy_column = "structure_acronym"

    if _anatomy_column is not None:
        _cohort_columns = ["training_cohort"] if cohort_toggle.value else []
        _count_columns = ["session_id", _anatomy_column, *_cohort_columns]
        _session_area_counts = (
            _summary[["session_id", "unit_id", _anatomy_column, *_cohort_columns]]
            .drop_duplicates()
            .groupby(_count_columns, observed=True)
            .size()
            .rename("cell_count")
            .reset_index()
        )
        _qualifying_pairs = _session_area_counts.loc[
            _session_area_counts["cell_count"] >= MIN_CELLS_PER_SESSION
        ].copy()
        _area_group_columns = [_anatomy_column, *_cohort_columns]
        _qualifying_groups = (
            _qualifying_pairs.groupby(_area_group_columns, observed=True)["session_id"]
            .nunique()
            .rename("qualifying_sessions")
            .reset_index()
            .loc[lambda frame: frame["qualifying_sessions"] >= MIN_SESSIONS_PER_AREA]
        )
        _qualifying_pairs = _qualifying_pairs.merge(
            _qualifying_groups[_area_group_columns],
            on=_area_group_columns,
            how="inner",
            validate="many_to_one",
        )
        _summary = _summary.merge(
            _qualifying_pairs[_count_columns],
            on=_count_columns,
            how="inner",
            validate="many_to_one",
        )

    _summary_figure, _summary_axis = plt.subplots(
        figsize=(12, 6.2), layout="constrained"
    )
    if _summary.empty:
        _summary_axis.text(
            0.5, 0.5, "No units match these filters",
            ha="center", va="center", transform=_summary_axis.transAxes,
        )
        _summary_axis.set_axis_off()
    else:
        _session_means = (
            _summary.groupby(
                [
                    "session_id", "training_cohort", "condition", "stratum"
                ],
                observed=True,
            )["baseline_corrected_hz"]
            .mean()
            .rename("session_mean_hz")
            .reset_index()
        )
        _session_means["condition_label"] = _session_means["condition"].map(
            CONDITION_LABELS
        )
        _condition_order = list(CONDITION_LABELS.values())
        _condition_palette = {
            CONDITION_LABELS[key]: color
            for key, color in CONDITION_COLORS.items()
        }
        _strata = sorted(_session_means["stratum"].unique().tolist())
        _unit_count = len(
            _summary[["session_id", "unit_id"]].drop_duplicates()
        )
        _session_count = _summary["session_id"].nunique()
        _cohort_split = cohort_toggle.value
        _structure_forest = (
            _group_mode == "Structure acronym" and len(_strata) > 1
        )
        _Line2D = __import__(
            "matplotlib.lines", fromlist=["Line2D"]
        ).Line2D

        if _structure_forest:
            _structure_order = (
                _summary.groupby("stratum", observed=True)
                .agg(
                    parent_area=("parent_area", "first"),
                    units=("unit_id", "size"),
                )
                .reset_index()
                .sort_values(
                    ["parent_area", "units", "stratum"],
                    ascending=[True, False, True],
                )["stratum"]
                .tolist()
            )
            _summary_figure.set_size_inches(
                12, max(6.2, 0.30 * len(_structure_order))
            )
            if _cohort_split:
                _cond_off_dots = dict(zip(
                    _condition_order,
                    np.linspace(-0.22, 0.22, len(_condition_order)),
                    strict=True,
                ))
                _coh_off_dots = {"Motor cohort": -0.045, "Sequence cohort": 0.045}
                _ypos_dots = {s: i for i, s in enumerate(_structure_order)}
                _rng_dots = np.random.default_rng(0)
                for _d in _session_means.itertuples(index=False):
                    if _d.stratum not in _ypos_dots:
                        continue
                    _dy = (
                        _ypos_dots[_d.stratum]
                        + _cond_off_dots[_d.condition_label]
                        + _coh_off_dots[_d.training_cohort]
                        + _rng_dots.uniform(-0.018, 0.018)
                    )
                    _dc = _condition_palette[_d.condition_label]
                    _summary_axis.scatter(
                        _d.session_mean_hz, _dy, s=15,
                        color=_dc,
                        facecolor=_dc if _d.training_cohort == "Motor cohort" else "white",
                        edgecolor=_dc, linewidth=0.6, alpha=0.5, zorder=3,
                    )
            else:
                sns.stripplot(
                    data=_session_means,
                    x="session_mean_hz",
                    y="stratum",
                    hue="condition_label",
                    order=_structure_order,
                    hue_order=_condition_order,
                    dodge=True,
                    jitter=0.16,
                    size=3.2,
                    alpha=0.28,
                    linewidth=0,
                    palette=_condition_palette,
                    legend=False,
                    ax=_summary_axis,
                )
            if _cohort_split:
                _cohort_stats = (
                    _session_means.groupby(
                        ["condition_label", "stratum", "training_cohort"],
                        observed=True,
                    )["session_mean_hz"]
                    .agg(mean="mean", std="std", sessions="count")
                    .reset_index()
                )
                _cohort_stats["sem"] = (
                    _cohort_stats["std"]
                    / np.sqrt(_cohort_stats["sessions"])
                ).fillna(0)
                _y_positions = {
                    structure: index
                    for index, structure in enumerate(_structure_order)
                }
                _condition_offsets = dict(
                    zip(
                        _condition_order,
                        np.linspace(-0.22, 0.22, len(_condition_order)),
                        strict=True,
                    )
                )
                _cohort_offsets = {
                    "Motor cohort": -0.045,
                    "Sequence cohort": 0.045,
                }
                for _row in _cohort_stats.itertuples(index=False):
                    _color = _condition_palette[_row.condition_label]
                    _filled = _row.training_cohort == "Motor cohort"
                    _summary_axis.errorbar(
                        _row.mean,
                        _y_positions[_row.stratum]
                        + _condition_offsets[_row.condition_label]
                        + _cohort_offsets[_row.training_cohort],
                        xerr=_row.sem,
                        fmt="D",
                        ms=5.2,
                        capsize=2.5,
                        color=_color,
                        markeredgecolor=_color,
                        markerfacecolor=_color if _filled else "white",
                        markeredgewidth=1.3,
                        zorder=5,
                    )
                _pvals = _cohort_pvalues(_session_means)
                for (_st, _lab), _p in _pvals.items():
                    _sub = _cohort_stats[
                        (_cohort_stats["stratum"] == _st)
                        & (_cohort_stats["condition_label"] == _lab)
                    ]
                    if _sub.empty:
                        continue
                    _xmax = float((_sub["mean"] + _sub["sem"]).max())
                    _ycen = (
                        _y_positions[_st]
                        + _condition_offsets[_lab]
                    )
                    _summary_axis.annotate(
                        f"{_sig_stars(_p)} p={_p:.3f}",
                        (_xmax, _ycen),
                        xytext=(6, 0),
                        textcoords="offset points",
                        va="center", ha="left", fontsize=7.5,
                        color="#111827", zorder=6,
                    )
                _condition_handles = [
                    _Line2D(
                        [], [], marker="o", linestyle="none",
                        color=_condition_palette[label], label=label,
                    )
                    for label in _condition_order
                ]
                _cohort_handles = [
                    _Line2D(
                        [], [], marker="D", linestyle="none", color="#374151",
                        markerfacecolor="#374151", label="Motor cohort",
                    ),
                    _Line2D(
                        [], [], marker="D", linestyle="none", color="#374151",
                        markerfacecolor="white", label="Sequence cohort",
                    ),
                ]
                _summary_axis.legend(
                    handles=[*_condition_handles, *_cohort_handles],
                    title="Color / diamond fill",
                    frameon=False,
                    loc="upper right",
                )
            else:
                sns.pointplot(
                    data=_session_means,
                    x="session_mean_hz",
                    y="stratum",
                    hue="condition_label",
                    order=_structure_order,
                    hue_order=_condition_order,
                    estimator="mean",
                    errorbar="se",
                    dodge=0.55,
                    markers="D",
                    linestyles="none",
                    capsize=0.16,
                    palette=_condition_palette,
                    legend=True,
                    ax=_summary_axis,
                )
                _summary_axis.legend(
                    title="Mismatch type", frameon=False, loc="upper right"
                )
            _summary_axis.axvline(0, color="#374151", lw=1, alpha=0.75)
            _summary_axis.set(
                xlabel="Mean baseline-corrected SDF during mismatch (Hz)",
                ylabel="Structure acronym",
                title=(
                    "Cross-session mismatch response by structure | "
                    f"{_unit_count:,} units across {_session_count} sessions"
                ),
            )
        else:
            _multi_stratum = len(_strata) > 1
            if _multi_stratum:
                _hue = "stratum"
                _hue_order = _strata
                _palette = {
                    stratum: plt.get_cmap("tab10")(index % 10)
                    for index, stratum in enumerate(_strata)
                }
            else:
                _hue = "condition_label"
                _hue_order = _condition_order
                _palette = _condition_palette

            if _cohort_split:
                _xpos_dots = {c: i for i, c in enumerate(_condition_order)}
                _strat_off_dots = dict(zip(
                    _strata,
                    np.linspace(-0.24, 0.24, len(_strata))
                    if _multi_stratum else [0.0],
                    strict=True,
                ))
                _coh_off_dots = {"Motor cohort": -0.045, "Sequence cohort": 0.045}
                _rng_dots = np.random.default_rng(0)
                for _d in _session_means.itertuples(index=False):
                    _dx = (
                        _xpos_dots[_d.condition_label]
                        + _strat_off_dots[_d.stratum]
                        + _coh_off_dots[_d.training_cohort]
                        + _rng_dots.uniform(-0.02, 0.02)
                    )
                    _dc = (
                        _palette[_d.stratum]
                        if _multi_stratum
                        else _condition_palette[_d.condition_label]
                    )
                    _summary_axis.scatter(
                        _dx, _d.session_mean_hz, s=22,
                        color=_dc,
                        facecolor=_dc if _d.training_cohort == "Motor cohort" else "white",
                        edgecolor=_dc, linewidth=0.7, alpha=0.55, zorder=3,
                    )
            else:
                sns.stripplot(
                    data=_session_means,
                    x="condition_label",
                    y="session_mean_hz",
                    hue=_hue,
                    order=_condition_order,
                    hue_order=_hue_order,
                    dodge=_multi_stratum,
                    jitter=0.16,
                    size=4.5,
                    alpha=0.30,
                    linewidth=0,
                    palette=_palette,
                    legend=False,
                    ax=_summary_axis,
                )
            if _cohort_split:
                _cohort_stats = (
                    _session_means.groupby(
                        ["condition_label", "stratum", "training_cohort"],
                        observed=True,
                    )["session_mean_hz"]
                    .agg(mean="mean", std="std", sessions="count")
                    .reset_index()
                )
                _cohort_stats["sem"] = (
                    _cohort_stats["std"]
                    / np.sqrt(_cohort_stats["sessions"])
                ).fillna(0)
                _x_positions = {
                    condition: index
                    for index, condition in enumerate(_condition_order)
                }
                _stratum_offsets = dict(
                    zip(
                        _strata,
                        np.linspace(-0.24, 0.24, len(_strata))
                        if _multi_stratum
                        else [0.0],
                        strict=True,
                    )
                )
                _cohort_offsets = {
                    "Motor cohort": -0.045,
                    "Sequence cohort": 0.045,
                }
                for _row in _cohort_stats.itertuples(index=False):
                    _color = (
                        _palette[_row.stratum]
                        if _multi_stratum
                        else _condition_palette[_row.condition_label]
                    )
                    _filled = _row.training_cohort == "Motor cohort"
                    _summary_axis.errorbar(
                        _x_positions[_row.condition_label]
                        + _stratum_offsets[_row.stratum]
                        + _cohort_offsets[_row.training_cohort],
                        _row.mean,
                        yerr=_row.sem,
                        fmt="D",
                        ms=6,
                        capsize=3,
                        color=_color,
                        markeredgecolor=_color,
                        markerfacecolor=_color if _filled else "white",
                        markeredgewidth=1.4,
                        zorder=5,
                    )
                _pvals = _cohort_pvalues(_session_means)
                for (_st, _lab), _p in _pvals.items():
                    _sub = _cohort_stats[
                        (_cohort_stats["stratum"] == _st)
                        & (_cohort_stats["condition_label"] == _lab)
                    ]
                    if _sub.empty:
                        continue
                    _ymax = float((_sub["mean"] + _sub["sem"]).max())
                    _xcen = (
                        _x_positions[_lab]
                        + _stratum_offsets[_st]
                    )
                    _summary_axis.annotate(
                        f"{_sig_stars(_p)}\np={_p:.3f}",
                        (_xcen, _ymax),
                        xytext=(0, 9),
                        textcoords="offset points",
                        va="bottom", ha="center", fontsize=7.5,
                        color="#111827", zorder=6,
                    )
                _color_handles = (
                    [
                        _Line2D(
                            [], [], marker="o", linestyle="none",
                            color=_palette[stratum], label=stratum,
                        )
                        for stratum in _strata
                    ]
                    if _multi_stratum
                    else []
                )
                _cohort_handles = [
                    _Line2D(
                        [], [], marker="D", linestyle="none", color="#374151",
                        markerfacecolor="#374151", label="Motor cohort",
                    ),
                    _Line2D(
                        [], [], marker="D", linestyle="none", color="#374151",
                        markerfacecolor="white", label="Sequence cohort",
                    ),
                ]
                _summary_axis.legend(
                    handles=[*_color_handles, *_cohort_handles],
                    title=(
                        f"{_group_mode} / cohort"
                        if _multi_stratum
                        else "Training cohort"
                    ),
                    frameon=False,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )
            else:
                sns.pointplot(
                    data=_session_means,
                    x="condition_label",
                    y="session_mean_hz",
                    hue=_hue,
                    order=_condition_order,
                    hue_order=_hue_order,
                    estimator="mean",
                    errorbar="se",
                    dodge=0.55 if _multi_stratum else False,
                    markers="D",
                    linestyles="none",
                    capsize=0.14,
                    palette=_palette,
                    legend=_multi_stratum,
                    ax=_summary_axis,
                )
                if _multi_stratum:
                    _summary_axis.legend(
                        title=_group_mode,
                        frameon=False,
                        bbox_to_anchor=(1.02, 1),
                        loc="upper left",
                    )
            _summary_axis.axhline(0, color="#374151", lw=1, alpha=0.75)
            _summary_axis.set(
                ylabel="Mean baseline-corrected SDF during mismatch (Hz)",
                xlabel="Mismatch type",
                title=(
                    "Cross-session mismatch response | "
                    f"{_unit_count:,} units across {_session_count} sessions"
                ),
            )

        _summary_axis.spines[["top", "right"]].set_visible(False)
        _summary_axis.text(
            0.01,
            1.01,
            (
                "Dots: session means; diamonds: cohort mean +/- SEM "
                "(filled Motor, open Sequence) | "
                if _cohort_split
                else "Dots: session means; diamonds: across-session mean +/- SEM | "
            )
            + "Response: 0-343 ms minus mean -1-0 s baseline",
            transform=_summary_axis.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#4b5563",
        )

    _summary_figure
    return (sns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Animal age and mismatch-response amplitude

    Each point is one animal's sensorimotor session. Response amplitude uses the
    same baseline-corrected 0-343 ms mean as the cross-session summary and follows
    its parent-area, structure, and neuron-type filters. Lines show the raw linear
    association. The table reports Pearson, Spearman, and a cohort-adjusted Pearson
    correlation obtained after removing Motor-versus-Sequence cohort means from
    both age and response.
    """)
    return


@app.cell(hide_code=True)
def _(
    CONDITION_COLORS,
    CONDITION_LABELS,
    MIN_CELLS_PER_SESSION,
    MIN_SESSIONS_PER_AREA,
    mo,
    neuron_type_picker,
    np,
    parent_area_picker,
    pd,
    plt,
    structure_picker,
    summary_df,
):
    from scipy.stats import pearsonr, spearmanr, t as student_t

    _age_summary = summary_df.copy()
    if parent_area_picker.value != "All areas":
        _age_summary = _age_summary.loc[
            _age_summary["parent_area"] == parent_area_picker.value
        ]
    if structure_picker.value != "All structures":
        _age_summary = _age_summary.loc[
            _age_summary["structure_acronym"] == structure_picker.value
        ]
    if neuron_type_picker.value != "All neuron types":
        _age_summary = _age_summary.loc[
            _age_summary["neuron_type"] == neuron_type_picker.value
        ]

    _age_unit_counts = (
        _age_summary[["session_id", "unit_id"]]
        .drop_duplicates()
        .groupby("session_id")
        .size()
    )
    _age_valid_sessions = _age_unit_counts.loc[
        _age_unit_counts >= MIN_CELLS_PER_SESSION
    ].index
    _age_summary = _age_summary.loc[
        _age_summary["session_id"].isin(_age_valid_sessions)
    ]
    _age_session_means = (
        _age_summary.groupby(
            [
                "session_id", "subject", "age_days",
                "training_cohort", "condition",
            ],
            observed=True,
        )["baseline_corrected_hz"]
        .mean()
        .rename("response_hz")
        .reset_index()
    )

    _age_figure, _age_axes = plt.subplots(
        1, 3, figsize=(16, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    _age_rows = []
    for _age_axis, (_condition, _condition_label) in zip(
        _age_axes, CONDITION_LABELS.items(), strict=True
    ):
        _condition_data = _age_session_means.loc[
            _age_session_means["condition"] == _condition
        ].copy()
        _x = _condition_data["age_days"].to_numpy(float)
        _y = _condition_data["response_hz"].to_numpy(float)
        _n = len(_condition_data)

        if _n >= MIN_SESSIONS_PER_AREA and np.ptp(_x) > 0 and np.ptp(_y) > 0:
            _pearson = pearsonr(_x, _y)
            _spearman = spearmanr(_x, _y)
            _cohort_indicator = (
                _condition_data["training_cohort"] == "Sequence cohort"
            ).to_numpy(float)
            _design = np.column_stack([np.ones(_n), _cohort_indicator])
            _age_residual = _x - _design @ np.linalg.lstsq(
                _design, _x, rcond=None
            )[0]
            _response_residual = _y - _design @ np.linalg.lstsq(
                _design, _y, rcond=None
            )[0]
            if np.ptp(_age_residual) > 0 and np.ptp(_response_residual) > 0:
                _adjusted_r = float(pearsonr(
                    _age_residual, _response_residual
                ).statistic)
                _adjusted_df = _n - 3
                _adjusted_t = _adjusted_r * np.sqrt(
                    _adjusted_df / max(1 - _adjusted_r ** 2, 1e-12)
                )
                _adjusted_p = float(
                    2 * student_t.sf(abs(_adjusted_t), df=_adjusted_df)
                )
            else:
                _adjusted_r = np.nan
                _adjusted_p = np.nan
            _slope, _intercept = np.polyfit(_x, _y, 1)
            _line_x = np.linspace(_x.min(), _x.max(), 100)
            _age_axis.plot(
                _line_x,
                _intercept + _slope * _line_x,
                color=CONDITION_COLORS[_condition],
                lw=2,
                alpha=0.75,
            )
            _age_rows.append({
                "Mismatch": _condition_label,
                "Sessions": _n,
                "Pearson r": float(_pearson.statistic),
                "Pearson p": float(_pearson.pvalue),
                "Spearman rho": float(_spearman.statistic),
                "Spearman p": float(_spearman.pvalue),
                "Cohort-adjusted r": _adjusted_r,
                "Adjusted p": _adjusted_p,
            })
            _stat_text = (
                f"r={_pearson.statistic:.2f}, p={_pearson.pvalue:.3f}\n"
                f"adjusted r={_adjusted_r:.2f}, p={_adjusted_p:.3f}"
            )
        else:
            _age_rows.append({
                "Mismatch": _condition_label,
                "Sessions": _n,
                "Pearson r": np.nan,
                "Pearson p": np.nan,
                "Spearman rho": np.nan,
                "Spearman p": np.nan,
                "Cohort-adjusted r": np.nan,
                "Adjusted p": np.nan,
            })
            _stat_text = f"n={_n}; insufficient variation"

        for _cohort, _filled in [
            ("Motor cohort", True),
            ("Sequence cohort", False),
        ]:
            _cohort_data = _condition_data.loc[
                _condition_data["training_cohort"] == _cohort
            ]
            _age_axis.scatter(
                _cohort_data["age_days"],
                _cohort_data["response_hz"],
                s=48,
                marker="D",
                facecolor=(
                    CONDITION_COLORS[_condition] if _filled else "white"
                ),
                edgecolor=CONDITION_COLORS[_condition],
                linewidth=1.4,
                alpha=0.9,
                label=_cohort,
                zorder=4,
            )
        _age_axis.axhline(0, color="#6b7280", lw=0.8, alpha=0.6)
        _age_axis.set_title(_condition_label)
        _age_axis.set_xlabel("Age at recording (postnatal days)")
        _age_axis.text(
            0.03, 0.97, _stat_text,
            transform=_age_axis.transAxes,
            ha="left", va="top", fontsize=9,
        )
        _age_axis.spines[["top", "right"]].set_visible(False)

    _age_axes[0].set_ylabel("Mean baseline-corrected response (Hz)")
    _age_axes[-1].legend(title="Training cohort", frameon=False, loc="best")
    _age_figure.suptitle(
        "Animal age versus mismatch-response amplitude | current filters pooled",
        fontsize=14,
    )
    _age_stats = pd.DataFrame(_age_rows).round(3)
    mo.vstack([
        _age_figure,
        mo.ui.table(
            _age_stats,
            selection=None,
            pagination=False,
            show_column_summaries=False,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Female versus male mismatch responses

    Session-level responses follow the current anatomical and neuron-type filters.
    Dots are animals; diamonds show sex-specific mean +/- SEM. The table includes
    Welch tests and Female-minus-Male effects adjusted for age and training cohort.
    """)
    return


@app.cell(hide_code=True)
def _(
    CONDITION_COLORS,
    CONDITION_LABELS,
    MIN_CELLS_PER_SESSION,
    mo,
    neuron_type_picker,
    np,
    parent_area_picker,
    pd,
    plt,
    sns,
    structure_picker,
    summary_df,
):
    from scipy.stats import ttest_ind as _ttest_ind, t as _student_t_sex

    _sex_data = summary_df.copy()
    if parent_area_picker.value != "All areas":
        _sex_data = _sex_data[_sex_data["parent_area"] == parent_area_picker.value]
    if structure_picker.value != "All structures":
        _sex_data = _sex_data[_sex_data["structure_acronym"] == structure_picker.value]
    if neuron_type_picker.value != "All neuron types":
        _sex_data = _sex_data[_sex_data["neuron_type"] == neuron_type_picker.value]
    _counts = _sex_data[["session_id", "unit_id"]].drop_duplicates().groupby("session_id").size()
    _sex_data = _sex_data[_sex_data["session_id"].isin(_counts[_counts >= MIN_CELLS_PER_SESSION].index)]
    _session_sex = (
        _sex_data.groupby(
            ["session_id", "sex", "age_days", "training_cohort", "condition"],
            observed=True,
        )["baseline_corrected_hz"].mean().rename("response_hz").reset_index()
    )
    _sex_fig, _sex_axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, layout="constrained")
    _sex_results = []
    for _ax, (_condition, _label) in zip(_sex_axs, CONDITION_LABELS.items(), strict=True):
        _data = _session_sex[_session_sex["condition"] == _condition].copy()
        sns.stripplot(
            data=_data, x="sex", y="response_hz", order=["Male", "Female"],
            color=CONDITION_COLORS[_condition], jitter=0.14, size=5,
            alpha=0.38, linewidth=0, ax=_ax,
        )
        sns.pointplot(
            data=_data, x="sex", y="response_hz", order=["Male", "Female"],
            estimator="mean", errorbar="se", markers="D", linestyles="none",
            capsize=0.16, color=CONDITION_COLORS[_condition], ax=_ax,
        )
        _female = _data.loc[_data["sex"] == "Female", "response_hz"].to_numpy(float)
        _male = _data.loc[_data["sex"] == "Male", "response_hz"].to_numpy(float)
        _raw_effect = float(_female.mean() - _male.mean())
        _welch = _ttest_ind(_female, _male, equal_var=False)
        _female_code = (_data["sex"] == "Female").to_numpy(float)
        _cohort_code = (_data["training_cohort"] == "Sequence cohort").to_numpy(float)
        _age = _data["age_days"].to_numpy(float)
        _age = _age - _age.mean()
        _response = _data["response_hz"].to_numpy(float)
        _X = np.column_stack([np.ones(len(_data)), _female_code, _cohort_code, _age])
        _beta = np.linalg.lstsq(_X, _response, rcond=None)[0]
        _resid = _response - _X @ _beta
        _df = len(_response) - _X.shape[1]
        _sigma2 = np.sum(_resid ** 2) / _df
        _cov = _sigma2 * np.linalg.pinv(_X.T @ _X)
        _adjusted = float(_beta[1])
        _adjusted_se = float(np.sqrt(_cov[1, 1]))
        _adjusted_p = float(2 * _student_t_sex.sf(abs(_adjusted / _adjusted_se), df=_df))
        _sex_results.append({
            "Mismatch": _label, "Female n": len(_female), "Male n": len(_male),
            "Female - Male (Hz)": _raw_effect, "Welch p": float(_welch.pvalue),
            "Adjusted difference (Hz)": _adjusted, "Adjusted p": _adjusted_p,
        })
        _ax.axhline(0, color="#6b7280", lw=0.8, alpha=0.6)
        _ax.set(title=_label, xlabel="Sex", ylabel="Mean baseline-corrected response (Hz)" if _ax is _sex_axs[0] else "")
        _ax.text(
            0.03, 0.97,
            f"Female-Male={_raw_effect:.2f} Hz, p={_welch.pvalue:.3f}\n"
            f"adjusted={_adjusted:.2f} Hz, p={_adjusted_p:.3f}",
            transform=_ax.transAxes, ha="left", va="top", fontsize=9,
        )
        _ax.spines[["top", "right"]].set_visible(False)
    _sex_fig.suptitle("Female versus male mismatch-response amplitude | current filters pooled", fontsize=14)
    _sex_stats = pd.DataFrame(_sex_results).round(3)
    mo.vstack([
        _sex_fig,
        mo.ui.table(_sex_stats, selection=None, pagination=False, show_column_summaries=False),
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def cohort_statistics(
    CONDITION_LABELS,
    MIN_CELLS_PER_SESSION,
    cohort_toggle,
    mo,
    neuron_type_picker,
    np,
    parent_area_picker,
    pd,
    structure_picker,
    summary_df,
    summary_group_picker,
):
    from scipy.stats import ttest_ind as _ttest_cohort

    # Statistical testing: Motor vs Sequence cohort, mirroring the summary filters.
    # Renders only when "Split by training cohort" is enabled.
    if not cohort_toggle.value:
        cohort_stats_table = mo.md(
            "*Enable **Split by training cohort** above to run Motor-vs-Sequence tests.*"
        )
    else:
        _cs = summary_df.copy()
        if parent_area_picker.value != "All areas":
            _cs = _cs.loc[_cs["parent_area"] == parent_area_picker.value]
        if structure_picker.value != "All structures":
            _cs = _cs.loc[_cs["structure_acronym"] == structure_picker.value]
        if neuron_type_picker.value != "All neuron types":
            _cs = _cs.loc[_cs["neuron_type"] == neuron_type_picker.value]

        _mode = summary_group_picker.value
        if _mode == "Parent area":
            _cs["stratum"] = _cs["parent_area"]
        elif _mode == "Structure acronym":
            _cs["stratum"] = _cs["structure_acronym"]
        elif _mode == "Neuron type":
            _cs["stratum"] = _cs["neuron_type"]
        else:
            _cs["stratum"] = "All selected units"

        # Sessions need enough cells (same gate as the summary panel)
        _counts = (
            _cs[["session_id", "unit_id"]].drop_duplicates()
            .groupby("session_id").size()
        )
        _cs = _cs.loc[
            _cs["session_id"].isin(_counts[_counts >= MIN_CELLS_PER_SESSION].index)
        ]

        # One value per session = the unit of analysis for the test
        _session_means = (
            _cs.groupby(
                ["session_id", "training_cohort", "condition", "stratum"],
                observed=True,
            )["baseline_corrected_hz"]
            .mean()
            .rename("session_mean_hz")
            .reset_index()
        )

        _rows = []
        for (_stratum, _condition), _grp in _session_means.groupby(
            ["stratum", "condition"], observed=True
        ):
            _motor = _grp.loc[
                _grp["training_cohort"] == "Motor cohort", "session_mean_hz"
            ].to_numpy()
            _seq = _grp.loc[
                _grp["training_cohort"] == "Sequence cohort", "session_mean_hz"
            ].to_numpy()
            _row = {
                "stratum": _stratum,
                "condition": CONDITION_LABELS.get(_condition, _condition),
                "n_motor": len(_motor),
                "n_sequence": len(_seq),
                "mean_motor": np.mean(_motor) if len(_motor) else np.nan,
                "mean_sequence": np.mean(_seq) if len(_seq) else np.nan,
            }
            if len(_motor) >= 2 and len(_seq) >= 2:
                _res = _ttest_cohort(_motor, _seq, equal_var=False)  # Welch
                _row["diff_motor_minus_seq"] = np.mean(_motor) - np.mean(_seq)
                _row["t"] = _res.statistic
                _row["p_value"] = _res.pvalue
            else:
                _row["diff_motor_minus_seq"] = (
                    np.mean(_motor) - np.mean(_seq)
                    if len(_motor) and len(_seq) else np.nan
                )
                _row["t"] = np.nan
                _row["p_value"] = np.nan
            _rows.append(_row)

        cohort_stats_df = pd.DataFrame(_rows)
        cohort_stats_df["significant"] = cohort_stats_df["p_value"] < 0.05
        cohort_stats_df = cohort_stats_df.round(4)
        cohort_stats_table = mo.vstack([
            mo.md(
                "### Motor vs Sequence cohort test\n"
                "Welch two-sample t-test on **session-mean** baseline-corrected "
                "response (Hz), per mismatch condition"
                + (f" and {_mode.lower()}." if _mode != "All together" else ".")
            ),
            mo.ui.table(cohort_stats_df, selection=None),
        ])

    cohort_stats_table
    return


if __name__ == "__main__":
    app.run()
