import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def sdf_configuration(pd):
    import json
    from pathlib import Path
    import shutil

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import scipy.ndimage
    import scipy.signal

    SDF_OUTPUT_ROOT = Path(r"C:\Users\Roberto\Data\sensorimotor_mismatch_psths")
    SDF_WINDOW = (-1.0, 1.5)
    SDF_SAMPLING_INTERVAL = 0.001
    SDF_SIGMA = 0.005
    SDF_UNIT_CHUNK_SIZE = 32

    AMPLITUDE_CUTOFF_THRESHOLD = 0.1
    ISI_VIOLATIONS_RATIO_THRESHOLD = 0.5
    PRESENCE_RATIO_THRESHOLD = 0.9
    MINIMUM_FIRING_RATE = 1.0

    DEFAULT_QC = (
        f"(amplitude_cutoff < {AMPLITUDE_CUTOFF_THRESHOLD}) & "
        f"(isi_violations_ratio < {ISI_VIOLATIONS_RATIO_THRESHOLD}) & "
        f"(presence_ratio > {PRESENCE_RATIO_THRESHOLD})"
    )

    SENSORIMOTOR_CONDITIONS = {
        "motor_halt": ("motor_halt",),
        "motor_omission": ("motor_omission",),
        "motor_orientation": ("motor_orientation_45", "motor_orientation_90"),
    }


    def get_SDF(
        spikes,
        startTimes,
        unit_id,
        window=SDF_WINDOW,
        sampInt=SDF_SAMPLING_INTERVAL,
        sigma=SDF_SIGMA,
        avg=True,
    ):
        # Based on corbennett/ephys_behavior analysis_utils.py.
        windowDur = window[1] - window[0]
        t = np.arange(0, windowDur + sampInt, sampInt)

        counts = np.zeros((len(startTimes), len(t) - 1))
        for i, start in enumerate(startTimes):
            rel_spikes = spikes[
                (spikes >= start + window[0]) & (spikes <= start + window[1])
            ] - (start + window[0])
            counts[i] = np.histogram(rel_spikes, bins=t)[0]

        filtPts = int(5 * sigma / sampInt)
        expFilt = np.zeros(filtPts * 2)
        expFilt[-filtPts:] = scipy.signal.windows.exponential(
            filtPts,
            center=0,
            tau=sigma / sampInt,
            sym=False,
        )
        expFilt /= expFilt.sum()
        sdf = scipy.ndimage.convolve1d(counts, expFilt, axis=1)

        if avg:
            sdf = sdf.mean(axis=0)
        sdf /= sampInt
        t_abs = np.arange(window[0], window[1] + sampInt, sampInt)[:-1]

        return pd.Series(sdf, index=t_abs, name=unit_id)

    return (
        AMPLITUDE_CUTOFF_THRESHOLD,
        DEFAULT_QC,
        ISI_VIOLATIONS_RATIO_THRESHOLD,
        MINIMUM_FIRING_RATE,
        PRESENCE_RATIO_THRESHOLD,
        Path,
        SDF_OUTPUT_ROOT,
        SDF_SAMPLING_INTERVAL,
        SDF_SIGMA,
        SDF_UNIT_CHUNK_SIZE,
        SDF_WINDOW,
        SENSORIMOTOR_CONDITIONS,
        get_SDF,
        json,
        np,
        pa,
        pq,
        shutil,
    )


@app.cell
def _():
    import h5py
    import remfile
    from dandi.dandiapi import DandiAPIClient
    from pynwb import NWBHDF5IO

    DANDISET_ID = "001637"  # Use "001768" for mesoscope data.
    DANDISET_VERSION = "draft"

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID, version_id=DANDISET_VERSION)
        asset = next(
            asset for asset in dandiset.get_assets() if asset.path.endswith(".nwb")
        )
        download_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    remote_file = remfile.File(download_url)
    h5_file = h5py.File(remote_file, mode="r")
    with NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        table_name = next(iter(nwbfile.intervals))
        intervals = nwbfile.intervals[table_name].to_dataframe()

        print(f"Streaming: {asset.path}")
        print(f"Session: {nwbfile.session_id}")
        print(f"Intervals table: {table_name}")
        print(intervals.head())

    remote_file.close()
    return DANDISET_ID, DANDISET_VERSION, DandiAPIClient, nwbfile


@app.cell
def _(nwbfile):
    nwbfile
    return


@app.cell
def sdf_export_helpers(
    AMPLITUDE_CUTOFF_THRESHOLD,
    DEFAULT_QC,
    ISI_VIOLATIONS_RATIO_THRESHOLD,
    MINIMUM_FIRING_RATE,
    PRESENCE_RATIO_THRESHOLD,
    SDF_OUTPUT_ROOT,
    SDF_SAMPLING_INTERVAL,
    SDF_SIGMA,
    SDF_UNIT_CHUNK_SIZE,
    SDF_WINDOW,
    SENSORIMOTOR_CONDITIONS,
    get_SDF,
    json,
    np,
    pa,
    pd,
    pq,
    shutil,
):
    def decode_nwb_strings(values):
        return np.asarray(
            [value.decode() if isinstance(value, bytes) else str(value) for value in values]
        )


    def get_sensorimotor_event_times(interval_group):
        trial_types = decode_nwb_strings(interval_group["TrialType"][:])
        start_times = interval_group["start_time"][:]
        available = set(trial_types)
        expected = {
            label
            for labels in SENSORIMOTOR_CONDITIONS.values()
            for label in labels
        }
        missing = expected - available
        if missing:
            raise ValueError(f"Missing sensorimotor TrialType labels: {sorted(missing)}")

        return {
            condition: np.sort(start_times[np.isin(trial_types, labels)])
            for condition, labels in SENSORIMOTOR_CONDITIONS.items()
        }


    def get_qc_units(units_group):
        required = [
            "id",
            "firing_rate",
            "amplitude_cutoff",
            "isi_violations_ratio",
            "presence_ratio",
            "spike_times",
            "spike_times_index",
        ]
        missing = [name for name in required if name not in units_group]
        if missing:
            raise ValueError(f"Missing required unit datasets: {missing}")

        values = {
            name: units_group[name][:]
            for name in required
            if name not in {"spike_times", "spike_times_index"}
        }
        spike_ends = units_group["spike_times_index"][:].astype(np.int64)
        spike_starts = np.concatenate(([0], spike_ends[:-1]))
        keep = (
            (values["firing_rate"] > MINIMUM_FIRING_RATE)
            & (values["amplitude_cutoff"] < AMPLITUDE_CUTOFF_THRESHOLD)
            & (values["isi_violations_ratio"] < ISI_VIOLATIONS_RATIO_THRESHOLD)
            & (values["presence_ratio"] > PRESENCE_RATIO_THRESHOLD)
        )

        return pd.DataFrame(
            {
                "unit_row": np.flatnonzero(keep),
                "unit_id": values["id"][keep].astype(np.int64),
                "firing_rate": values["firing_rate"][keep],
                "amplitude_cutoff": values["amplitude_cutoff"][keep],
                "isi_violations_ratio": values["isi_violations_ratio"][keep],
                "presence_ratio": values["presence_ratio"][keep],
                "spike_start": spike_starts[keep],
                "spike_end": spike_ends[keep],
            }
        )


    def write_sdf_chunk(writer, session_row, condition, unit_ids, sdf_values, time_s):
        unit_ids = np.asarray(unit_ids, dtype=np.int64)
        sdf_values = np.asarray(sdf_values, dtype=np.float32)
        rows = len(unit_ids) * len(time_s)
        table = pa.table(
            {
                "subject": pa.array([str(session_row["subject"])] * rows),
                "session_id": pa.array([str(session_row["session_id"])] * rows),
                "session_date": pa.array([str(session_row["session_date"])] * rows),
                "unit_id": pa.array(np.repeat(unit_ids, len(time_s))),
                "condition": pa.array([condition] * rows),
                "time_s": pa.array(np.tile(time_s, len(unit_ids))),
                "sdf_hz": pa.array(sdf_values.reshape(-1)),
            }
        )
        writer.write_table(table)


    def export_sensorimotor_session(asset, session_row, output_root=SDF_OUTPUT_ROOT):
        import h5py as _h5py_export
        import remfile as _remfile_export

        session_slug = f"sub-{session_row['subject']}_ses-{session_row['session_date']}"
        output_root.mkdir(parents=True, exist_ok=True)
        final_dir = output_root / session_slug
        success_file = final_dir / "_SUCCESS"
        if success_file.exists():
            with (final_dir / "manifest.json").open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            return {"status": "already_complete", **manifest}
        if final_dir.exists():
            raise FileExistsError(f"Incomplete output directory exists: {final_dir}")

        temporary_dir = output_root / f".{session_slug}.tmp"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=True)

        remote_file = None
        h5_file = None
        writers = {}
        try:
            remote_file = _remfile_export.File(
                asset.get_content_url(follow_redirects=1, strip_query=True)
            )
            h5_file = _h5py_export.File(remote_file, mode="r")
            interval_path = "intervals/Sensory-motor mismatch block_presentations"
            if interval_path not in h5_file:
                raise ValueError(f"Missing interval table in {asset.path}: {interval_path}")

            event_times = get_sensorimotor_event_times(h5_file[interval_path])
            qc_units = get_qc_units(h5_file["units"])
            if qc_units.empty:
                raise ValueError(f"No units passed QC in {asset.path}")
            unit_metadata = qc_units.drop(columns=["spike_start", "spike_end"])
            pq.write_table(
                pa.table(
                    {
                        column: pa.array(unit_metadata[column].to_numpy())
                        for column in unit_metadata.columns
                    }
                ),
                temporary_dir / "units.parquet",
                compression="zstd",
            )

            schema = pa.schema(
                [
                    ("subject", pa.string()),
                    ("session_id", pa.string()),
                    ("session_date", pa.string()),
                    ("unit_id", pa.int64()),
                    ("condition", pa.string()),
                    ("time_s", pa.float64()),
                    ("sdf_hz", pa.float32()),
                ]
            )
            for condition in SENSORIMOTOR_CONDITIONS:
                writers[condition] = pq.ParquetWriter(
                    temporary_dir / f"{condition}.parquet",
                    schema,
                    compression="zstd",
                    use_dictionary=["subject", "session_id", "session_date", "condition"],
                )

            time_s = np.arange(
                SDF_WINDOW[0],
                SDF_WINDOW[1] + SDF_SAMPLING_INTERVAL,
                SDF_SAMPLING_INTERVAL,
            )[:-1]
            spike_times = h5_file["units/spike_times"]
            for chunk_start in range(0, len(qc_units), SDF_UNIT_CHUNK_SIZE):
                chunk = qc_units.iloc[chunk_start : chunk_start + SDF_UNIT_CHUNK_SIZE]
                chunk_sdfs = {condition: [] for condition in SENSORIMOTOR_CONDITIONS}
                for unit in chunk.itertuples(index=False):
                    spikes = spike_times[unit.spike_start : unit.spike_end]
                    for condition, starts in event_times.items():
                        sdf = get_SDF(spikes, starts, unit.unit_id)
                        chunk_sdfs[condition].append(sdf.to_numpy(dtype=np.float32))

                for condition, values in chunk_sdfs.items():
                    write_sdf_chunk(
                        writers[condition],
                        session_row,
                        condition,
                        chunk["unit_id"].to_numpy(),
                        values,
                        time_s,
                    )
                completed = min(chunk_start + len(chunk), len(qc_units))
                print(f"{session_slug}: {completed}/{len(qc_units)} QC units")

            for writer in writers.values():
                writer.close()
            writers.clear()
            h5_file.close()
            h5_file = None
            remote_file.close()
            remote_file = None

            manifest = {
                "subject": str(session_row["subject"]),
                "session_id": str(session_row["session_id"]),
                "session_date": str(session_row["session_date"]),
                "source_path": asset.path,
                "qc_query": f"(firing_rate > {MINIMUM_FIRING_RATE}) & {DEFAULT_QC}",
                "qc_unit_count": int(len(qc_units)),
                "trial_counts": {
                    condition: int(len(starts))
                    for condition, starts in event_times.items()
                },
                "sdf": {
                    "window_s": list(SDF_WINDOW),
                    "sampling_interval_s": SDF_SAMPLING_INTERVAL,
                    "sigma_s": SDF_SIGMA,
                    "filter": "causal_exponential",
                    "averaged_across_trials": True,
                    "units": "Hz",
                },
                "files": {
                    condition: f"{condition}.parquet"
                    for condition in SENSORIMOTOR_CONDITIONS
                },
            }
            with (temporary_dir / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
            (temporary_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
            temporary_dir.rename(final_dir)
            return {"status": "written", **manifest}
        except Exception:
            for writer in writers.values():
                writer.close()
            if h5_file is not None:
                h5_file.close()
            if remote_file is not None:
                remote_file.close()
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise


    def validate_session_export(session_dir):
        with (session_dir / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        expected_time = np.arange(
            SDF_WINDOW[0],
            SDF_WINDOW[1] + SDF_SAMPLING_INTERVAL,
            SDF_SAMPLING_INTERVAL,
        )[:-1]
        expected_units = manifest["qc_unit_count"]
        checks = {}
        for condition in SENSORIMOTOR_CONDITIONS:
            table = pq.read_table(
                session_dir / f"{condition}.parquet",
                columns=["unit_id", "condition", "time_s", "sdf_hz"],
            )
            unit_ids = table["unit_id"].to_numpy()
            conditions = set(table["condition"].to_pylist())
            times = table["time_s"].to_numpy()
            values = table["sdf_hz"].to_numpy()
            unique_units, counts = np.unique(unit_ids, return_counts=True)
            expected_rows = expected_units * len(expected_time)
            if table.num_rows != expected_rows:
                raise ValueError(
                    f"{condition}: expected {expected_rows} rows, found {table.num_rows}"
                )
            if len(unique_units) != expected_units or not np.all(counts == len(expected_time)):
                raise ValueError(f"{condition}: invalid unit/time row counts")
            if conditions != {condition}:
                raise ValueError(f"{condition}: unexpected condition values {conditions}")
            if not np.allclose(times.reshape(-1, len(expected_time)), expected_time):
                raise ValueError(f"{condition}: invalid or duplicate time grid")
            if not np.isfinite(values).all():
                raise ValueError(f"{condition}: non-finite SDF values")
            checks[condition] = {
                "rows": int(table.num_rows),
                "units": int(len(unique_units)),
                "time_samples": int(len(expected_time)),
            }
        return checks

    return export_sensorimotor_session, validate_session_export


@app.cell
def collect_metadata(DANDISET_ID, DANDISET_VERSION):
    import h5py as _h5py
    import pandas as pd
    import remfile as _remfile
    from dandi.dandiapi import DandiAPIClient as _DandiAPIClient

    # Map the paradigm-specific NWB interval table to its semantic session name.
    _session_table_to_name = {
        "Standard mismatch block_presentations": "Standard mismatch",
        "Sensory-motor mismatch block_presentations": "Sensorimotor mismatch",
        "Sequence mismatch block_presentations": "Sequence mismatch",
        "Duration mismatch block_presentations": "Duration mismatch",
    }

    _records = []
    with _DandiAPIClient() as _client:
        _dandiset = _client.get_dandiset(DANDISET_ID, version_id=DANDISET_VERSION)
        for _asset in _dandiset.get_assets():
            # Keep the canonical ecephys exports; the draft also contains duplicate NWB files.
            if not _asset.path.endswith("_ecephys.nwb"):
                continue

            _meta = _asset.get_metadata()
            _subj = _meta.wasAttributedTo[0] if _meta.wasAttributedTo else None
            _session = next(
                (
                    _activity
                    for _activity in _meta.wasGeneratedBy
                    if _activity.schemaKey == "Session"
                ),
                None,
            )

            _remote_file = _remfile.File(
                _asset.get_content_url(follow_redirects=1, strip_query=True)
            )
            _h5_file = _h5py.File(_remote_file, mode="r")
            try:
                _interval_names = set(_h5_file["intervals"])
            finally:
                _h5_file.close()
                _remote_file.close()

            _session_types = [
                _semantic_name
                for _table_name, _semantic_name in _session_table_to_name.items()
                if _table_name in _interval_names
            ]
            if len(_session_types) != 1:
                raise ValueError(
                    f"Expected one mismatch paradigm in {_asset.path}, found {_session_types}"
                )

            _records.append({
                "path": _asset.path,
                "session_id": getattr(_session, "identifier", None),
                "session_date": (
                    _session.startDate.date().isoformat()
                    if _session is not None and _session.startDate is not None
                    else None
                ),
                "session_type": _session_types[0],
                "subject": getattr(_subj, "identifier", None),
                "species": getattr(getattr(_subj, "species", None), "name", None),
                "genotype": getattr(_subj, "genotype", None),
                "sex": getattr(getattr(_subj, "sex", None), "name", None),
                "age": getattr(getattr(_subj, "age", None), "value", None),
            })

    meta_df = pd.DataFrame(_records)
    meta_df
    return meta_df, pd


@app.cell
def dataset_overview(meta_df):
    # Dataset overview: animals, genotypes, species, sex, and paradigm sessions
    _session_order = [
        "Standard mismatch",
        "Sensorimotor mismatch",
        "Sequence mismatch",
        "Duration mismatch",
    ]

    dataset_summary = {
        "total_sessions": len(meta_df),
        "unique_animals": meta_df["subject"].nunique(),
        "session_types": meta_df["session_type"].value_counts().to_dict(),
        "genotypes": meta_df["genotype"].value_counts().to_dict(),
        "species": meta_df["species"].value_counts().to_dict(),
        "sex_sessions": meta_df["sex"].value_counts().to_dict(),
    }

    _session_dates = meta_df.pivot(
        index="subject",
        columns="session_type",
        values="session_date",
    ).reindex(columns=_session_order)
    _session_dates.columns.name = None

    animal_summary = (
        meta_df.groupby("subject").agg(
            sessions=("path", "count"),
            genotype=("genotype", "first"),
            sex=("sex", "first"),
            species=("species", "first"),
        )
        .join(_session_dates)
    )

    print(dataset_summary)
    animal_summary
    return


@app.cell(hide_code=True)
def pilot_sensorimotor_export(
    DANDISET_ID,
    DANDISET_VERSION,
    DandiAPIClient,
    SDF_OUTPUT_ROOT,
    export_sensorimotor_session,
    meta_df,
    pd,
    validate_session_export,
):
    sensorimotor_sessions = (
        meta_df.loc[meta_df["session_type"] == "Sensorimotor mismatch"]
        .sort_values(["session_date", "subject"])
        .reset_index(drop=True)
    )

    _pilot_row = sensorimotor_sessions.loc[
        (sensorimotor_sessions["subject"] == "820454")
        & (sensorimotor_sessions["session_date"] == "2025-11-04")
    ].iloc[0]

    with DandiAPIClient() as _pilot_client:
        _pilot_dandiset = _pilot_client.get_dandiset(
            DANDISET_ID,
            version_id=DANDISET_VERSION,
        )
        _pilot_asset = next(
            _candidate
            for _candidate in _pilot_dandiset.get_assets()
            if _candidate.path == _pilot_row["path"]
        )
        pilot_export_result = export_sensorimotor_session(_pilot_asset, _pilot_row)

    _pilot_dir = SDF_OUTPUT_ROOT / "sub-820454_ses-2025-11-04"
    pilot_validation = validate_session_export(_pilot_dir)

    pd.DataFrame(
        [
            {
                "status": pilot_export_result["status"],
                "subject": pilot_export_result["subject"],
                "session_date": pilot_export_result["session_date"],
                "qc_units": pilot_export_result["qc_unit_count"],
                "halt_trials": pilot_export_result["trial_counts"]["motor_halt"],
                "omission_trials": pilot_export_result["trial_counts"]["motor_omission"],
                "orientation_trials": pilot_export_result["trial_counts"]["motor_orientation"],
                "output": str(_pilot_dir),
            }
        ]
    )
    return (sensorimotor_sessions,)


@app.cell(hide_code=True)
def batch_sensorimotor_export(
    DANDISET_ID,
    DANDISET_VERSION,
    DandiAPIClient,
    SDF_OUTPUT_ROOT,
    export_sensorimotor_session,
    json,
    pd,
    sensorimotor_sessions,
    validate_session_export,
):
    _batch_rows = []
    _batch_paths = set(sensorimotor_sessions["path"])
    with DandiAPIClient() as _batch_client:
        _batch_dandiset = _batch_client.get_dandiset(
            DANDISET_ID,
            version_id=DANDISET_VERSION,
        )
        _batch_assets = {
            _candidate.path: _candidate
            for _candidate in _batch_dandiset.get_assets()
            if _candidate.path in _batch_paths
        }
        _missing_assets = _batch_paths - set(_batch_assets)
        if _missing_assets:
            raise ValueError(f"Missing DANDI assets: {sorted(_missing_assets)}")

        for _, _session_row in sensorimotor_sessions.iterrows():
            _result = export_sensorimotor_session(
                _batch_assets[_session_row["path"]],
                _session_row,
            )
            _session_dir = SDF_OUTPUT_ROOT / (
                f"sub-{_session_row['subject']}_ses-{_session_row['session_date']}"
            )
            _validation = validate_session_export(_session_dir)
            _batch_rows.append(
                {
                    "status": _result["status"],
                    "subject": _result["subject"],
                    "session_id": _result["session_id"],
                    "session_date": _result["session_date"],
                    "qc_units": _result["qc_unit_count"],
                    "halt_trials": _result["trial_counts"]["motor_halt"],
                    "omission_trials": _result["trial_counts"]["motor_omission"],
                    "orientation_trials": _result["trial_counts"]["motor_orientation"],
                    "output": str(_session_dir),
                    "validated": True,
                }
            )
            print(
                f"Validated {_result['subject']} {_result['session_date']}: "
                f"{_result['qc_unit_count']} units"
            )

    batch_export_summary = pd.DataFrame(_batch_rows).sort_values(
        ["session_date", "subject"]
    ).reset_index(drop=True)

    _batch_manifest = {
        "dandiset_id": DANDISET_ID,
        "dandiset_version": DANDISET_VERSION,
        "session_type": "Sensorimotor mismatch",
        "expected_sessions": int(len(sensorimotor_sessions)),
        "completed_sessions": int(len(batch_export_summary)),
        "total_qc_units": int(batch_export_summary["qc_units"].sum()),
        "sessions": batch_export_summary.to_dict(orient="records"),
    }
    with (SDF_OUTPUT_ROOT / "batch_manifest.json").open("w", encoding="utf-8") as _stream:
        json.dump(_batch_manifest, _stream, indent=2)

    batch_export_summary
    return (batch_export_summary,)


@app.cell(hide_code=True)
def verify_sensorimotor_exports(
    AMPLITUDE_CUTOFF_THRESHOLD,
    ISI_VIOLATIONS_RATIO_THRESHOLD,
    MINIMUM_FIRING_RATE,
    PRESENCE_RATIO_THRESHOLD,
    Path,
    batch_export_summary,
    json,
    np,
    pd,
    pq,
    sensorimotor_sessions,
):
    _verification_rows = []
    for _session in batch_export_summary.itertuples(index=False):
        _session_dir = Path(_session.output)
        _units = pq.read_table(_session_dir / "units.parquet").to_pydict()
        _qc_ok = (
            np.all(np.asarray(_units["firing_rate"]) > MINIMUM_FIRING_RATE)
            and np.all(
                np.asarray(_units["amplitude_cutoff"])
                < AMPLITUDE_CUTOFF_THRESHOLD
            )
            and np.all(
                np.asarray(_units["isi_violations_ratio"])
                < ISI_VIOLATIONS_RATIO_THRESHOLD
            )
            and np.all(
                np.asarray(_units["presence_ratio"])
                > PRESENCE_RATIO_THRESHOLD
            )
        )
        with (_session_dir / "manifest.json").open(encoding="utf-8") as _stream:
            _manifest = json.load(_stream)
        _files_complete = all(
            (_session_dir / _filename).is_file()
            for _filename in [
                "motor_halt.parquet",
                "motor_omission.parquet",
                "motor_orientation.parquet",
                "units.parquet",
                "manifest.json",
                "_SUCCESS",
            ]
        )
        _verification_rows.append(
            {
                "subject": _session.subject,
                "session_date": _session.session_date,
                "qc_units": len(_units["unit_id"]),
                "qc_thresholds_pass": bool(_qc_ok),
                "trial_counts_pass": _manifest["trial_counts"]
                == {"motor_halt": 35, "motor_omission": 35, "motor_orientation": 70},
                "files_complete": _files_complete,
            }
        )

    export_verification = pd.DataFrame(_verification_rows)
    if len(export_verification) != len(sensorimotor_sessions):
        raise ValueError("Exported session count does not match sensorimotor metadata")
    if not export_verification[
        ["qc_thresholds_pass", "trial_counts_pass", "files_complete"]
    ].all().all():
        raise ValueError("One or more exported sessions failed final verification")

    export_verification
    return


if __name__ == "__main__":
    app.run()
