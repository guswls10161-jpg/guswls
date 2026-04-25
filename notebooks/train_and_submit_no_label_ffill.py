import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

ROOT = Path(r"C:\Users\pjs87\guswls")
SOURCE = Path(r"C:\Users\pjs87\phems_outputs\run_20260425_1337")
OUT = Path(r"C:\Users\pjs87\phems_outputs\run_20260425_no_label_ffill")


def log(msg: str) -> None:
    print(msg, flush=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Post-pipeline features with chronological cumulative calculations."""
    print(f"[Features] Input: {df.shape}", flush=True)
    df = df.copy()

    SBP = "systolic_blood_pressure"
    DBP = "diastolic_blood_pressure"
    HR = "heart_rate"
    TEMP = "body_temperature"
    RR = "respiratory_rate"
    SPO2 = "measurement_of_oxygen_saturation_at_periphery"
    GCS = "glasgow_coma_scale"
    LACT = "lactate_moles_volume_in_blood"
    CRP = "c_reactive_protein_mass_volume_in_serum_or_plasma"
    WBC = "white_blood_cell_count"
    CREA = "creatinine_mass_volume_in_blood"
    FIO2 = "oxygen_gas_total_pure_volume_fraction_inhaled_gas"
    PAO2 = "oxygen_partial_pressure_in_arterial_blood"

    df["qsofa"] = 0
    for col, cond in [(SBP, lambda x: x <= 100), (RR, lambda x: x >= 22), (GCS, lambda x: x < 15)]:
        if col in df.columns:
            df["qsofa"] += cond(df[col]).astype(int).fillna(0)

    df["sirs"] = 0
    if TEMP in df.columns:
        df["sirs"] += ((df[TEMP] > 38) | (df[TEMP] < 36)).astype(int).fillna(0)
    if HR in df.columns:
        df["sirs"] += (df[HR] > 90).astype(int).fillna(0)
    if RR in df.columns:
        df["sirs"] += (df[RR] > 20).astype(int).fillna(0)
    if WBC in df.columns:
        df["sirs"] += ((df[WBC] > 12) | (df[WBC] < 4)).astype(int).fillna(0)

    if HR in df.columns and SBP in df.columns:
        df["shock_idx"] = df[HR] / df[SBP].replace(0, np.nan)
    if SBP in df.columns and DBP in df.columns:
        df["map_press"] = (df[SBP] + 2 * df[DBP]) / 3
    if PAO2 in df.columns and FIO2 in df.columns:
        df["pf_ratio"] = df[PAO2] / df[FIO2].replace(0, np.nan)

    if "drug_str" in df.columns:
        ds = df["drug_str"].fillna("").astype(str)
        vps = ["epinephrine", "norepinephrine", "dopamine", "phenylephrine", "dobutamine", "milrinone", "levosimendan"]
        abx_broad = ["meropenem", "piperacillin", "ceftazidime", "cefepime", "colistin"]
        abx_all = abx_broad + [
            "vancomycin", "amoxicillin", "cefotaxime", "ampicillin", "tobramycin",
            "gentamicin", "ciprofloxacin", "linezolid",
        ]

        df["on_vp"] = ds.apply(lambda x: int(any(v in x for v in vps)))
        df["on_abx"] = ds.apply(lambda x: int(any(v in x for v in abx_all)))
        df["on_broad_abx"] = ds.apply(lambda x: int(any(v in x for v in abx_broad)))
        df["vp_plus_abx"] = ((df["on_vp"] == 1) & (df["on_abx"] == 1)).astype(int)
        if SBP in df.columns:
            df["vp_hypo"] = ((df["on_vp"] == 1) & (df[SBP] < 90)).astype(int)
        if "map_press" in df.columns:
            df["vp_map_fail"] = ((df["on_vp"] == 1) & (df["map_press"] < 65)).astype(int)

    df["measurement_datetime"] = pd.to_datetime(df["measurement_datetime"])
    df = df.sort_values(["person_id", "measurement_datetime", "visit_occurrence_id"], kind="mergesort")

    grp_key = "person_id"
    trajectory_cols = [c for c in [HR, SBP, TEMP, RR, LACT] if c in df.columns]
    for col in trajectory_cols:
        short = col.split("_")[0][:8]
        expanding = df.groupby(grp_key, sort=False)[col].expanding(min_periods=1)

        cum_mean = expanding.mean().reset_index(level=0, drop=True)
        cum_std = expanding.std().reset_index(level=0, drop=True)
        cum_min = expanding.min().reset_index(level=0, drop=True)
        cum_max = expanding.max().reset_index(level=0, drop=True)

        df[f"{short}_cmean"] = cum_mean
        df[f"{short}_cstd"] = cum_std
        df[f"{short}_cmin"] = cum_min
        df[f"{short}_cmax"] = cum_max
        df[f"{short}_dev"] = df[col] - cum_mean
        rng = cum_max - cum_min
        df[f"{short}_pos"] = (df[col] - cum_min) / rng.replace(0, np.nan)

    df["hour_of_day"] = df["measurement_datetime"].dt.hour
    df["is_night"] = df["hour_of_day"].between(0, 6).astype(int)
    df["meas_order"] = df.groupby(grp_key, sort=False).cumcount() + 1

    abnormals = []
    if HR in df.columns:
        abnormals.append((df[HR] > 130) | (df[HR] < 50))
    if SBP in df.columns:
        abnormals.append(df[SBP] < 90)
    if TEMP in df.columns:
        abnormals.append((df[TEMP] > 38.5) | (df[TEMP] < 35.5))
    if RR in df.columns:
        abnormals.append((df[RR] > 25) | (df[RR] < 8))
    if SPO2 in df.columns:
        abnormals.append(df[SPO2] < 92)
    if GCS in df.columns:
        abnormals.append(df[GCS] < 12)
    if abnormals:
        df["n_abnormal"] = sum(c.fillna(False).astype(int) for c in abnormals)

    delta_cols = [c for c in [HR, SBP, LACT, CRP, CREA] if c in df.columns]
    for col in delta_cols:
        short = col.split("_")[0][:8]
        df[f"{short}_delta"] = df.groupby(grp_key, sort=False)[col].diff()

    print(f"[Features] Output: {df.shape}", flush=True)
    return df


def person_datetime_id(df: pd.DataFrame) -> pd.Series:
    return df["person_id"].astype(int).astype(str) + "_" + pd.to_datetime(df["measurement_datetime"]).astype(str)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    log("[run] Loading feature tables")
    train_df = pd.read_csv(SOURCE / "train_data_w_features_FINAL.csv")
    test_df = pd.read_csv(SOURCE / "test_data_w_features_FINAL.csv")
    labels = pd.read_csv(ROOT / "data" / "training_data" / "SepsisLabel_train.csv")
    test_index = pd.read_csv(ROOT / "data" / "testing_data" / "SepsisLabel_test.csv")

    train_df["measurement_datetime"] = pd.to_datetime(train_df["measurement_datetime"])
    test_df["measurement_datetime"] = pd.to_datetime(test_df["measurement_datetime"])
    labels["measurement_datetime"] = pd.to_datetime(labels["measurement_datetime"])
    test_index["measurement_datetime"] = pd.to_datetime(test_index["measurement_datetime"])

    # Drop cached labels before feature engineering so derived features cannot
    # accidentally use a target column if add_features changes later.
    label_cols = ["sepsislabel", "SepsisLabel"]
    cached_label_cols = [c for c in label_cols if c in train_df.columns or c in test_df.columns]
    if cached_label_cols:
        log(f"[run] Dropping cached label columns before add_features: {cached_label_cols}")
    train_df = train_df.drop(columns=label_cols, errors="ignore")
    test_df = test_df.drop(columns=label_cols, errors="ignore")

    log("[run] Applying post-pipeline add_features")
    train_df = add_features(train_df)
    test_df = add_features(test_df)
    if any(c.lower() == "sepsislabel" for c in train_df.columns) or any(c.lower() == "sepsislabel" for c in test_df.columns):
        raise RuntimeError("Label column appeared before official label merge.")

    # Fix #2: use only official label rows. Duplicate labels are all equal except one conflict;
    # max preserves the manual correction used in the notebook for 2024-09-09 09:00.
    labels = labels.groupby(["person_id", "measurement_datetime"], as_index=False)["SepsisLabel"].max()
    train_df = train_df.drop(columns=["sepsislabel", "SepsisLabel"], errors="ignore").merge(
        labels.rename(columns={"SepsisLabel": "sepsislabel"}),
        on=["person_id", "measurement_datetime"],
        how="inner",
    )

    y = train_df["sepsislabel"].astype(int)
    drop_cols = [
        "drug_concept_id", "route_concept_id", "drug_str", "route_str",
        "sepsislabel", "person_id", "measurement_datetime", "visit_occurrence_id",
    ]
    X = train_df.drop(columns=drop_cols, errors="ignore").copy()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in categorical_features:
        X[col] = X[col].fillna("missing").astype(str)

    X_test = test_df.drop(columns=["person_id_datetime"] + drop_cols, errors="ignore").copy()
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    extra_cols = [col for col in X_test.columns if col not in X.columns]
    if extra_cols:
        X_test = X_test.drop(columns=extra_cols)
    X_test = X_test[X.columns]
    for col in categorical_features:
        X_test[col] = X_test[col].fillna("missing").astype(str)

    groups = train_df["person_id"]
    gkf = GroupKFold(n_splits=5)
    params = {
        "iterations": 1000,
        "learning_rate": 0.01,
        "depth": 6,
        "task_type": "CPU",
        "eval_metric": "PRAUC",
        "early_stopping_rounds": 100,
        "class_weights": {0: 1.0, 1: 110.0},
        "verbose": 100,
    }

    log(f"[run] Train rows={len(X)} positives={int(y.sum())} features={X.shape[1]} cats={categorical_features}")
    oof = np.zeros(len(X), dtype=float)
    test_preds = []
    folds = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        log(f"[run] Training fold {fold}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostClassifier(**params)
        model.fit(
            Pool(X_train, y_train, cat_features=categorical_features),
            eval_set=Pool(X_val, y_val, cat_features=categorical_features),
            verbose=100,
        )
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        prauc = float(average_precision_score(y_val, val_pred))
        best = float(model.best_score_["validation"]["PRAUC"])
        folds.append({"fold": fold, "average_precision": prauc, "catboost_prauc": best, "n_val": int(len(val_idx)), "pos_val": int(y_val.sum())})
        log(f"[run] Fold {fold} AP={prauc} CatBoostPRAUC={best}")
        model.save_model(str(OUT / f"catboost_fold_{fold}.cbm"))
        test_preds.append(model.predict_proba(Pool(X_test, cat_features=categorical_features))[:, 1])

    mean_pred = np.mean(np.vstack(test_preds), axis=0)
    pred_df = pd.DataFrame({"person_id_datetime": person_datetime_id(test_df), "SepsisLabel": mean_pred}).drop_duplicates("person_id_datetime")
    test_index["person_id_datetime"] = person_datetime_id(test_index)
    submission = test_index[["person_id_datetime"]].merge(pred_df, on="person_id_datetime", how="left")
    missing = int(submission["SepsisLabel"].isna().sum())
    if missing:
        raise RuntimeError(f"Missing predictions: {missing}")
    submission.to_csv(OUT / "submission.csv", index=False)

    metrics = {
        "train_rows": int(len(X)),
        "positive_rows": int(y.sum()),
        "n_features": int(X.shape[1]),
        "categorical_features": categorical_features,
        "folds": folds,
        "mean_fold_average_precision": float(np.mean([f["average_precision"] for f in folds])),
        "mean_fold_catboost_prauc": float(np.mean([f["catboost_prauc"] for f in folds])),
        "oof_average_precision": float(average_precision_score(y, oof)),
        "submission_rows": int(len(submission)),
        "submission_mean": float(submission["SepsisLabel"].mean()),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log("[run] DONE")
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
