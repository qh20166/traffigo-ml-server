# -*- coding: utf-8 -*-
"""
train_model.py — Huấn luyện mô hình dự đoán tốc độ giao thông cho server ML TraffiGo.

Dữ liệu:
  1. Toàn bộ snapshot traffic_hcm_*.csv của đồ án DATN_172 ("Giải pháp phân tích tình
     trạng giao thông dựa trên kỹ thuật gom cụm dữ liệu", HCMUT) — ~170k mẫu.
  2. Thời tiết ngày của TP.HCM cho đúng khoảng thời gian các snapshot (Open-Meteo
     archive API, đã lưu sẵn ở hcmc_weather_daily.json) — khớp mô tả hệ thống trong
     báo cáo: mô hình dùng cả đặc trưng thời tiết.

Mô hình (theo mục 5.1 báo cáo đồ án — học máy giám sát, dự đoán theo từng đoạn đường):
  - RandomForestRegressor dự đoán currentSpeed (km/h) từ đặc trưng hạ tầng + thời gian
    + thời tiết. KHÔNG dùng currentSpeed/congestionIndex/jamFactor làm input (leakage —
    các chỉ số này được suy ra trực tiếp từ currentSpeed).
  - KMeans k=6 (mô hình gom cụm của đồ án, copy từ cluster_model.json của app) gán mỗi
    đoạn đường vào 1 trong 6 trạng thái giao thông đã diễn giải trong báo cáo.

Chạy:  python train_model.py
Đầu ra: models/speed_rf.joblib, models/cluster_model.json, models/meta.json
"""
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

SNAPSHOT_GLOB = r"E:\DATN_172\DATA\DATN\traffic_hcm_*.csv"
WEATHER_JSON = os.path.join(HERE, "hcmc_weather_daily.json")
CLUSTER_MODEL_SRC = r"E:\Android\TraffiGo\app\src\main\assets\cluster_model.json"

# Đặc trưng đầu vào (theo mục 5.1.2-5.1.3 báo cáo: thuộc tính số + phân loại one-hot)
NUMERIC_FEATURES = [
    "freeFlowSpeed",
    "speedLimit",
    "hourOfDay",
    "dayOfWeek",
    "laneCount_aggregated",
    "intersectionCount",
    "routeSlopePercent",
    "curvatureIndex",
    "lengthKm",
    # thời tiết (join theo ngày từ Open-Meteo archive)
    "precipitation_sum",
    "windspeed_10m_max",
    "temperature_2m_max",
]
WEATHER_FEATURES = ["precipitation_sum", "windspeed_10m_max", "temperature_2m_max"]
CATEGORICAL_FEATURES = ["roadType", "surface", "dayType"]  # one-hot
TARGET = "currentSpeed"


def load_snapshots() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(SNAPSHOT_GLOB)):
        try:
            df = pd.read_csv(path)
            if "currentSpeed" in df.columns and "freeFlowSpeed" in df.columns:
                frames.append(df)
        except Exception as e:
            print(f"[train] bỏ qua {os.path.basename(path)}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns=lambda c: c.strip())

    # timeStamp dạng yymmddhhmm (vd 2603101003 = 2026-03-10 10:03) -> ngày ISO để join thời tiết
    ts = pd.to_numeric(df["timeStamp"], errors="coerce").astype("Int64")
    iso = ts.map(lambda v: f"20{str(v)[:2]}-{str(v)[2:4]}-{str(v)[4:6]}" if pd.notna(v) else None)
    df["date_iso"] = iso

    for col in [c for c in NUMERIC_FEATURES if c not in WEATHER_FEATURES] + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df.loc[df[col].isin(["nan", "none", "", "<na>"]), col] = "unknown"

    df = df[(df[TARGET] > 0) & (df[TARGET] < 150)]
    df = df[(df["hourOfDay"].between(0, 23)) & (df["dayOfWeek"].between(0, 6))]
    df = df.dropna(subset=[TARGET] + [c for c in NUMERIC_FEATURES if c not in WEATHER_FEATURES])
    return df.reset_index(drop=True)


def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Join thời tiết ngày (temperature/precipitation/windspeed) theo date_iso."""
    if not os.path.exists(WEATHER_JSON):
        print("[train] Không có hcmc_weather_daily.json -> bỏ đặc trưng thời tiết")
        for c in ["precipitation_sum", "windspeed_10m_max", "temperature_2m_max"]:
            df[c] = 0.0
        return df
    with open(WEATHER_JSON, encoding="utf-8") as f:
        w = json.load(f)
    wdf = pd.DataFrame(w).rename(columns={"time": "date_iso"})
    # Thiếu giá trị -> 0 (không mưa, không gió); nhiệt độ thiếu -> median
    for c in ["precipitation_sum", "windspeed_10m_max", "temperature_2m_max"]:
        wdf[c] = pd.to_numeric(wdf.get(c), errors="coerce")
    wdf["temperature_2m_max"] = wdf["temperature_2m_max"].fillna(wdf["temperature_2m_max"].median())
    wdf[["precipitation_sum", "windspeed_10m_max"]] = wdf[["precipitation_sum", "windspeed_10m_max"]].fillna(0.0)
    df = df.merge(wdf, on="date_iso", how="left")
    has_weather = df["temperature_2m_max"].notna().sum()
    print(f"[train] Join thời tiết: {has_weather}/{len(df)} mẫu có dữ liệu thời tiết")
    df["precipitation_sum"] = df["precipitation_sum"].fillna(0.0)
    df["windspeed_10m_max"] = df["windspeed_10m_max"].fillna(0.0)
    df["temperature_2m_max"] = df["temperature_2m_max"].fillna(df["temperature_2m_max"].median())
    return df


def main():
    print(f"[train] Đọc snapshot từ {SNAPSHOT_GLOB}")
    df = load_snapshots()
    print(f"[train] Sau khi làm sạch: {len(df)} mẫu, {df['segmentId'].nunique()} segment")
    df = attach_weather(df)
    if len(df) < 100:
        print("[train] Dữ liệu quá ít.", file=sys.stderr)
        sys.exit(1)

    X_num = df[NUMERIC_FEATURES].astype(float)
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # Cấu hình "lite" cho môi trường 512MB RAM (Render free):
    # 60 cây + giới hạn độ sâu + lá dày hơn -> bộ nhớ ~1/3, độ chính xác gần tương đương
    model = RandomForestRegressor(
        n_estimators=60,
        min_samples_leaf=6,
        max_depth=22,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    print(f"[train] MAE={mae:.2f} km/h | RMSE={rmse:.2f} km/h | R2={r2:.3f} (n_test={len(y_test)})")

    # Baseline: luôn dự đoán = freeFlowSpeed (tức là giả định đường luôn thoáng)
    ff_test = X_test["freeFlowSpeed"].to_numpy()
    base_mae = float(mean_absolute_error(y_test, ff_test))
    print(f"[train] Baseline 'dự đoán = freeFlowSpeed': MAE={base_mae:.2f} km/h")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "speed_rf.joblib"), compress=9)

    # Copy mô hình gom cụm k=6 của đồ án vào models/ để server tự chứa, không phụ thuộc thư mục app
    with open(CLUSTER_MODEL_SRC, encoding="utf-8-sig") as f:
        cluster_model = json.load(f)
    with open(os.path.join(MODELS_DIR, "cluster_model.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_model, f, ensure_ascii=False)

    meta = {
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "one_hot_columns": [c for c in X.columns if c not in NUMERIC_FEATURES],
        "metrics": {
            "mae_kmh": round(mae, 2),
            "rmse_kmh": round(rmse, 2),
            "r2": round(r2, 3),
            "baseline_freeflow_mae_kmh": round(base_mae, 2),
            "n_test": int(len(y_test)),
        },
        "n_samples": int(len(df)),
        "has_weather": bool(os.path.exists(WEATHER_JSON)),
        "sklearn_version": __import__("sklearn").__version__,
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[train] Đã lưu model + meta vào {MODELS_DIR} (nén level 9 để vừa giới hạn 100MB/file của GitHub)")


if __name__ == "__main__":
    main()
