# -*- coding: utf-8 -*-
"""
app.py — Server ML TraffiGo (FastAPI).

Hiện thực bộ API Machine Learning của hệ thống theo Bảng 17 trong báo cáo đồ án
DATN_172: /api/ml/predict, /api/ml/predict-batch, /api/ml/recommend-route,
/api/ml/compare-routes, /api/ml/detect-incident(s), /api/ml/traffic-status.

Mô hình:
  - RandomForestRegressor (train_model.py) dự đoán tốc độ lưu thông theo đoạn đường
    từ đặc trưng hạ tầng + thời gian + thời tiết.
  - KMeans k=6 của đồ án (models/cluster_model.json) gán trạng thái giao thông,
    sắp xếp mức độ kẹt theo cluster_severity_rank — cùng mô hình đang nhúng trong app.
  - Phát hiện sự cố: luật trên tỉ lệ tốc độ + sai số giữa tốc độ dự đoán và thực đo.

Chạy:
  pip install -r requirements.txt
  python train_model.py            (1 lần, tạo model trong models/)
  uvicorn app:app --host 0.0.0.0 --port 8000
  Swagger UI: http://localhost:8000/docs
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
APP_DIR = os.path.abspath(os.path.join(HERE, ".."))
LOCAL_PROPERTIES = os.path.join(APP_DIR, "local.properties")
GEOMETRY_CSV = os.path.join(APP_DIR, "app", "src", "main", "assets", "geometry.csv")

# ===== Cấu hình bộ cập nhật dữ liệu live (TomTom) =====
# Mặc định: 50 đoạn đầu trong geometry.csv, mỗi 15 phút (giữ trong quota free).
REFRESH_ENABLED = os.environ.get("TRAFFIGO_REFRESH_DISABLE", "") != "1"
REFRESH_INTERVAL = int(os.environ.get("TRAFFIGO_REFRESH_INTERVAL", "900"))   # giây
REFRESH_MAX_SEGMENTS = int(os.environ.get("TRAFFIGO_REFRESH_MAX", "50"))

app = FastAPI(
    title="TraffiGo ML Server",
    description="API dự đoán giao thông bằng Machine Learning — đồ án DATN_172",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ============================== Tải model lúc khởi động ==============================

class ClusterModel:
    """KMeans k=6 của đồ án: nearest-centroid trên đặc trưng đã chuẩn hóa."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.feature_columns = raw["feature_columns"]
        self.mean = np.array(raw["scaler_mean"])
        self.scale = np.array(raw["scaler_scale"])
        self.centroids = {int(k): np.array(v) for k, v in raw["centroids"].items()}
        self.severity_rank = {int(k): int(v) for k, v in raw["cluster_severity_rank"].items()}

    def classify(self, features: Dict[str, float]) -> tuple:
        x = np.array([float(features.get(c, 0.0)) for c in self.feature_columns])
        xs = (x - self.mean) / np.where(self.scale == 0, 1.0, self.scale)
        best = min(self.centroids.items(), key=lambda kv: float(np.sum((xs - kv[1]) ** 2)))
        cid = best[0]
        return cid, self.severity_rank.get(cid, 5)


try:
    SPEED_MODEL = joblib.load(os.path.join(MODELS_DIR, "speed_rf.joblib"))
    with open(os.path.join(MODELS_DIR, "meta.json"), encoding="utf-8") as f:
        MODEL_META = json.load(f)
    print(f"[server] Đã nạp model dự đoán tốc độ (MAE={MODEL_META['metrics']['mae_kmh']} km/h)")
except Exception as e:  # model chưa train — server vẫn chạy ở chế độ heuristic
    SPEED_MODEL = None
    MODEL_META = {"error": str(e)}
    print(f"[server] CẢNH BÁO: chưa nạp được model ({e}) -> dùng heuristic")

try:
    CLUSTER = ClusterModel(os.path.join(MODELS_DIR, "cluster_model.json"))
    print("[server] Đã nạp mô hình gom cụm KMeans k=6 của đồ án")
except Exception as e:
    CLUSTER = None
    print(f"[server] CẢNH BÁO: không nạp được cluster_model.json ({e})")


# ============================== Schemas ==============================

class SegmentFeatures(BaseModel):
    """Đặc trưng 1 đoạn đường. currentSpeed/freeFlowSpeed/jamFactor theo định dạng
    ví dụ trong báo cáo; các trường còn lại khớp cột của traffic_data.csv."""

    currentSpeed: float = Field(default=0, description="Tốc độ đo được hiện tại (km/h)")
    freeFlowSpeed: float = Field(default=50, description="Tốc độ dòng tự do (km/h)")
    jamFactor: float = Field(default=0, ge=0, le=10)
    hourOfDay: int = Field(default=12, ge=0, le=23)
    dayOfWeek: int = Field(default=1, ge=0, le=6, description="0=CN .. 6=Thứ 7")
    speedLimit: Optional[float] = None
    laneCount: Optional[float] = None
    intersectionCount: Optional[float] = None
    routeSlopePercent: Optional[float] = None
    curvatureIndex: Optional[float] = None
    lengthKm: Optional[float] = None
    roadType: Optional[str] = None
    surface: Optional[str] = None
    dayType: Optional[str] = None  # weekday/weekend — tự suy nếu bỏ trống
    # Thời tiết (nếu app/server không có thì dùng trung bình theo mùa của bộ dữ liệu)
    precipitationSum: Optional[float] = None
    windspeed: Optional[float] = None
    temperature: Optional[float] = None


class PredictRequest(BaseModel):
    segment: SegmentFeatures


class PredictBatchRequest(BaseModel):
    segments: List[SegmentFeatures]


class RouteCandidate(BaseModel):
    id: str
    durationSec: int = 0
    distanceMeters: int = 0
    samples: List[SegmentFeatures] = Field(default_factory=list)


class RecommendRouteRequest(BaseModel):
    routes: List[RouteCandidate] = Field(..., min_length=1)


class CompareRoutesRequest(BaseModel):
    routeA: RouteCandidate
    routeB: RouteCandidate


class TrafficStatusRequest(BaseModel):
    segments: List[SegmentFeatures] = Field(default_factory=list)


# ============================== Helpers ==============================

# Thời tiết trung bình mùa khô TP.HCM (fallback khi client không gửi)
DEFAULT_WEATHER = {"precipitation_sum": 0.5, "windspeed_10m_max": 12.0, "temperature_2m_max": 32.0}

# Ngưỡng mức giao thông — nhất quán với quy ước congestionIndex của app
# (congestionIndex thấp = kẹt): <0.7 HIGH, <0.85 MODERATE, ngược lại LOW.
LEVEL_THRESHOLDS = ((0.70, "High"), (0.85, "Moderate"))


def level_for_ratio(ratio: float) -> str:
    for th, name in LEVEL_THRESHOLDS:
        if ratio < th:
            return name
    return "Low"


def infer_day_type(day_of_week: int) -> str:
    return "weekend" if day_of_week in (0, 6) else "weekday"


def feature_vector(s: SegmentFeatures) -> Dict[str, Any]:
    """Dựng vector đặc trưng cho model dự đoán tốc độ."""
    free_flow = float(s.freeFlowSpeed or 50.0)
    w = DEFAULT_WEATHER
    return {
        "freeFlowSpeed": free_flow,
        "speedLimit": float(s.speedLimit) if s.speedLimit is not None else free_flow,
        "hourOfDay": float(s.hourOfDay),
        "dayOfWeek": float(s.dayOfWeek),
        "laneCount_aggregated": float(s.laneCount) if s.laneCount is not None else 1.0,
        "intersectionCount": float(s.intersectionCount or 0.0),
        "routeSlopePercent": float(s.routeSlopePercent or 0.0),
        "curvatureIndex": float(s.curvatureIndex or 0.0),
        "lengthKm": float(s.lengthKm) if s.lengthKm is not None else 1.0,
        "precipitation_sum": float(s.precipitationSum) if s.precipitationSum is not None else w["precipitation_sum"],
        "windspeed_10m_max": float(s.windspeed) if s.windspeed is not None else w["windspeed_10m_max"],
        "temperature_2m_max": float(s.temperature) if s.temperature is not None else w["temperature_2m_max"],
        "roadType": (s.roadType or "unknown").lower(),
        "surface": (s.surface or "unknown").lower(),
        "dayType": (s.dayType or infer_day_type(s.dayOfWeek)).lower(),
    }


def predict_speed(vec: Dict[str, Any]) -> float:
    """Dự đoán tốc độ: RF nếu đã train, không thì heuristic theo tỉ lệ giờ cao điểm."""
    if SPEED_MODEL is not None:
        cols = MODEL_META["numeric_features"] + MODEL_META["one_hot_columns"]
        row = []
        for c in cols:
            v = vec.get(c, 0.0)
            row.append(float(v) if isinstance(v, (int, float, np.floating)) else 0.0)
        # one-hot: cột dạng "roadType.trunk" khớp giá trị phân loại
        for c in MODEL_META["one_hot_columns"]:
            if "." in c:
                prefix, val = c.split(".", 1)
                row[cols.index(c)] = 1.0 if vec.get(prefix) == val else 0.0
        return float(np.clip(SPEED_MODEL.predict([row])[0], 2.0, 150.0))
    # Heuristic: giờ cao điểm sáng 7-9 / chiều 17-20 kéo giảm 30%
    peak = (7 <= vec["hourOfDay"] <= 9) or (17 <= vec["hourOfDay"] <= 20)
    weekend = vec["dayType"] == "weekend"
    factor = 0.72 if (peak and not weekend) else (0.85 if peak else 1.0)
    if vec["precipitation_sum"] > 5:
        factor *= 0.85  # mưa to đi chậm thêm
    return float(np.clip(vec["freeFlowSpeed"] * factor, 2.0, 150.0))


def cluster_features_for(s: SegmentFeatures, predicted_speed: float) -> Dict[str, float]:
    """9 đặc trưng KMeans của đồ án — cùng công thức DATA/src/h.py và app (buildClusterFeatures)."""
    free_flow = float(s.freeFlowSpeed or 50.0)
    speed = float(s.currentSpeed) if s.currentSpeed and s.currentSpeed > 0 else predicted_speed
    congestion = speed / free_flow if free_flow > 0 else 1.0
    length_km = float(s.lengthKm) if s.lengthKm is not None else 1.0
    speed_limit = float(s.speedLimit) if s.speedLimit is not None else free_flow
    cross_time = length_km / (speed / 3.6) if speed > 0 else 0.0
    lanes = float(s.laneCount) if s.laneCount is not None else 1.0
    utilization = max(0.05, min(1.0, 1.2 - congestion))
    return {
        "speedLimitRatio": speed / speed_limit if speed_limit > 0 else 1.0,
        "crossTime": cross_time,
        "trafficVolume": 1800.0 * lanes * utilization,
        "congestionIndex": congestion,
        "occupancy": 100.0 * (1.0 - congestion),
        "relativeCongestionIndex": 1.0 - congestion,
        "freeFlowSpeed": free_flow,
        "lengthKm": length_km,
        "curvatureIndex": float(s.curvatureIndex or 0.0),
    }


def incident_risk(s: SegmentFeatures, predicted_speed: float) -> Dict[str, Any]:
    """Phát hiện sự cố: luật trên tỉ lệ tốc độ + chênh lệch thực đo vs dự đoán."""
    free_flow = float(s.freeFlowSpeed or 50.0)
    ratio = (float(s.currentSpeed) / free_flow) if (free_flow > 0 and s.currentSpeed) else 1.0
    risk = "HIGH" if ratio < 0.30 else ("MEDIUM" if ratio < 0.55 else "LOW")
    incident = False
    confidence = 0.0
    reason = "Tốc độ bình thường so với dòng tự do"
    if s.currentSpeed and s.currentSpeed > 0:
        drop = predicted_speed - float(s.currentSpeed)
        if drop > 15 and ratio < 0.5:
            incident = True
            confidence = min(0.95, 0.5 + drop / 60.0)
            reason = f"Tốc độ thực đo thấp hơn dự đoán {drop:.0f} km/h — nghi sự cố/tai nạn"
    return {"risk": risk, "incident": incident, "confidence": round(confidence, 2), "reason": reason}


def predict_segment(s: SegmentFeatures) -> Dict[str, Any]:
    vec = feature_vector(s)
    predicted_speed = predict_speed(vec)
    ratio = (predicted_speed / vec["freeFlowSpeed"]) if vec["freeFlowSpeed"] > 0 else 1.0
    level = level_for_ratio(ratio)
    cluster_id, severity_rank = (None, None)
    if CLUSTER is not None:
        cluster_id, severity_rank = CLUSTER.classify(cluster_features_for(s, predicted_speed))
    return {
        "predictedSpeed": round(predicted_speed, 1),
        "trafficLevel": level,
        "congestionIndex": round(ratio, 3),
        "clusterId": cluster_id,
        "severityRank": severity_rank,
        "incident": incident_risk(s, predicted_speed),
    }


def route_score(route: RouteCandidate) -> Dict[str, Any]:
    """Điểm tuyến: tổng thời gian dự đoán theo tốc độ ML tại các điểm mẫu,
    quy về toàn tuyến theo tỉ lệ độ dài mẫu. Càng nhỏ càng tốt."""
    if not route.samples:
        return {"id": route.id, "predictedDurationSec": float(route.durationSec), "avgLevel": "n/a"}
    vecs = [feature_vector(s) for s in route.samples]
    speeds = [max(3.0, predict_speed(v)) for v in vecs]
    lengths = [max(0.05, v["lengthKm"]) for v in vecs]
    sample_km = sum(lengths)
    hours = sum(l / sp for l, sp in zip(lengths, speeds))
    full_km = max(sample_km, route.distanceMeters / 1000.0, 1e-6)
    predicted_sec = hours * 3600.0 * (full_km / sample_km)
    # Kết hợp duration gốc (Google đã tính traffic) — lấy trung bình 2 ước lượng
    blended = predicted_sec if not route.durationSec else (predicted_sec + route.durationSec) / 2.0
    ratios = [sp / v["freeFlowSpeed"] for sp, v in zip(speeds, vecs) if v["freeFlowSpeed"] > 0]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
    return {
        "id": route.id,
        "predictedDurationSec": round(blended),
        "predictedSpeedAvg": round(sum(speeds) / len(speeds), 1),
        "avgCongestionIndex": round(avg_ratio, 3),
        "avgLevel": level_for_ratio(avg_ratio),
    }


# ============================== Endpoints ==============================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelLoaded": SPEED_MODEL is not None,
        "clusterModelLoaded": CLUSTER is not None,
        "metrics": MODEL_META.get("metrics"),
    }


# ============================== Bộ cập nhật dữ liệu live (TomTom) ==============================
# Server KHÔNG chỉ phục vụ dự đoán từ model: một luồng nền chạy định kỳ gọi TomTom Flow
# API cho một tập đoạn đường giám sát và lưu snapshot mới nhất trong bộ nhớ. Nhờ đó
# /api/ml/live-status luôn trả về trạng thái giao thông "đà lấy gần nhất" kèm mốc thời gian.

def _load_tomtom_keys() -> List[str]:
    env = os.environ.get("TOMTOM_KEYS", "")
    keys = [k.strip() for k in env.split(",") if k.strip()]
    if keys:
        return keys
    try:
        with open(LOCAL_PROPERTIES, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("TOMTOM_KEYS="):
                    return [k.strip() for k in line.split("=", 1)[1].split(",") if k.strip()]
    except Exception:
        pass
    return []


def _load_monitored_segments() -> List[Dict[str, Any]]:
    """Đọc geometry.csv của app, lấy tối đa REFRESH_MAX_SEGMENTS đoạn đầu tiên.
    Chỉ cần 5 cột đầu (tên đường + tọa độ hai đầu) — đứng trước cột geometry quoted
    nên tách đơn giản bằng dấu phẩy là đủ. Điểm giám sát = trung bình hai đầu đoạn."""
    segs = []
    try:
        with open(GEOMETRY_CSV, encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                cols = line.split(",")
                if len(cols) < 5:
                    continue
                try:
                    lat = (float(cols[2]) + float(cols[4])) / 2
                    lon = (float(cols[1]) + float(cols[3])) / 2
                except ValueError:
                    continue
                segs.append({"segmentId": f"seg{len(segs):04d}", "name": cols[0].strip(),
                             "lat": lat, "lon": lon})
                if len(segs) >= REFRESH_MAX_SEGMENTS:
                    break
    except Exception as e:
        print(f"[refresher] Không đọc được geometry.csv: {e}")
    return segs


MONITORED_SEGMENTS: List[Dict[str, Any]] = []
LIVE_DATA: Dict[str, Dict[str, Any]] = {}
REFRESH_STATE: Dict[str, Any] = {
    "enabled": REFRESH_ENABLED,
    "running": False,
    "cycles": 0,
    "lastCycleAt": None,
    "lastOk": 0,
    "lastFail": 0,
    "nextAt": None,
}
_tomtom_keys = _load_tomtom_keys()
_key_index = [0]


def _tomtom_fetch(lat: float, lon: float) -> Optional[Dict[str, float]]:
    if not _tomtom_keys:
        return None
    url = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?"
           + urllib.parse.urlencode({"point": f"{lat:.6f},{lon:.6f}", "unit": "KMPH",
                                     "key": _tomtom_keys[_key_index[0] % len(_tomtom_keys)]}))
    for _ in range(min(len(_tomtom_keys), 3)):
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                if r.status != 200:
                    _key_index[0] += 1
                    continue
                flow = json.load(r).get("flowSegmentData") or {}
                cur = float(flow.get("currentSpeed", 0) or 0)
                ff = float(flow.get("freeFlowSpeed", 0) or 0)
                if ff <= 0:
                    return None
                return {"currentSpeed": cur, "freeFlowSpeed": ff,
                        "congestionIndex": round(cur / ff, 3)}
        except Exception:
            _key_index[0] += 1
            url = url.replace(f"key={_tomtom_keys[(_key_index[0] - 1) % len(_tomtom_keys)]}",
                              f"key={_tomtom_keys[_key_index[0] % len(_tomtom_keys)]}")
    return None


def _refresh_cycle():
    ok = fail = 0
    for seg in MONITORED_SEGMENTS:
        if seg["lat"] is None:
            continue
        res = _tomtom_fetch(seg["lat"], seg["lon"])
        if res is None:
            fail += 1
            continue
        ok += 1
        LIVE_DATA[seg["segmentId"]] = {**seg, **res, "fetchedAt": datetime.now().isoformat(timespec="seconds")}
    REFRESH_STATE.update({
        "cycles": REFRESH_STATE["cycles"] + 1,
        "lastCycleAt": datetime.now().isoformat(timespec="seconds"),
        "lastOk": ok,
        "lastFail": fail,
        "nextAt": datetime.fromtimestamp(time.time() + REFRESH_INTERVAL).isoformat(timespec="seconds"),
    })
    print(f"[refresher] Chu kỳ #{REFRESH_STATE['cycles']}: OK={ok} FAIL={fail} "
          f"(tổng {len(LIVE_DATA)} đoạn có dữ liệu)")


def _refresher_loop():
    if not _tomtom_keys:
        print("[refresher] TẮT: không tìm thấy TOMTOM_KEYS (env hoặc local.properties)")
        REFRESH_STATE["enabled"] = False
        return
    print(f"[refresher] Bật: {len(MONITORED_SEGMENTS)} đoạn, chu kỳ {REFRESH_INTERVAL}s, {len(_tomtom_keys)} key")
    while True:
        try:
            _refresh_cycle()
        except Exception as e:
            print(f"[refresher] Lỗi chu kỳ: {e}")
        time.sleep(REFRESH_INTERVAL)


@app.on_event("startup")
def _start_refresher():
    global MONITORED_SEGMENTS
    MONITORED_SEGMENTS = _load_monitored_segments()
    if REFRESH_ENABLED and MONITORED_SEGMENTS:
        threading.Thread(target=_refresher_loop, daemon=True).start()


@app.get("/api/ml/live-status")
def api_live_status():
    """Trạng thái dữ liệu live server đang giám sát: chu kỳ, số đoạn OK/FAIL,
    và danh sách 10 đoạn có dữ liệu mới nhất."""
    latest = sorted(LIVE_DATA.values(), key=lambda s: s.get("fetchedAt", ""), reverse=True)[:10]
    return {
        "refresher": REFRESH_STATE,
        "monitoredSegments": len(MONITORED_SEGMENTS),
        "intervalSec": REFRESH_INTERVAL,
        "tomtomKeys": len(_tomtom_keys),
        "storedSegments": len(LIVE_DATA),
        "latest": latest,
    }


@app.post("/api/ml/predict")
def api_predict(req: PredictRequest):
    """Dự đoán giao thông cho một đoạn đường (ví dụ request/response trong báo cáo)."""
    return {"prediction": predict_segment(req.segment)}


@app.post("/api/ml/predict-batch")
def api_predict_batch(req: PredictBatchRequest):
    return {"predictions": [predict_segment(s) for s in req.segments]}


@app.post("/api/ml/recommend-route")
def api_recommend_route(req: RecommendRouteRequest):
    scores = sorted([route_score(r) for r in req.routes], key=lambda x: x["predictedDurationSec"])
    best = scores[0]["id"]
    out = []
    for s in scores:
        item = dict(s)
        item["recommended"] = s["id"] == best
        out.append(item)
    return {"recommendedId": best, "scores": out}


@app.post("/api/ml/compare-routes")
def api_compare_routes(req: CompareRoutesRequest):
    a, b = route_score(req.routeA), route_score(req.routeB)
    if a["predictedDurationSec"] <= b["predictedDurationSec"]:
        winner, other = "A", "B"
        delta = b["predictedDurationSec"] - a["predictedDurationSec"]
    else:
        winner, other = "B", "A"
        delta = a["predictedDurationSec"] - b["predictedDurationSec"]
    return {
        "winner": winner,
        "deltaSeconds": round(delta),
        "routeA": a,
        "routeB": b,
        "reason": f"Tuyến {winner} dự kiến nhanh hơn ~{round(delta / 60)} phút theo mô hình ML",
    }


@app.post("/api/ml/detect-incident")
def api_detect_incident(req: PredictRequest):
    p = predict_segment(req.segment)
    inc = p["incident"]
    return {"incident": inc["incident"], "risk": inc["risk"], "confidence": inc["confidence"],
            "reason": inc["reason"], "prediction": p}


@app.post("/api/ml/detect-incidents")
def api_detect_incidents(req: PredictBatchRequest):
    out = []
    for s in req.segments:
        p = predict_segment(s)
        out.append({"incident": p["incident"]["incident"], "risk": p["incident"]["risk"],
                    "confidence": p["incident"]["confidence"], "prediction": p})
    return {"results": out}


@app.post("/api/ml/traffic-status")
def api_traffic_status(req: TrafficStatusRequest):
    if not req.segments:
        return {"summary": {"level": "n/a", "avgCongestionIndex": None, "counts": {}}}
    preds = [predict_segment(s) for s in req.segments]
    counts = {"High": 0, "Moderate": 0, "Low": 0}
    for p in preds:
        counts[p["trafficLevel"]] += 1
    n = len(preds)
    avg_ratio = sum(p["congestionIndex"] for p in preds) / n
    overall = "Kẹt nặng" if avg_ratio < 0.70 else ("Trung bình" if avg_ratio < 0.85 else "Thông thoáng")
    return {
        "summary": {
            "level": overall,
            "avgCongestionIndex": round(avg_ratio, 3),
            "counts": counts,
            "highPct": round(100.0 * counts["High"] / n, 1),
            "total": n,
        },
        "predictions": preds,
    }
