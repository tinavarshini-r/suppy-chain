from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import uuid
import math
import random
import threading

# --- ML model (scikit-learn) ---
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI(title="Supply Chain Logistics Prototype",
              description="Demo backend with demand forecasting and simple route optimization",
              version="0.1")

# ---------- In-memory DB & cache (demo) ----------
JOBS: Dict[str, Dict] = {}        # job_id -> job details
SHIPMENTS: Dict[str, Dict] = {}   # shipment_id -> tracking info (simulate real-time)
CACHE: Dict[str, Dict] = {}       # simple cache placeholder (simulate Redis)

# ---------- Data models ----------
class JobCreate(BaseModel):
    title: str
    origin: List[float] = Field(..., description="Lat, Lon")
    destination: List[float] = Field(..., description="Lat, Lon")
    weight_kg: float
    volume_m3: Optional[float] = None
    pickup_time: Optional[datetime] = None
    delivery_deadline: Optional[datetime] = None
    vehicle_required: Optional[str] = None

class Job(BaseModel):
    job_id: str
    title: str
    origin: List[float]
    destination: List[float]
    weight_kg: float
    volume_m3: Optional[float]
    pickup_time: Optional[datetime]
    delivery_deadline: Optional[datetime]
    created_at: datetime

class ForecastRequest(BaseModel):
    sku_id: str
    history_days: int = 30

class ForecastResponse(BaseModel):
    sku_id: str
    predicted_next_7_days: List[float]
    model_info: str

class RouteRequest(BaseModel):
    depot: List[float] = Field(..., description="Lat, Lon of depot/start")
    vehicles: List[Dict] = Field(..., description="Each vehicle: {'id':str,'capacity_kg':float,'start_time':datetime}")
    jobs: List[str] = Field(..., description="List of job_ids to serve")

class RouteStop(BaseModel):
    job_id: str
    eta: datetime
    location: List[float]

class RoutePlan(BaseModel):
    vehicle_id: str
    stops: List[RouteStop]
    total_distance_km: float

# ---------- Synthetic demand-forecasting model ----------
# We'll build a small LinearRegression per SKU using synthetic historic features.
SKU_MODELS: Dict[str, LinearRegression] = {}

def generate_synthetic_time_series(sku_id: str, days: int):
    # Simple seasonal + trend + noise synthetic data
    rng = np.random.default_rng(abs(hash(sku_id)) % (2**32))
    base = 50 + (hash(sku_id) % 20)  # stable per-sku bias
    trend = np.linspace(0, 5, days)
    season = 10 * np.sin(np.arange(days) * 2 * math.pi / 7)  # weekly seasonality
    noise = rng.normal(0, 3, size=days)
    series = base + trend + season + noise
    series = np.clip(series, 0, None)
    return series

def train_or_get_model(sku_id: str, history_days: int = 30):
    # For demo: train linear regression on lag features (sliding window)
    key = f"{sku_id}:{history_days}"
    if key in CACHE:
        return CACHE[key]['model']
    series = generate_synthetic_time_series(sku_id, history_days)
    # features: previous 7 days to predict day t
    window = 7
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(series[i])
    if len(X) == 0:
        # fallback: trivial constant model
        lr = LinearRegression()
        lr.coef_ = np.zeros(window)
        lr.intercept_ = float(np.mean(series)) if len(series) > 0 else 0.0
        CACHE[key] = {'model': lr}
        return lr
    X = np.array(X)
    y = np.array(y)
    lr = LinearRegression()
    lr.fit(X, y)
    CACHE[key] = {'model': lr}
    return lr

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Predict next 7 days demand for given SKU using a small scikit-learn linear regression trained on synthetic data.
    """
    model = train_or_get_model(req.sku_id, req.history_days)
    hist = generate_synthetic_time_series(req.sku_id, req.history_days)
    # take last 7 days as features
    window = 7
    if len(hist) < window:
        last = np.pad(hist, (window - len(hist), 0), 'constant', constant_values=0)
    else:
        last = hist[-window:]
    preds = []
    cur_input = last.copy()
    for _ in range(7):
        pred = float(model.predict(cur_input.reshape(1, -1))[0])
        preds.append(max(0.0, round(pred, 2)))
        # slide
        cur_input = np.roll(cur_input, -1)
        cur_input[-1] = pred
    return ForecastResponse(
        sku_id=req.sku_id,
        predicted_next_7_days=preds,
        model_info=f"LinearRegression (trained on {req.history_days} synthetic days)"
    )

# ---------- Job & shipment endpoints ----------
@app.post("/jobs", response_model=Job)
def create_job(j: JobCreate):
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    job = {
        "job_id": job_id,
        "title": j.title,
        "origin": j.origin,
        "destination": j.destination,
        "weight_kg": j.weight_kg,
        "volume_m3": j.volume_m3,
        "pickup_time": j.pickup_time,
        "delivery_deadline": j.delivery_deadline,
        "created_at": now
    }
    JOBS[job_id] = job
    # create a shipment record for tracking demo
    shipment_id = str(uuid.uuid4())
    SHIPMENTS[shipment_id] = {
        "shipment_id": shipment_id,
        "job_id": job_id,
        "location": j.origin,
        "status": "created",
        "last_updated": datetime.utcnow()
    }
    job["shipment_id"] = shipment_id
    return Job(**job)

@app.get("/jobs", response_model=List[Job])
def list_jobs():
    return [Job(**v) for v in JOBS.values()]

# ---------- Simple route optimizer ----------
def haversine_km(a, b):
    # a, b = [lat, lon]
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371.0
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(x))

@app.post("/optimize_route", response_model=List[RoutePlan])
def optimize_route(req: RouteRequest):
    """
    Greedy capacity-aware route optimizer:
    - Assigns jobs to vehicles by nearest-next-job subject to capacity.
    - Respects (soft) delivery_deadline by ordering early deadlines first per vehicle.
    - Returns ETA computed by simple average speed assumption.
    Note: This is a demo heuristic (max 2 minutes processing expected).
    """
    # validate jobs
    jobs_to_assign = []
    for jid in req.jobs:
        if jid not in JOBS:
            raise HTTPException(status_code=404, detail=f"job {jid} not found")
        jobs_to_assign.append(JOBS[jid])
    # vehicle structures
    vehicles = []
    for v in req.vehicles:
        vehicles.append({
            "id": v.get("id", str(uuid.uuid4())),
            "capacity_kg": float(v.get("capacity_kg", 1000.0)),
            "start_time": v.get("start_time", datetime.utcnow()),
            "remaining": float(v.get("capacity_kg", 1000.0)),
            "route": [],
            "pos": req.depot[:]  # current location
        })
    # simple assignment: sort jobs by weight desc (heavy jobs first)
    jobs_sorted = sorted(jobs_to_assign, key=lambda x: -x['weight_kg'])
    UNASSIGNED = []
    for job in jobs_sorted:
        # find nearest vehicle that can take weight
        best_v = None
        best_dist = None
        for v in vehicles:
            if v["remaining"] + 1e-9 >= job['weight_kg']:
                d = haversine_km(v["pos"], job['origin'])
                if best_v is None or d < best_dist:
                    best_v = v
                    best_dist = d
        if best_v is None:
            UNASSIGNED.append(job)
        else:
            best_v["route"].append(job)
            best_v["remaining"] -= job['weight_kg']
            best_v["pos"] = job['origin']  # move to pickup
    # build RoutePlan outputs: compute ETA assuming average speed 40 km/h
    AVG_SPEED_KMPH = 40.0
    plans = []
    for v in vehicles:
        current_time = v["start_time"] if isinstance(v["start_time"], datetime) else datetime.utcnow()
        cur_loc = req.depot[:]
        total_distance = 0.0
        stops = []
        # order route by earliest delivery_deadline first to respect deadlines
        route_jobs = sorted(v["route"], key=lambda x: x.get('delivery_deadline') or datetime.max)
        for job in route_jobs:
            # drive to pickup
            d1 = haversine_km(cur_loc, job['origin'])
            travel_minutes = (d1 / AVG_SPEED_KMPH) * 60
            current_time = current_time + timedelta(minutes=travel_minutes)
            total_distance += d1
            # pickup -> delivery
            d2 = haversine_km(job['origin'], job['destination'])
            travel_minutes2 = (d2 / AVG_SPEED_KMPH) * 60
            current_time = current_time + timedelta(minutes=travel_minutes2)
            total_distance += d2
            stop = RouteStop(
                job_id=job['job_id'],
                eta=current_time,
                location=job['destination']
            )
            stops.append(stop)
            cur_loc = job['destination']
        plans.append(RoutePlan(
            vehicle_id=v['id'],
            stops=stops,
            total_distance_km=round(total_distance, 3)
        ))
    if UNASSIGNED:
        # include a warning in an exception with unassigned job ids (caller can still see assigned plans)
        raise HTTPException(status_code=400, detail=f"Some jobs could not be assigned due to capacity limits: {[j['job_id'] for j in UNASSIGNED]}")
    return plans

# ---------- Shipment tracking endpoints ----------
@app.get("/track/{shipment_id}")
def track(shipment_id: str):
    if shipment_id not in SHIPMENTS:
        raise HTTPException(status_code=404, detail="shipment not found")
    s = SHIPMENTS[shipment_id]
    # return last known
    return {
        "shipment_id": s["shipment_id"],
        "job_id": s["job_id"],
        "location": s["location"],
        "status": s["status"],
        "last_updated": s["last_updated"]
    }

# ---------- Health and utilities ----------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow()}

# ---------- Simulated background updater for shipments (updates every ~30 minutes in real app) ----------
def shipment_simulator():
    """Periodically move shipments a little towards their destination (demo)."""
    while True:
        # sleep 180 seconds to simulate updates during demo (not 30 minutes, to be demo friendly)
        import time
        time.sleep(180)  # adjust as needed; in production real updates come from telematics or APIs
        for s in list(SHIPMENTS.values()):
            job = JOBS.get(s['job_id'])
            if not job:
                continue
            # move 10% closer to destination
            lat_cur, lon_cur = s['location']
            lat_dst, lon_dst = job['destination']
            new_lat = lat_cur + 0.1 * (lat_dst - lat_cur)
            new_lon = lon_cur + 0.1 * (lon_dst - lon_cur)
            s['location'] = [new_lat, new_lon]
            s['status'] = 'in_transit'
            s['last_updated'] = datetime.utcnow()

# start background thread
t = threading.Thread(target=shipment_simulator, daemon=True)
t.start()




