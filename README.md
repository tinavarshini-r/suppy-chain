
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import asyncio
import redis
import psycopg2
import joblib
import datetime

app = FastAPI(title="AI Supply Chain Optimization")


r = redis.Redis(host='localhost', port=6379, db=0)


conn = psycopg2.connect(
    dbname="supply_chain", user="admin", password="password", host="localhost", port="5432"
)


class DemandRequest(BaseModel):
    product_id: str
    region: str
    history: list  # historical demand data

class RouteRequest(BaseModel):
    origin: str
    destinations: list
    vehicle_capacity: int
    time_windows: list


try:
    demand_model = joblib.load("models/demand_forecast.pkl")
except:
    demand_model = None

@app.post("/forecast_demand")
async def forecast_demand(req: DemandRequest):
    # Example with scikit-learn model
    if not demand_model:
        return {"error": "Model not available"}
    X = [[len(req.history), sum(req.history)]]
    prediction = demand_model.predict(X)
    return {"product_id": req.product_id, "forecast": float(prediction[0])}

@app.post("/optimize_route")
async def optimize_route(req: RouteRequest):
    # Simplified dummy route optimization
    # Normally you'd run a VRP solver with TensorFlow or heuristic search
    route = sorted(req.destinations)[:req.vehicle_capacity]
    await asyncio.sleep(1)  # Simulating computation (<2 min constraint)
    return {"origin": req.origin, "optimized_route": route}

@app.get("/track/{shipment_id}")
async def track_shipment(shipment_id: str):
    cached = r.get(shipment_id)
    if cached:
        return {"shipment_id": shipment_id, "status": cached.decode()}
    # Simulate API call to logistics provider
    status = "In Transit - {}".format(datetime.datetime.now())
    r.setex(shipment_id, 1800, status)  # cache for 30 min
    return {"shipment_id": shipment_id, "status": status}



import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export default function App() {
  const [shipments, setShipments] = useState([]);

  useEffect(() => {
    fetch('/track/SHIP123').then(res => res.json()).then(data => setShipments([data]));
  }, []);

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">Supply Chain Dashboard</h1>
      <MapContainer center={[20.5937, 78.9629]} zoom={5} style={{ height: "500px", width: "100%" }}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {shipments.map((s, i) => (
          <Marker key={i} position={[28.6139, 77.2090]}>
            <Popup>{s.shipment_id} - {s.status}</Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}



import joblib
from sklearn.linear_model import LinearRegression


X = [[10, 100], [20, 200], [30, 300]]  # features: [history_length, sum(history)]
y = [110, 210, 310]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "models/demand_forecast.pkl")
print("Model trained and saved.")
