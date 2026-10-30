import os
import json
from fastapi import FastAPI, Query

app = FastAPI()

# Get the directory where api/index.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../dashboard_data.json")
# Alternatively, if dashboard_data.json is inside api/:
# DATA_PATH = os.path.join(BASE_DIR, "dashboard_data.json")

def load_data():
    # Fallback path checks
    paths = [
        DATA_PATH,
        os.path.join(os.getcwd(), "dashboard_data.json"),
        os.path.join(os.getcwd(), "api/dashboard_data.json")
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    return {}

@app.get("/api/simulate")
def simulate(
    stt_bps: float = Query(10.0),
    brokerage_bps: float = Query(2.0),
    slippage_bps: float = Query(5.0)
):
    data = load_data()
    return data
