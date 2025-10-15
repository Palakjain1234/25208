"""
api.py

FastAPI server for Rake Formation Optimization System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import os
import json
import asyncio
import uuid
from datetime import datetime
import joblib

from optim import optimize_rake_allocation
from multi_model import MultiModelPipeline

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
DATA_PATH = "./bokaro_to_cmo_customers.csv"
MODEL_DIR = "./saved_models"
OUTPUT_DIR = "./api_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------------
app = FastAPI(
    title="Rake Formation Optimization API",
    description="API for optimizing rake allocation and formation planning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------
class OptimizationRequest(BaseModel):
    plan_ids: Optional[List[str]] = None
    use_sample: bool = True
    custom_data: Optional[List[Dict]] = None

class OptimizationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    results_path: Optional[str] = None
    summary: Optional[Dict] = None
    timestamp: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    results: Optional[Dict] = None
    timestamp: str

class PlanDetailsResponse(BaseModel):
    plan_id: str
    optimized_mode: str
    q_rail_tons: float
    optimized_total_cost: float
    planned_qty_t: float
    customer_name: str
    destination: str
    pred_rail_cost_total: float
    pred_road_cost_total: float
    on_time_prob: float

# -------------------------------------------------------------------------
# Global State
# -------------------------------------------------------------------------
jobs = {}
pipeline = None

# -------------------------------------------------------------------------
# Startup Event
# -------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize the ML pipeline on startup"""
    global pipeline
    try:
        print("🔹 Initializing ML pipeline...")
        pipeline = MultiModelPipeline(DATA_PATH, MODEL_DIR)
        
        # Load feature columns
        feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
        if os.path.exists(feature_path):
            pipeline.feature_cols = joblib.load(feature_path)
            print(f"✅ Loaded {len(pipeline.feature_cols)} feature columns")
        else:
            print("❌ Feature columns not found")
            
        print("✅ ML pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ML pipeline: {e}")

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def load_and_preprocess_data(plan_ids=None):
    """Load and preprocess data for optimization"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Filter by plan_ids if provided
        if plan_ids and len(plan_ids) > 0:
            df = df[df['plan_id'].isin(plan_ids)]
            if len(df) == 0:
                raise ValueError("No matching plan_ids found")
        
        # Recreate engineered date features
        if "plan_date" in df.columns:
            try:
                df["plan_date"] = pd.to_datetime(df["plan_date"], dayfirst=True)
                df["plan_dayofweek"] = df["plan_date"].dt.dayofweek
                df["plan_month"] = df["plan_date"].dt.month
                df["plan_day"] = df["plan_date"].dt.day
            except Exception as e:
                print(f"⚠️ Date feature creation failed: {e}")
        
        return df
    except Exception as e:
        raise Exception(f"Data loading failed: {e}")

def run_optimization_job(job_id: str, plan_ids: Optional[List[str]] = None):
    """Run optimization in background"""
    try:
        jobs[job_id] = {
            "status": "running", 
            "progress": 0.2,
            "message": "Loading data...",
            "timestamp": datetime.now().isoformat()
        }
        
        # Load data
        df = load_and_preprocess_data(plan_ids)
        jobs[job_id]["progress"] = 0.4
        jobs[job_id]["message"] = "Running predictions..."
        
        # Run predictions
        preds = pipeline.predict_all(df)
        df_pred = df.copy()
        for k, v in preds.items():
            df_pred[k] = v
        
        jobs[job_id]["progress"] = 0.7
        jobs[job_id]["message"] = "Running optimization..."
        
        # Run optimization
        result_df = optimize_rake_allocation(df_pred)
        
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["message"] = "Optimization completed"
        
        # Save results
        output_path = os.path.join(OUTPUT_DIR, f"results_{job_id}.csv")
        result_df.to_csv(output_path, index=False)
        
        # Create summary
        total_orders = len(result_df)
        rail_orders = result_df["y_rail"].sum() if "y_rail" in result_df.columns else 0
        road_orders = total_orders - rail_orders
        rail_tonnage = result_df["q_rail_tons"].sum() if "q_rail_tons" in result_df.columns else 0
        total_tonnage = result_df["planned_qty_t"].sum()
        total_cost = result_df["optimized_total_cost"].sum() if "optimized_total_cost" in result_df.columns else 0
        
        summary = {
            "total_orders": total_orders,
            "rail_orders": int(rail_orders),
            "road_orders": int(road_orders),
            "rail_orders_percentage": round(rail_orders/total_orders*100, 1),
            "rail_tonnage": round(rail_tonnage, 0),
            "total_tonnage": round(total_tonnage, 0),
            "rail_tonnage_percentage": round(rail_tonnage/total_tonnage*100, 1) if total_tonnage > 0 else 0,
            "total_cost": round(total_cost, 2),
            "results_file": output_path
        }
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = summary
        jobs[job_id]["results_df"] = result_df
        
        print(f"✅ Optimization job {job_id} completed")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {str(e)}"
        print(f"❌ Optimization job {job_id} failed: {e}")

# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Rake Formation Optimization API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    pipeline_status = "ready" if pipeline else "not_ready"
    return {
        "status": "healthy",
        "pipeline": pipeline_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_rake_formation(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start rake formation optimization"""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())[:8]
        
        # Validate request
        if request.custom_data and len(request.custom_data) > 0:
            # TODO: Implement custom data handling
            raise HTTPException(status_code=501, detail="Custom data not yet supported")
        
        # Start background job
        background_tasks.add_task(run_optimization_job, job_id, request.plan_ids)
        
        # Store job info
        jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued for processing",
            "timestamp": datetime.now().isoformat()
        }
        
        return OptimizationResponse(
            job_id=job_id,
            status="queued",
            message="Optimization job started",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get optimization job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        results=job.get("results"),
        timestamp=job["timestamp"]
    )

@app.get("/results/{job_id}")
async def get_optimization_results(job_id: str, limit: int = 100, offset: int = 0):
    """Get detailed optimization results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    try:
        result_df = job.get("results_df")
        if result_df is None:
            # Try to load from file
            results_path = os.path.join(OUTPUT_DIR, f"results_{job_id}.csv")
            if not os.path.exists(results_path):
                raise HTTPException(status_code=404, detail="Results file not found")
            result_df = pd.read_csv(results_path)
        
        # Convert to JSON-friendly format
        total_records = len(result_df)
        paginated_df = result_df.iloc[offset:offset + limit]
        
        results = []
        for _, row in paginated_df.iterrows():
            result = {
                "plan_id": row.get("plan_id", ""),
                "optimized_mode": row.get("optimized_mode", "Road"),
                "q_rail_tons": float(row.get("q_rail_tons", 0)),
                "optimized_total_cost": float(row.get("optimized_total_cost", 0)),
                "planned_qty_t": float(row.get("planned_qty_t", 0)),
                "customer_name": row.get("customer_name", ""),
                "destination": row.get("destination", ""),
                "pred_rail_cost_total": float(row.get("pred_rail_cost_total", 0)),
                "pred_road_cost_total": float(row.get("pred_road_cost_total", 0)),
                "on_time_prob": float(row.get("on_time_prob", 0)),
                "priority_score": float(row.get("priority_score", 0)),
                "rake_available": int(row.get("rake_available", 0)),
                "on_time_label": int(row.get("on_time_label", 0))
            }
            results.append(result)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "pagination": {
                "total": total_records,
                "limit": limit,
                "offset": offset,
                "returned": len(results)
            },
            "results": results,
            "summary": job.get("results")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")

@app.get("/plans")
async def get_available_plans(limit: int = 50):
    """Get list of available plans for optimization"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        plans = []
        for _, row in df.head(limit).iterrows():
            plan = {
                "plan_id": row.get("plan_id", ""),
                "customer_name": row.get("customer_name", ""),
                "destination": row.get("destination", ""),
                "planned_qty_t": float(row.get("planned_qty_t", 0)),
                "priority_score": float(row.get("priority_score", 0))
            }
            plans.append(plan)
        
        return {
            "total_plans": len(df),
            "returned": len(plans),
            "plans": plans
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load plans: {str(e)}")

@app.get("/plan/{plan_id}")
async def get_plan_details(plan_id: str):
    """Get detailed information for a specific plan"""
    try:
        df = pd.read_csv(DATA_PATH)
        plan_data = df[df['plan_id'] == plan_id]
        
        if len(plan_data) == 0:
            raise HTTPException(status_code=404, detail="Plan not found")
        
        row = plan_data.iloc[0]
        
        return {
            "plan_id": plan_id,
            "customer_name": row.get("customer_name", ""),
            "destination": row.get("destination", ""),
            "planned_qty_t": float(row.get("planned_qty_t", 0)),
            "priority_score": float(row.get("priority_score", 0)),
            "min_rake_tonnage": float(row.get("min_rake_tonnage", 0)),
            "terminal_cost": float(row.get("terminal_cost", 0)),
            "expected_demurrage": float(row.get("expected_demurrage", 0))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load plan details: {str(e)}")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )