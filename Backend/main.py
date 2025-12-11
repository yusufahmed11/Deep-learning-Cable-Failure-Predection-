from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import io
import contextlib
from model_utils import load_resources, predict_row

# Initialize App
app = FastAPI(title="Cable Failure Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model on Startup
@app.on_event("startup")
async def startup_event():
    load_resources()

@app.get("/")
async def root():
    return {"status": "API running", "docs": "/docs"}

# Data Models
class SinglePredictionInput(BaseModel):
    # Accepting frontend keys
    age: float
    partialDischarge: float
    neutralCorrosion: float
    loading: float
    visualCondition: str
    # Optional ID
    id: Optional[str] = "SINGLE"

class PredictionResponse(BaseModel):
    id: str
    age: float
    partialDischarge: float
    neutralCorrosion: float
    loading: float
    visualCondition: str
    predictedHealthIndex: int
    riskLevel: str
    explanation: List[str]

# Column Mapping Logic
COLUMN_MAP = {
    "age": ["age", "years", "yr", "cable_age"],
    "partial discharge": ["pd", "partial_discharge", "pd_value", "discharge", "partialdischarge"],
    "neutral corrosion": ["corrosion", "neutral_corrosion", "cor", "cor_lvl", "neutralcorrosion"],
    "loading": ["load", "loading", "ld", "load_current", "current_load"],
    "visual condition": ["visual", "visual_condition", "condition", "cond", "visualcondition"],
    "id": ["id", "cable_id", "number", "index"]
}

def map_columns(df: pd.DataFrame):
    # Normalize columns to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    new_cols = {}
    missing = []
    
    # Target columns needed for model
    targets = ["age", "partial discharge", "neutral corrosion", "loading", "visual condition"]
    
    for target in targets:
        found = False
        # Check aliases
        aliases = COLUMN_MAP.get(target, [])
        aliases.append(target) # Add self
        
        for alias in aliases:
            if alias in df.columns:
                new_cols[alias] = target.title() # Map back to Title Case for internal consistency
                found = True
                break
        
        if not found:
            missing.append(target)
            
    # Also map ID if present
    for alias in COLUMN_MAP["id"]:
        if alias in df.columns:
            new_cols[alias] = "ID"
            break
            
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
        
    return df.rename(columns=new_cols)

@app.post("/predict_single", response_model=PredictionResponse)
async def predict_single(input_data: SinglePredictionInput):
    # Map input to model expected keys
    row = {
        "Age": input_data.age,
        "Partial Discharge": input_data.partialDischarge,
        "Neutral Corrosion": input_data.neutralCorrosion,
        "Loading": input_data.loading,
        "Visual Condition": input_data.visualCondition
    }
    
    result = predict_row(row)
    
    return {
        "id": input_data.id,
        "age": input_data.age,
        "partialDischarge": input_data.partialDischarge,
        "neutralCorrosion": input_data.neutralCorrosion,
        "loading": input_data.loading,
        "visualCondition": input_data.visualCondition,
        "predictedHealthIndex": result["health_index"],
        "riskLevel": result["risk_level"],
        "explanation": result["explanation"]
    }

@app.post("/predict_bulk")
async def predict_bulk(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Use CSV or Excel.")
            
        # Map Columns
        try:
            df = map_columns(df)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        # Process
        results = []
        for _, row in df.iterrows():
            # Extract features
            row_dict = {
                "Age": row.get("Age"),
                "Partial Discharge": row.get("Partial Discharge"),
                "Neutral Corrosion": row.get("Neutral Corrosion"),
                "Loading": row.get("Loading"),
                "Visual Condition": row.get("Visual Condition")
            }
            
            pred = predict_row(row_dict)
            
            results.append({
                "id": row.get("ID", f"Row-{_}"),
                "age": row.get("Age"),
                "partialDischarge": row.get("Partial Discharge"),
                "neutralCorrosion": row.get("Neutral Corrosion"),
                "loading": row.get("Loading"),
                "visualCondition": row.get("Visual Condition"),
                "predictedHealthIndex": pred["health_index"],
                "riskLevel": pred["risk_level"],
                "explanation": pred["explanation"]
            })
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
