#!/usr/bin/env python3
"""
DigitalOcean deployment: FastAPI service for Aesthetic RAG Search.

2 APIs:
1) Get common concerns for a selected sub-zone
2) Search with concerns array + procedure type

Run locally:
  pip install -r requirements.txt
  uvicorn api_server:app --host 0.0.0.0 --port 8000

Env:
  DB_XLSX=database.xlsx
  EMB_CACHE=treatment_embeddings.pkl
  LOCAL_LLM_PROVIDER=ollama|transformers
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL=llama3.2:1b
"""


from __future__ import annotations
import difflib
import os
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_treatment_app import RAGTreatmentSearchApp


# ---------------------------- app init ----------------------------

APP_TITLE = "Aesthetic RAG Search API"

rag = RAGTreatmentSearchApp(
    excel_path=os.getenv("DB_XLSX", "database.xlsx"),
    embeddings_cache_path=os.getenv("EMB_CACHE", "treatment_embeddings.pkl"),
)

app = FastAPI(title=APP_TITLE, version="2.0.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------- hardcoded sub-zones ----------------------------

FACE_SUB_ZONES = [
    'hair', 'temples', 'forehead', 'eyebrows', 'eyes', 'nose', 'lips', 
    'chin', 'jawline', 'cheeks', 'ear', 'tearTrough', 'neck', 'face'
]

BODY_SUB_ZONES = [
    'abdomen', 'arms', 'breasts', 'chest', 'buttocks', 'feet', 
    'legs', 'pubis', 'waist', 'body'
]

ALL_SUB_ZONES = FACE_SUB_ZONES + BODY_SUB_ZONES
# Case-insensitive allowlist
ALLOWED_SUBZONES = {z.strip().lower(): z for z in ALL_SUB_ZONES}
ALLOWED_SUBZONES_LIST = sorted(ALLOWED_SUBZONES.keys())

def normalize_subzone_from_request(raw: str) -> str:
    """
    Accepts user input like:
      - tearTrough / teartrough / Tear Trough
    Returns the closest hardcoded token (lowercase) if it's close enough.
    """
    s = (raw or "").strip().lower()
    if not s:
        return ""

    # Exact match
    if s in ALLOWED_SUBZONES:
        return s

    # Fuzzy match to hardcoded list (general, not hardcoded aliases)
    best = difflib.get_close_matches(s, ALLOWED_SUBZONES_LIST, n=1, cutoff=0.72)
    return best[0] if best else ""


# ---------------------------- schemas ----------------------------

ProcedureType = Literal["surgical", "non_surgical", "both"]


class CommonConcernsRequest(BaseModel):
    sub_zone: str = Field(..., min_length=1, description="Selected sub-zone")


class CommonConcernsResponse(BaseModel):
    sub_zone: str
    common_concerns: List[str] = Field(description="Common concerns for this sub-zone from database")


class SearchRequest(BaseModel):
    sub_zone: str = Field(..., min_length=1, description="Selected sub-zone")
    concerns: List[str] = Field(..., min_items=1, description="User selected concerns array")
    procedure_type: ProcedureType = Field(..., description="Procedure type: surgical, non_surgical, or both")
    retrieval_k: int = Field(12, ge=3, le=50, description="Number of candidates to retrieve")
    final_k: int = Field(5, ge=1, le=10, description="Number of final recommendations")


class SearchResponse(BaseModel):
    mismatch: bool
    notice: str = ""
    recommended_procedures: List[str] = Field(default_factory=list, description="List of procedure names")
    suggested_region_subzones: List[dict] = Field(default_factory=list)


# ---------------------------- endpoints ----------------------------

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "aesthetic-rag-api", "version": "2.0.0"}


@app.post("/common_concerns", response_model=CommonConcernsResponse)
def get_common_concerns(req: CommonConcernsRequest):
    """
    API 1: Get common concerns for a selected sub-zone from database.
    
    Args:
        sub_zone: Selected sub-zone (e.g., 'eyes', 'nose', 'cheeks')
    
    Returns:
        sub_zone: The requested sub-zone
        common_concerns: List of common concerns from database for this sub-zone
    """
    #sub_zone = req.sub_zone.strip().lower()
    sub_zone = normalize_subzone_from_request(req.sub_zone)

    if not (req.sub_zone or "").strip():
        raise HTTPException(status_code=400, detail="sub_zone is required")

    if not sub_zone:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sub_zone. Must be one of: {', '.join(ALLOWED_SUBZONES_LIST)}"
        )

    
    # Validate sub-zone
    # if sub_zone not in ALLOWED_SUBZONES:
    #     raise HTTPException(
    #         status_code=400, 
    #         detail=f"Invalid sub_zone. Must be one of: {', '.join(ALLOWED_SUBZONES_LIST)}"
    #     )
    
    # Get common concerns for this sub-zone from database
    concerns = rag.get_concerns_for_subzone(sub_zone)
    
    if not concerns:
        raise HTTPException(status_code=404, detail=f"No concerns found for sub-zone: {sub_zone}")
    
    return {
        "sub_zone": sub_zone,
        "common_concerns": concerns
    }


@app.post("/search", response_model=SearchResponse)
def search_procedures(req: SearchRequest):
    """
    API 2: Search for treatment procedures.
    
    User selects:
    - concerns array (from API 1 response)
    - procedure_type (surgical, non_surgical, or both)
    
    Args:
        sub_zone: Selected sub-zone
        concerns: Array of selected concerns (from API 1)
        procedure_type: surgical, non_surgical, or both
        retrieval_k: Number of candidates (optional)
        final_k: Number of final results (optional)
    
    Returns:
        mismatch: Whether mismatch was detected
        notice: Message if mismatch
        recommended_procedures: List of procedure names (empty if mismatch)
        suggested_region_subzones: Suggestions if mismatch
    """
    sub_zone = normalize_subzone_from_request(req.sub_zone)

    concerns = [c.strip() for c in req.concerns if c.strip()]
    
    if not (req.sub_zone or "").strip():
        raise HTTPException(status_code=400, detail="sub_zone is required")

    if not sub_zone:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sub_zone. Must be one of: {', '.join(ALLOWED_SUBZONES_LIST)}"
        )

    
    if not concerns:
        raise HTTPException(status_code=400, detail="At least one concern is required")
    
    # Validate sub-zone
    # if sub_zone not in ALLOWED_SUBZONES:
    #     raise HTTPException(
    #         status_code=400, 
    #         detail=f"Invalid sub_zone. Must be one of: {', '.join(ALLOWED_SUBZONES_LIST)}"
    #     )
    
    # Get region from sub-zone
    region = rag.get_region_from_subzone(sub_zone)
    
    if not region:
        raise HTTPException(status_code=404, detail=f"No region found for sub-zone: {sub_zone}")
    
    # Map procedure_type to treatment preference format
    type_mapping = {
        "surgical": "Surgical",
        "non_surgical": "Non-Surgical",
        "both": "Both"
    }
    treatment_preference = type_mapping.get(req.procedure_type, "Both")
    
    # Perform search
    result = rag.search_by_concerns(
        region=region,
        sub_zone=sub_zone,
        type_choice=treatment_preference,
        concerns=concerns,
        retrieval_k=req.retrieval_k,
        final_k=req.final_k,
    )
    
    return result
