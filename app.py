"""
Seller Copilot - FastAPI Web Application & Backend API
=====================================================
Serves the dark mode HTML/CSS/JS frontend dashboard and REST API endpoints.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from backend.inference import SellerCopilotEngine

app = FastAPI(title="Seller Copilot - AI Insight Toko Game")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Initialize inference engine
engine = SellerCopilotEngine()

# API Endpoints
@app.get("/api/search")
def search(
    type: str = Query("game", description="Search type: 'game' or 'user'"),
    query: str = Query(..., description="Game name or User ID")
):
    query_clean = query.strip()
    if not query_clean:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    if type == "user":
        # Parse user_id
        try:
            user_id = int(query_clean)
        except ValueError:
            # Fallback if non-numeric string passed
            user_id = abs(hash(query_clean)) % 4000 + 1000
        return engine.get_user_insight(user_id)
    else:
        # Default game search
        return engine.get_game_insight(query_clean)

@app.get("/api/autocomplete")
def autocomplete(
    type: str = Query("game", description="'game' or 'user'"),
    q: str = Query("", description="Query prefix")
):
    if type == "user":
        results = engine.autocomplete_users(q, limit=8)
    else:
        results = engine.autocomplete_games(q, limit=8)
    return {"results": results}

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Seller Copilot Backend API Running. Static index.html not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
