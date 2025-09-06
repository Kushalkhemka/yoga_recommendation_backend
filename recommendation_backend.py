from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import logging
import os
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Yoga Recommendation API",
    description="AI-powered yoga asana recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and data
try:
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Model loaded successfully")
    
    logger.info("Loading yoga embeddings data...")
    with open("yoga_embeddings.pkl", "rb") as f:
        df = pickle.load(f)
    logger.info(f"Data loaded successfully with {len(df)} yoga poses")
except Exception as e:
    logger.error(f"Failed to initialize model or data: {str(e)}")
    raise e

class UserInput(BaseModel):
    age: int
    height: int
    weight: int
    goals: List[str]
    physical_issues: List[str]
    mental_issues: List[str]
    level: str

@app.get("/")
async def root():
    return {"message": "Yoga Recommendation API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": df is not None and len(df) > 0,
        "total_poses": len(df) if df is not None else 0
    }

def recommend_asanas(user_profile):
    try:
        logger.info(f"Generating recommendations for user profile: {user_profile}")
        
        # Validate input
        if not user_profile.get("goals") or not user_profile.get("physical_issues") or not user_profile.get("mental_issues"):
            raise ValueError("Missing required fields: goals, physical_issues, or mental_issues")
        
        user_emb = {
            "goals": model.encode(" ".join(user_profile["goals"]), normalize_embeddings=True),
            "physical_issues": model.encode(" ".join(user_profile["physical_issues"]), normalize_embeddings=True),
            "mental_issues": model.encode(" ".join(user_profile["mental_issues"]), normalize_embeddings=True),
        }

        recommendations = []
        weights = {
            "goals_benefits": 4,
            "physical_benefits": 4,
            "mental_benefits": 4,
            "physical_match": 2,
            "mental_match": 2,
        }
        total_weight = sum(weights.values())

        for _, row in df.iterrows():
            try:
                score = 0.0
                contra_text = str(row["Contraindications"]).lower()

                discard = False
                for issue in user_profile["physical_issues"] + user_profile["mental_issues"]:
                    issue = issue.lower()
                    if issue in contra_text:
                        discard = True
                        break
                    if util.cos_sim(model.encode(issue, normalize_embeddings=True), row["Contraindications_emb"]).item() > 0.25:
                        discard = True
                        break
                if discard:
                    continue

                score += weights["goals_benefits"] * util.cos_sim(user_emb["goals"], row["Benefits_emb"]).item()
                score += weights["physical_benefits"] * util.cos_sim(user_emb["physical_issues"], row["Benefits_emb"]).item()
                score += weights["mental_benefits"] * util.cos_sim(user_emb["mental_issues"], row["Benefits_emb"]).item()

                score += weights["physical_match"] * util.cos_sim(user_emb["physical_issues"], row["Targeted Physical Problems_emb"]).item()
                score += weights["mental_match"] * util.cos_sim(user_emb["mental_issues"], row["Targeted Mental Problems_emb"]).item()

                score /= total_weight

                if score > 0:
                    recommendations.append({
                        "name": row["AName"],
                        "score": round(score, 3),
                        "benefits": row["Benefits"],
                        "contraindications": row["Contraindications"]
                    })
            except Exception as e:
                logger.warning(f"Error processing row {row.get('AName', 'unknown')}: {str(e)}")
                continue

        recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations[:10]
        
    except Exception as e:
        logger.error(f"Error in recommend_asanas: {str(e)}")
        raise e

@app.post("/recommend/")
async def get_recommendations(user_input: UserInput):
    try:
        logger.info("Received recommendation request")
        user_profile = user_input.dict()
        recommended_asanas = recommend_asanas(user_profile)
        
        if not recommended_asanas:
            logger.warning("No recommendations generated")
            return {
                "recommended_asanas": [],
                "message": "No suitable yoga poses found for your profile. Please try adjusting your goals or issues.",
                "status": "success"
            }
        
        return {
            "recommended_asanas": recommended_asanas,
            "total_recommendations": len(recommended_asanas),
            "status": "success"
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while generating recommendations")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
