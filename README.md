# Yoga Recommendation API

An AI-powered yoga asana recommendation system built with FastAPI and sentence transformers.

## Features

- AI-powered yoga pose recommendations based on user profile
- Comprehensive error handling and logging
- Health check endpoints
- CORS enabled for web access
- Ready for deployment on Render

## API Endpoints

- `GET /` - Root endpoint with basic status
- `GET /health` - Health check with system status
- `POST /recommend/` - Get yoga recommendations

## Request Format

```json
{
  "age": 25,
  "height": 170,
  "weight": 70,
  "goals": ["flexibility", "stress relief"],
  "physical_issues": ["back pain"],
  "mental_issues": ["anxiety"],
  "level": "beginner"
}
```

## Response Format

```json
{
  "recommended_asanas": [
    {
      "name": "Pose Name",
      "score": 0.85,
      "benefits": "Benefits description",
      "contraindications": "Contraindications"
    }
  ],
  "total_recommendations": 10,
  "status": "success"
}
```

## Deployment on Render

1. Push your code to a GitHub repository
2. Connect your GitHub account to Render
3. Create a new Web Service
4. Select your repository
5. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python recommendation_backend.py`
   - **Python Version**: 3.11.0

The `render.yaml` file is included for automatic configuration.

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python recommendation_backend.py
   ```

3. Access the API at `http://localhost:8000`

## Requirements

- Python 3.11+
- FastAPI
- Sentence Transformers
- PyTorch
- Pandas
- NumPy

Make sure you have the `yoga_embeddings.pkl` file in the same directory as the application.
