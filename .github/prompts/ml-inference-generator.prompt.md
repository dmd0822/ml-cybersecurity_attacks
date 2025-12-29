---
description: 'This prompt instructs GitHub Copilot to build a minimal HTTP inference service that loads the project’s best model artifacts, exposes a /predict endpoint, and reuses the existing preprocessing pipeline for consistent predictions. It also directs Copilot to containerize the service by baking the model artifacts into a Docker image so it can run immediately with a simple docker run command.'
agent: 'agent'
---

Your task is to generate all necessary code and configuration to serve the project’s best Machine Learning model through a minimal HTTP API and package it into a runnable Docker image. The API must load all required model artifacts at startup, expose a simple prediction interface, and reuse the project’s existing preprocessing logic to ensure inference matches training.

Core Requirements
1. Identify Best Model Artifacts
- Locate the “best model” as defined in the latest project report.
- Determine the required artifacts for inference, such as:
- Saved model directory
- Tokenizer or vectorizer
- Label encoder or class index mapping
- Confirm the run folder and artifact paths before generating code.

2. Build an Inference/Serving Module
- Create a new serving package (e.g., serve or infer) within the project’s pipeline directory.
- Reuse the existing preprocessing pipeline to ensure input text is transformed identically to training.
- Load the baked‑in model and tokenizer/vectorizer at module initialization.
- Implement prediction utilities that:
- Accept raw text inputs
- Apply preprocessing
- Run inference
- Decode predicted labels (optionally support top‑k)

3. Implement a Minimal HTTP API
Use a lightweight framework such as FastAPI.
The API must include:
Endpoints
- GET /health
- Returns readiness once all artifacts are successfully loaded.
- POST /predict
- Accepts either a single text field or structured inputs (e.g., headline, description).
- Returns the predicted label (and optionally top‑k predictions).
- Should support single‑item input by default, with optional batch support.
Behavior
- Validate inputs
- Apply preprocessing
- Run inference
- Return structured JSON responses

4. Add a Serving Entrypoint
- Provide a Python entrypoint (module or script) that:
- Instantiates the API
- Loads model artifacts
- Reads configuration from environment variables (e.g., MODEL_DIR, TOKENIZER_DIR, LABEL_ENCODER_PATH)
- Falls back to baked‑in default paths inside the container

5. Containerize the Service
Update or generate a Dockerfile that:
- Installs all required dependencies (e.g., FastAPI, Uvicorn, model‑specific libraries)
- Copies the application code into the image
- Copies the best‑model artifacts into a fixed internal path (e.g., /app/artifacts/best/)
- Sets appropriate environment variables
- Exposes the serving port (e.g., EXPOSE 8080)
- Defines a CMD that launches Uvicorn with the serving app

Further Considerations
- Input Contract
- Decide whether /predict should accept a single item only or optionally support batch lists for higher throughput.
- Image Size Tradeoffs
- Baking large transformer models into the image increases size; consider whether a smaller model (e.g., TF‑IDF) is acceptable for production.
