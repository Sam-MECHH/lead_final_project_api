import io
import json
import os
import uuid
from datetime import datetime

import boto3
import mlflow.pytorch
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from PIL import Image
from transformers import AutoModel, AutoTokenizer

app = FastAPI(title="BioVil Cross-Attention+MLP Inference API")

# Global instances
device = None
tokenizer = None
text_model = None
image_model = None
image_transform = None
cross_att_classifier = None
s3_client = None
S3_BUCKET_NAME = os.environ.get("S3_MONITORING_BUCKET", "smec-lead-fp-mlops-monitoring")
REGISTERED_MODEL_NAME = "biovil_t_lead_demo"
MODEL_ALIAS = "prod"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"


# Startup component
@app.on_event("startup")
def load_all_models_and_assets():
    global \
        device, \
        tokenizer, \
        text_model, \
        image_model, \
        image_transform, \
        cross_att_classifier, \
        s3_client
    try:
        # Setup device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize S3 Client
        s3_client = boto3.client("s3")

        # Load the comprehensive BioViL-T repo for Text
        model_id = "microsoft/BiomedVLP-BioViL-T"

        # Specialized CXR-BERT tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        text_model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(
            device
        )

        # Instantiate the BioViL-T Image Engine
        image_model = get_biovil_t_image_encoder().to(device)
        image_transform = create_chest_xray_transform_for_inference(
            resize=512, center_crop_size=448
        )

        # Connect to Hugging Face MLflow instance and pull the Cross-Attention Classifier
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_URI"))
        cross_att_classifier = mlflow.pytorch.load_model(
            MODEL_URI, map_location=torch.device("cpu")
        )
        cross_att_classifier.to(device).eval()

        print("Models and processors loaded successfully!")
    except Exception as e:
        print(f"❌ Startup Error: {e!s}")
        raise e


# Background task for S3 logging
def save_production_data_to_s3(
    request_id: str,
    raw_text: str,
    image_bytes: bytes,
    prediction: int,
    probability: float,
):
    try:
        # 1. Save raw Image file to S3
        image_s3_key = f"production_data/images/{request_id}.jpg"
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=image_s3_key,
            Body=image_bytes,
            ContentType="image/jpg",
        )

        # 2. Save Metadata & Text to S3 as JSON (for Evidently AI reading)
        metadata = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "raw_text": raw_text,
            "image_s3_uri": f"s3://{S3_BUCKET_NAME}/{image_s3_key}",
            "prediction": prediction,
            "probability": probability,
        }

        metadata_s3_key = f"production_data/reports_metadata/{request_id}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=metadata_s3_key,
            Body=json.dumps(metadata),
            ContentType="application/json",
        )
        print(f"Logged request {request_id} to S3 successfully.")
    except Exception as e:
        print(f" Failed to log to S3 for request {request_id}: {e!s}")


# Preprocessing pipelines
def get_text_embeddings(report_text):
    inputs = tokenizer(
        report_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True,
        )
    return outputs.last_hidden_state


def get_image_embeddings_from_pil(pil_image):
    # Adjusted to accept the PIL image object directly from memory
    raw_image = pil_image.convert("L")
    processed_tensor = image_transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_outputs = image_model(processed_tensor)
    return image_outputs.projected_patch_embeddings


# Predict endpoint
@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    text_input: str = Form(...),
    image_file: UploadFile = File(...),
):
    # Guard against queries hitting the server before models are fully loaded
    if None in (cross_att_classifier, text_model, image_model):
        raise HTTPException(
            status_code=503, detail="Models are initializing. Try again shortly."
        )

    try:
        # Read incoming file stream directly into memory as a PIL Image
        image_bytes = await image_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Use custom preprocessing pipelines
        sequence_outputs = get_text_embeddings(text_input)
        patch_img_emb = get_image_embeddings_from_pil(pil_image)

        # Run inputs through registered Cross-Attention Classifier
        with torch.no_grad():
            outputs = cross_att_classifier(
                patch_img_emb, sequence_outputs[:, :256, :]
            ).squeeze(1)
            probability = torch.sigmoid(outputs).item()
            prediction = int(probability >= 0.5)

        # Generate unique ID for matching image and metadata in S3
        request_id = str(uuid.uuid4())

        # Schedule S3 upload in background (Executes AFTER function returns)
        background_tasks.add_task(
            save_production_data_to_s3,
            request_id=request_id,
            raw_text=text_input,
            image_bytes=image_bytes,
            prediction=prediction,
            probability=probability,
        )
        # Return JSON response back to Streamlit
        return {
            "status": "success",
            "prediction": prediction,
            "probability": round(probability, 4),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e!s}")


@app.post("/reload-model")
async def reload_model():
    """
    Reload the model from MLflow without reastarting API
    """
    global cross_att_classifier

    try:
        # Retrieve the model with production alias
        cross_att_classifier = mlflow.pytorch.load_model(
            MODEL_URI, map_location=torch.device("cpu")
        )
        cross_att_classifier.to(device).eval()

        return {
            "status": "success",
            "message": f"Model {REGISTERED_MODEL_NAME}@{MODEL_ALIAS} loaded successfully !",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Fail to load the model : {e!s}",
        }
