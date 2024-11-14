# FastAPI server config:

# auth token = 2ofZGlWe57sGu6PZd7oGHdxzmOB_4UaN8pEawCaFxC88zEVGw

pip install fastapi uvicorn nest_asyncio pyngrok

# 2. Set up ngrok authentication
ngrok authtoken "2ofZGlWe57sGu6PZd7oGHdxzmOB_4UaN8pEawCaFxC88zEVGw"  # Replace with token from ngrok.com

# 3. Import all required libraries
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from pyngrok import ngrok
import base64
import io
from PIL import Image
import tensorflow as tf
import numpy as np
from urllib.parse import unquote
import logging
import tempfile
import os
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 4. Create FastAPI app
app = FastAPI()

# 5. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Define preprocessing function
def preprocess_image(image_data):
    try:
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        # Decode base64 string
        image_bytes = base64.b64decode(image_data)

        # Open image using PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to match the model's expected input size
        image = image.resize((299, 299))  # Make sure this matches your IMAGE_SIZE

        # Convert to array and preprocess
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        img_array = img_array / 255.0  # Normalize pixel values

        logger.info("Image preprocessed successfully")
        return img_array

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

# 7. Define request model
class ImageRequest(BaseModel):
    image: str

# 8. Define endpoint
@app.post("/generate-caption")
async def process_image(request: ImageRequest):
    try:
        logger.info("Received image request")

        # Remove data URL prefix if present
        image_data = request.image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        # Decode base64 string
        image_bytes = base64.b64decode(image_data)

        # Save image bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            temp_filename = tmp.name
            tmp.write(image_bytes)

        # Pass the temporary file path to greedy_algorithm
        caption = greedy_algorithm(temp_filename)
        logger.info(f"Generated caption: {caption}")

        # Clean up temporary file
        os.remove(temp_filename)

        return {"caption": caption, "status": "success"}

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"error": str(e), "status": "error"}

# 9. Set up and start the server
nest_asyncio.apply()

# Create the ngrok tunnel
public_url = ngrok.connect(8000)
print('Public URL:', public_url)  # Save this URL for your browser extension

# Start the server
uvicorn.run(app, port=8000)