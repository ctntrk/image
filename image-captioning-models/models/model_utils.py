from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_model(model_type="base"):
    if model_type == "large":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    else:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model

def generate_caption(
    image_path, 
    model_type="base",
    max_length=50,
    num_beams=5,
    repetition_penalty=1.5,
    temperature=0.7
):
    processor, model = load_model(model_type)
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(model.device)
    
    out = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=False
    )
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
