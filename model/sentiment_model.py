from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model from the Hugging Face Model Hub
model_name = "moazx/AraBERT-Restaurant-Sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the device to run inference on (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)

def predict_sentiment(review):
    # Step 1: Tokenization
    encoded_text = tokenizer(
        review, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )

    # Move input tensors to the appropriate device
    input_ids = encoded_text["input_ids"].to(device)
    attention_mask = encoded_text["attention_mask"].to(device)

    # Step 2: Inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Step 3: Prediction with probabilities
    probs = torch.softmax(outputs.logits, dim=-1)
    probs = (
        probs.squeeze().cpu().numpy()
    )  # Convert to numpy array and remove the batch dimension

    # Map predicted class index to label
    label_map = {0: 'سلبي', 1: 'إيجابي'}

    output_dict = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    return output_dict
