# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from transformers import (
    pipeline,
    AutoFeatureExtractor, 
    AutoModelForImageClassification,
)

def download_model():

    # food classifier
    food_model_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path = 'stochastic/102722run') #this can also be a tokenizer
    food_classification_model = AutoModelForImageClassification.from_pretrained("stochastic/102722run")

    # food describer
    pipeline("text2text-generation", model='google/flan-t5-large',)

if __name__ == "__main__":
    download_model()