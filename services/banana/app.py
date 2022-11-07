import torch
import transformers
import gradio as gr

from onnxruntime import InferenceSession
from PIL import Image

from transformers import (
    pipeline,
    AutoFeatureExtractor, 
    AutoModelForImageClassification,
)

###########################
# HELPER FUNCTIONS START ##
###########################

def preprocess_image(image):
    food_model_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path = 'stochastic/102722run')
    preprocessed_imaged = food_model_extractor(Image.open(image).convert("RGB"), return_tensors="np") #onnx expects numpy
    return preprocessed_imaged

def clean_up_final_answer(flan_answer):
    
    # replace extra periods 
    if flan_answer[-1] == '.':
        pass
    else:
        flan_answer = flan_answer + "."

    # get rid of underscores from answers
    clean_answer = flan_answer.replace("_", " ")

    return clean_answer

###########################
# HELPER FUNCTIONS END ##
###########################

def init():
    global food_classification_model
    global session
    global food_describer

    # image classifier: which do i pick
    session = InferenceSession("../../onnx/vit-model.onnx")
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    food_classification_model = AutoModelForImageClassification.from_pretrained("stochastic/102722run")

    # image describer
    food_describer = pipeline("text2text-generation", model='google/flan-t5-large')

def inference(model_inputs:dict) -> dict:
    global food_classification_model
    global session
    global food_describer

    image = model_inputs.get("image", None)
    output_name = session.get_outputs()[0].name

    if image is None:
        return {"message": "No image provided"}

    # food classifer
    image_preprocessed = preprocess_image(image)
    outputs = session.run(output_names = [output_name], input_feed=dict(image_preprocessed))
    predicted_class_idx = outputs[0].argmax(-1).item()
    predicted_food = food_classification_model.config.id2label[predicted_class_idx]
    
    # food describer
    first_sentence = f"This looks like {predicted_food}! "
    prompts = {
        f"Answer the following question: How do you enjoy {predicted_food}?"                       : f'I recommend to ',
        f"Answer the following question: What are the flavors of {predicted_food}"                 : 'The flavors are similar to ',
        f"Answer the following question: What are the textures of {predicted_food}?"               : 'The textures are ',
        f"Answer the following question: What are the aromas of {predicted_food}?"                 : 'The armoas will be ',
        f"Answer the following question: How do you improve the flavor of {predicted_food}?"       : 'To enjoy, you can '
    }

    answers = []
    for question in prompts.keys():
        answer = food_describer(question, max_length = 100)[0]["generated_text"].lower()
        if answer[-1] == '.':
            pass
        else:
            answer = answer + "."
            rec = prompts[question] + answer
            answers.append(rec)
    final_answer = clean_up_final_answer(first_sentence + " ".join(answers))
    return final_answer
