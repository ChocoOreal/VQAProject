import streamlit as st
from PIL import Image
import predict
from transformers import AutoTokenizer, AutoImageProcessor, logging
import os
from datasets import set_caching_enabled
import torch

# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_caching_enabled(True)
logging.set_verbosity_error()
device_option = None
device_option = st.selectbox("Do you want to run model on GPU?", ("None", "Yes", "No"))
if (device_option == "None"):
    st.warning("Please choose accelerator")
    st.stop()
if (device_option == "Yes"):
    if (torch.cuda.is_available()):
        device = torch.device('cuda:0')
    else:
        st.warning("GPU is not available")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')


#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

@st.cache_resource
def load_model():
    model, answer_space = predict.deploy_model(device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    return model, answer_space, tokenizer, image_processor

model, answer_space, tokenizer, image_processor = load_model()

def processInput(image, question, answer_space):
    
    pixel_values = image_processor(image.convert("RGB"), return_tensors="pt")
    tokens = tokenizer(question, padding="max_length", truncation=True, max_length=100, return_tensors="pt")

    return pixel_values['pixel_values'], tokens


st.title("Visual question answering system")
uploaded_file = st.file_uploader("Upload an image")
if (uploaded_file is not None):
    image = Image.open(uploaded_file)
    st.image(image)

question = st.text_input("Question")
if (uploaded_file is not None and question is not None):
    pixel_values, tokens = processInput(image, question, answer_space)

    answer = predict.predict(pixel_values, tokens, model, answer_space)
    if (answer is not None):  st.write(answer)

