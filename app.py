import os
import io
import base64 
from diffusers import DiffusionPipeline
from transformers import pipeline
import gradio as gr


# text summarization
get_completion_summarizer = pipeline("summarization", model = 'sshleifer/distilbart-cnn-12-6')

def summarize(input):
    output = get_completion_summarizer(input)
    return output[0]['summary_text']

    
# NER 
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens


get_completion_ner = pipeline("ner", model="dslim/bert-base-NER")

def ner(input):
    output = get_completion_ner(input)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}



# image captioning
get_completion_captioning = pipeline("image-to-text")

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion_captioning(base64_image)
    return result[0]['generated_text']



# build tabs in interface
io1 = gr.Interface(fn=summarize, 
                inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                outputs=[gr.Textbox(label="Result", lines=3)],
                title="Text summarization with distilbart-cnn",
                description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
                )


io2 = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Andrew, I'm building DeeplearningAI and I live in California", "My name is Poli, I live in Vienna and work at HuggingFace"])


io3 = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never",
                   )


gr.TabbedInterface(
    [io1, io2, io3], ["Text Summarization", "Named Entity Recognition", "Image Captioning"]
).launch()


