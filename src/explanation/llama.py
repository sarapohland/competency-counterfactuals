import os
import torch

from PIL import Image

from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

from unsloth import FastVisionModel
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

# Set device to GPU 0 or 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


class Llama():

    def __init__(self, model_name, max_new_tokens=200):
        # Load the pre-trained Llama 3.2 Vision model
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit = True,
            use_gradient_checkpointing = "unsloth",
        )
        self.model.to(device)

        # Set up the LoRA (Low-Rank Adaptation)
        try:
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers     = True, 
                finetune_language_layers   = True, 
                finetune_attention_modules = True,
                finetune_mlp_modules       = True,
                r = 16,           
                lora_alpha = 16,
                lora_dropout = 0,
                bias = "none",
                random_state = 3443,
                use_rslora = False,
                loftq_config = None,
            )
        except:
            pass

        # Set maximum length of response
        self.max_new_tokens = max_new_tokens

    def convert_response(self, image, instructions, response):
        # Convert response to appropriate format 
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instructions}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ],
            },
        ]
        return {"messages": messages}

    def query_model(self, image, instructions):
        # Change model to run in evaluation mode
        FastVisionModel.for_inference(self.model)

        # Query the model with the user question
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instructions},
                ],
            }
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract response after the prompt
        assistant_start = decoded_output.find("assistant")
        if assistant_start != -1:
            response = decoded_output[assistant_start + len("assistant"):].strip()
        else:
            response = decoded_output.strip()
        return response


if __name__ == "__main__":

    from PIL import Image

    # Write high-level instructions for language model
    instructions = """
    You are a helpful chatbot that writes short, concise answers.
    Write a description for the image provided.
    """

    # Load Llama 3.2 model
    model = "unsloth/Llama-3.2-11B-Vision-Instruct"
    llama = Llama(model)

    for idx in range(10):
        # Load an image
        image_path = f'results/lunar/explainability/images/spatial/{idx}_orig.png'
        image = Image.open(image_path).convert('RGB').resize((320,240))

        # Generate response from Llama
        answer = llama.query_model(image, instructions)

        print(f'Llama description for image {idx}:')
        print(answer)
