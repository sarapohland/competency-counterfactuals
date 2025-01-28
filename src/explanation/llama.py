import torch
import requests

from transformers import MllamaForConditionalGeneration, AutoProcessor


class Llama():

    def __init__(self, max_new_tokens=200):
        # Load pretrained Llama 3.2 11B model and processor
        model_id = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
        self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Set maximum length of response
        self.max_new_tokens = max_new_tokens

    def query_model(self, image, query, instructions=None):
        # Query the model with the user question
        if instructions is None:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]}
            ]
        else:
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]}
            ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)

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
    instructions = "You are a helpful chatbot that writes short, concise answers."

    # Write user prompt for language model
    query = "Could you please describe this image?"

    # Load Llama 3.2 model
    llama = Llama()

    for idx in range(10):
        # Load an image
        image_path = f'results/lunar/explainability/images/spatial/{idx}_orig.png'
        image = Image.open(image_path).convert('RGB').resize((320,240))

        # Generate response from Llama
        answer = llama.query_model(image, query, instructions)

        print(f'Llama description for image {idx}:')
        print(answer)
