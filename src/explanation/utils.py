import textwrap
import matplotlib.pyplot as plt

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

def get_max_new_tokens(response_type):
    if response_type == 'free':
        return 100
    elif response_type == 'mc':
        return 40

    else:
        raise NotImplementedError('The response type should either be `free` or `mc`.')
    
def get_llama_instructions(dataset, response_type, counterfactual):

    # Generate descriptions of the scenario
    if dataset == 'lunar':
        description = (
            "I trained a CNN for image classification from a set of images obtained from a "
            "simulated lunar environment. The classifier learns to distinguish between "
            "different regions in this environment, such as regions with smooth terrain, "
            "regions with bumpy terrain, regions at the edge of a crater, regions inside "
            "a crater, and regions near a hill."
        )
    elif dataset == 'speed':
        description = (
            "I trained a CNN for image classification from a dataset containing speed "
            "limit signs. The classifier learns to distinguish between seven (7) "
            "different speed limits: 30, 50, 60, 70, 80, 100, and 120 km/hr."
        )
    else:
        description = "I trained a CNN for image classification."
    
    description += (
        " In addition to the classification model, I trained a reconstruction-based "
        "competency estimator that estimates the probability that the classifier's "
        "prediction is accurate for a given image."
    )

    # Generate instructions for the language model
    if response_type == 'free':
        if counterfactual == 'none':
            instructions = (
                " Here is an image for which the classifier is not confident. In a "
                "single sentence, explain what properties of the image itself might "
                "have led to the observed reduction in model confidence."
            )
        else:
            instructions = (
                " Here are two images side-by-side. The first (on the left) is the "
                "original image, for which the classifier is not confident. The "
                "second image (on the right) is a similar image, for which the model is "
                "more confident. In a single sentence, explain what properties of the "
                "original image might have led to the observed reduction in confidence."
            )
    
    elif response_type == 'mc':
        if counterfactual == 'none':
            instructions = (
                " Here is an image for which the classifier is not confident. There "
                "are six potential reasons the model may not be confident for this "
                "particular image: (1) the image contains spatial anaomlies in particular "
                "regions of the image, (2) the image has an unusual brightness level, "
                "(3) the image has an unusual contrast level, (4) the image has an unusual "
                "saturation level, (5) the image is noisy, or (6) the image has been "
                "pixelated. Select the most likely cause for low confidence. Be sure to "
                "include the number corresponding to your selection and then explain your "
                "answer."
            )
        else:
            instructions = (
                " Here are two images side-by-side. The first (on the left) is the "
                "original image, for which the classifier is not confident. The "
                "second image (on the right) is a similar image, for which the model is "
                "more confident. There are six potential reasons the model may not be "
                "confident for the original image: (1) the image contains spatial "
                "anaomlies in particular regions of the image, (2) the image has an "
                "unusual brightness level, (3) the image has an unusual contrast level, "
                "(4) the image has an unusual saturation level, (5) the image is noisy, "
                "or (6) the image has been pixelated. Select the most likely cause for "
                "low confidence in the original image. Be sure to include the number "
                "corresponding to your selection and then explain your answer."
            )
    
    else:
        raise NotImplementedError('The response type should either be `free` or `mc`.')
    
    instructions = description + instructions
    return instructions

def combine_images(orig, gen, pad=10):
    # Paste images side-by-side
    width = orig.width + gen.width + 3 * pad
    height = max(orig.height, gen.height) + 2 * pad
    image = Image.new("RGB", (width, height), (255, 255, 255))
    image.paste(orig, (pad, pad))
    image.paste(gen, (orig.width + 2 * pad, pad))
    return image

def display_images(expl, orig, gen=None, buffer=250, file=None):
    if gen is not None:
        # Create blank figure
        tot_width = orig.width + gen.width
        img_height = max(orig.height, gen.height)
        tot_height = img_height + buffer
        image = Image.new("RGB", (tot_width, tot_height), 'white')

        # Visualize original and generated image
        image.paste(orig, (0, 0))
        image.paste(gen, (orig.width, 0))

        # Visualize explanation under image
        draw = ImageDraw.Draw(image)
        font_path = font_manager.findfont('DejaVu Sans')
        font = ImageFont.truetype(font_path, size=20)
        
        font_width = font.getbbox("A")[2] # Get character width
        lines = textwrap.wrap(expl, width=int(tot_width / font_width))
        y_text = orig.height + 20 # Start drawing below the image
        for line in lines:
            text_width = font.getbbox(line)[2]
            text_height = font.getbbox(line)[3] - font.getbbox(line)[1]  # Line height
            x_text = (tot_width - text_width) // 2  # Center the text
            draw.text((x_text, y_text), line, fill="black", font=font)
            y_text += text_height  # Move to the next line

        # Display images with explanation
        fig = plt.figure(figsize=(6, 4))
        plt.imshow(image)
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if file is None:
            plt.show(block=False)
        else:
            plt.savefig(file)

    else:
        # Create blank figure
        tot_width = 2 * orig.width
        tot_height = orig.height + buffer
        image = Image.new("RGB", (tot_width, tot_height), 'white')

        # Visualize original image
        image.paste(orig, (int(orig.width / 2), 0))

        # Visualize explanation under image
        draw = ImageDraw.Draw(image)
        font_path = font_manager.findfont('DejaVu Sans')
        font = ImageFont.truetype(font_path, size=20)
        
        font_width = font.getbbox("A")[2] # Get character width
        lines = textwrap.wrap(expl, width=int(tot_width / font_width))
        y_text = orig.height + 20 # Start drawing below the image
        for line in lines:
            text_width = font.getbbox(line)[2]
            text_height = font.getbbox(line)[3] - font.getbbox(line)[1]  # Line height
            x_text = (tot_width - text_width) // 2  # Center the text
            draw.text((x_text, y_text), line, fill="black", font=font)
            y_text += text_height  # Move to the next line

        # Display image with explanation
        plt.figure(figsize=(6, 4))
        plt.imshow(image)
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if file is None:
            plt.show(block=False)
        else:
            plt.savefig(file)

def get_true_explanation(dataset, property, factor):
    if property == 'spatial':
        if dataset == 'lunar':
            return (
                "The original image contains a human-made structure that is unfamiliar "
                "to the perception model."
            )
        elif dataset == 'speed':
            return (
                "The original image displays a speed limit of 20, which is not a class "
                "present in the training set."
            )
        else:
            return "The original image contains a spatial anomaly."

    elif property == 'brightness':
        if factor < 1:
            return "The original image has low brightness levels."
        else:
            return "The original image has high brightness levels."
    
    elif property == 'contrast':
        if factor < 1:
            return "The original image has low levels of contrast."
        else:
            return "The original image has high levels of contrast."
    
    elif property == 'saturation':
        if factor < 1:
            return "The original image has low levels of saturation."
        else:
            return "The original image is over-saturated."

    elif property == 'noise':
        return "The original image is noisy."

    elif property == 'pixelate':
        return "The original image is pixelated."

    else:
        raise NotImplementedError('Unknown image property.')