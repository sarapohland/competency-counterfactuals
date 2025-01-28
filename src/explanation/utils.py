import textwrap
import matplotlib.pyplot as plt

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


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
