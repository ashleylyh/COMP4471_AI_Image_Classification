from PIL import Image, ImageChops, ImageEnhance
import os
from tqdm import tqdm

def convert_to_ela_image(image_path, quality=90):
    # Save the image at the given quality
    temp_file = 'temp.jpg'
    im = Image.open(image_path).convert('RGB')
    im.save(temp_file, 'JPEG', quality=quality)

    # Open the saved image and the original image
    saved = Image.open(temp_file)
    orignal = Image.open(image_path)

    # Find the absolute difference between the images
    diff = ImageChops.difference(orignal, saved)

    # Normalize the difference by multiplying with a scale factor and convert to grayscale
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    # Remove the temporary file
    os.remove(temp_file)

    return diff


def convert_file_images_to_ela(image_files, output_directory='./dataset/test_ela'):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    for image in tqdm(image_files):
        # Convert image to ELA image
        ela_image = convert_to_ela_image(image)
        # Save the ELA image
        ela_image.save(os.path.join(output_directory, os.path.basename(image)))