import os
import json
from PIL import Image, ImageStat  # pip install Pillow


def compute_average_rgb(image):
    """
    Given a PIL Image, return its average (R, G, B) as a tuple of ints.
    """
    image = image.convert("RGB")
    stat = ImageStat.Stat(image)
    r, g, b = stat.mean
    return int(round(r)), int(round(g)), int(round(b))


def get_color_name_from_filename(filepath):
    """
    Given a file path, return the color name (basename without extension).
    e.g. '/path/cadmium red.jpg' -> 'cadmium red'
    """
    base = os.path.basename(filepath)
    name, _ = os.path.splitext(base)
    return name


def extract_color_averages_from_folder(folder_path):
    """
    Iterate over JPG/JPEG files in a folder (non-recursive),
    and build a dict {color_name: (R, G, B)} based on average color.
    """
    color_dict = {}

    for entry in os.listdir(folder_path):
        filepath = os.path.join(folder_path, entry)

        # Skip directories
        if not os.path.isfile(filepath):
            continue

        # Check extension
        ext = os.path.splitext(entry)[1].lower()
        if ext not in [".png"]:
            continue

        color_name = get_color_name_from_filename(filepath)

        # Open image and compute average
        with Image.open(filepath) as img:
            avg_rgb = compute_average_rgb(img)

        color_dict[color_name] = avg_rgb

    return color_dict


if __name__ == "__main__":
    # Change this to your folder path, or pass via sys.argv if you prefer
    folder = "schminke_norma_colors"

    color_averages = extract_color_averages_from_folder(folder)

    # Print nicely as JSON
    print(json.dumps(color_averages, indent=4))
