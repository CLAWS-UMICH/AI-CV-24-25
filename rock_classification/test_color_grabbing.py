from PIL import Image
from colorthief import ColorThief
import webcolors
from webcolors import rgb_to_name

def get_dominant_color(image_path):
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    return dominant_color

def closest_color(requested_colour):
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_color_name(requested_color):
    try:
        # Direct match
        return webcolors.rgb_to_name(requested_color)
    except ValueError:
        # No direct match, find the closest color
        return closest_color(requested_color)
    
image_path = "data/train/Granite/3.jpg"
dominant_color = get_dominant_color(image_path)
print(dominant_color)
print(get_color_name(dominant_color))
