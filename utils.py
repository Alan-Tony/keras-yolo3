import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw
import re
from itertools import compress

def drawDetection(image, box, text, font, thickness, text_box_fill = (0, 85, 255)):

    left, top, right, bottom = box
    text_color = (0, 0, 0)

    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(text, font)

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=text_box_fill)

    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=text_box_fill)
    draw.text(text_origin, text, fill=text_color, font=font)

    del draw
    return image


def displayColors(colors):

    if colors is None:
        colors = []
    
    if not len(colors):
        print("No colors to display. Using default color")
        colors.append((255, 0, 0))

    #Colors should be a list of tuples
    num_colors = len(colors)

    img = np.zeros((25, num_colors*25, 3), dtype='uint8')
    for i, color in enumerate(colors):

        b, g, r = color
        img[:, i*25:(i+1)*25, 0] = b
        img[:, i*25:(i+1)*25, 1] = g
        img[:, i*25:(i+1)*25, 2] = r

    cv.imshow('Colors', img)
    cv.waitKey(0)
    cv.destroyWindow('Colors')


def get_colors(img, numcolors=10, resize=150):

    # Resize image to speed up processing
    img = Image.fromarray(img)
    img.thumbnail((resize, resize))

    # Reduce to palette
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=numcolors)

    # Find dominant colors
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    colors = list()
    for i in range(len(color_counts)):
        palette_index = color_counts[i][1]
        dominant_color = palette[palette_index*3:palette_index*3+3]
        colors.append(tuple(dominant_color))

    return colors, [x[0] for x in color_counts]


"""
bbox should be in the format (x0, y0, w, h)
"""
def getJerseyColors(img, default_color = (255, 0, 0)):

    assert img is not None, "Empty image received"
    assert type(img) == np.ndarray, "Image should be a numpy array (uint8)"
    assert len(list(img.shape)) == 3 and img.shape[2] == 3, "Image should be in  3 channel BGR format"

    if not (img.shape[0] & img.shape[1]):   #Check if the image a line
        return default_color

    #Getting all prominent colors and their frequency
    colors, color_counts = get_colors(img)

    #Filter out green colors (color of field)
    gr_limit = 1.08
    gb_limit = 1.5
    blue_ratios = np.array([ 1 if abs(g-b) < 3 else gb_limit if not b else g/b for (b, g, r) in colors])
    red_ratios = np.array([ 1 if abs(g-r) < 3 else gr_limit if not r else g/r for (b, g, r) in colors])
    idx = np.where( (blue_ratios < gb_limit) | (red_ratios < gr_limit), True, False)
    non_green_colors = list(compress(colors, idx))
    non_green_color_counts = list(compress(color_counts, idx))

    if not len(non_green_colors):
        return default_color    #Default color in case no color passed the filters

    #Filter based on brightness and saturation
    np_non_green = np.array([np.array(x, dtype= 'uint8') for x in non_green_colors])
    np_non_green = np_non_green.reshape((len(non_green_colors), 1, 3))
    hsv_non_green = (cv.cvtColor(np_non_green, cv.COLOR_BGR2HSV)).reshape((len(non_green_colors), 3))
    saturation = hsv_non_green[:, 1]
    value = hsv_non_green[:, 2]
    idx = np.where((saturation > 100) & (value > 80), True, False)
    filtered_colors = list(compress(non_green_colors, idx))
    filtered_color_counts = list(compress(non_green_color_counts, idx))

    #Get the most frequent non green color
    if not len(filtered_colors):
        return default_color  #Default color in case no color passed the filters
    else:
        jersey_color = filtered_colors[np.argmax(filtered_color_counts)]

    return jersey_color


"""
Variable "reader" should be an EasyOCR reader object
"""
def player_detection(image, original, bbox, reader, font, thickness, text_box_fill):

    cv_img = np.asarray(original)
    left, top, right, bottom = bbox
    cv_img = cv_img[top:bottom, left:right]

    detections = reader.readtext(cv_img)

    numbers = []
    for detection in detections:

        _, text, _ = detection
        #Finding all numbers in the player's frame
        numbers += re.findall('\d+', text)

    numbers = list(map(int, numbers))
    #print(numbers)
    jnos = list(range(1, 101))
    common = list((set(numbers).intersection(set(jnos))))
    if len(common):
        text = 'Number: ' + str(max(common))
    else:
        text = "None"

    return drawDetection(image, bbox, text, font, thickness, text_box_fill)