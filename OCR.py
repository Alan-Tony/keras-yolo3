import easyocr
import cv2 as cv
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torch._C import wait

def drawDetection(image, box, text, font, thickness):

    left, top, right, bottom = box
    color = (255, 0, 0)

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
            outline=color)

    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=color)
    draw.text(text_origin, text, fill=(0, 0, 0), font=font)

    del draw
    return image

if __name__ == '__main__':

    reader = easyocr.Reader(['en'])

    vid = cv.VideoCapture('football2.mp4')

    while vid.isOpened():

        retval, frame = vid.read()

        if not retval or cv.waitKey(1) & 0xFF == ord('q'):
            break

        image = Image.fromarray(frame)

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 600

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        detections = reader.readtext(rgb_frame)
        
        result = image.copy()
        for detection in detections:
            bbox, text, prob = detection

            p1, p2, p3, p4 = bbox
            left, top = p1
            right, bottom = p3
            box = (left, top, right, bottom)

            result =  drawDetection(result, box, text, font, thickness)

        result = np.asarray(result)
        cv.imshow('Result', result)

    #cv.waitKey(0)    
    cv.destroyAllWindows()