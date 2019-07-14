import cv2
import numpy as np
from main import result
from Model import Model, DecoderType

#import image
# image = cv2.imread('/home/yasmine/PycharmProjects/ProjetAnnuel/text-segmentation/src/imgproc/cpp/img02.png', cv2.IMREAD_GRAYSCALE)

# replace path 
image = cv2.imread('/home/yasmine/PycharmProjects/ProjetAnnuel/text-segmentation/src/imgproc/cpp/img023.png')

model = Model(open('../model/charList.txt').read(), mustRestore=True)

def to_lines (image):
    linesList = []

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    # print(gray.shape)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('second', thresh)
    # cv2.waitKey(0)

    # dilation
    kernel = np.ones((5, 900), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # cv2.imshow('dilated', img_dilation)
    # cv2.waitKey(0)

    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]

        # show ROI
        linesList.append(roi)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
        # print(1)

    # cv2.imwrite('final_bounded_box_image.png',image)
    # cv2.imshow('marked areas', image)
    # print(2)

    return reversed(linesList)
    # return linesList


def to_words (image):

    wordList = []

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('second', thresh)
    # cv2.waitKey(0)

    # dilation
    kernel = np.ones((5, 35), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # cv2.imshow('dilated', img_dilation)
    # cv2.waitKey(0)

    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]

        # show ROI
        roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if roi.shape > (50,50) :
            wordList.append(roi)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
        # print(1)

    # cv2.imwrite('final_bounded_box_image.png',image)
    # cv2.imshow('marked areas', image)
    # print(2)

    return wordList


s = ""

lines = to_lines(image)
for line in lines:
    words = to_words(line)
    for word in words:
        # print(word.shape)
        w, prob = result(word, model)
        s = s + " " + w
#         cv2.imshow('word', line)
#         cv2.waitKey(0)
# words = to_words(lines[0])
# # print("yasss ", words[0].shape)
# for word in words:
#     w, prob = result(word, model)
#     s = s+" "+w

    # cv2.imshow('word', word)
    # cv2.waitKey(0)

print(s)