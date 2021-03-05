# import libraries here
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """
    red_blood_cell_count = 0
    white_blood_cell_count = 0
    has_leukemia = None

    img1 = cv.imread(image_path)
    img = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img_red = 255 - img[:, :, 1]

    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure
    white_blood_cell_count = count_white_blood_cells(img_red)
    red_blood_cell_count = count_red_blood_cells(img_red)
    #red_blood_cell_count -= white_blood_cell_count

    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure
    #has_leukemia = (white_blood_cell_count / red_blood_cell_count) > 0.08
    has_leukemia = (red_blood_cell_count - white_blood_cell_count) > 25

    return red_blood_cell_count, white_blood_cell_count, has_leukemia


def count_red_blood_cells(img):
    img_bin = (img > 70) * (img < 120)
    img_bin = np.uint8(img_bin)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)

    _, contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = []
    for c in contours:
        areas.append(cv.contourArea(c))
    mean_area = sum(areas) / len(areas)

    filtered = []
    additional = 0
    for cont_index in range(len(contours)):
        c = contours[cont_index]
        if cv.contourArea(c) > 0.5 * mean_area and hierarchy[0][cont_index][3] == -1:
            if cv.contourArea(c) > 10 * mean_area:
                if cv.contourArea(c) > 15 * mean_area:
                    if cv.contourArea(c) > 20 * mean_area:
                        additional += 2
                    additional += 2
                additional += 2
            filtered.append(c)

    image = img.copy()
    cv.drawContours(image, filtered, -1, (255, 0, 0), 1)
    plt.imshow(image)
    plt.show()
    return 1.1 * len(filtered) + additional


def count_white_blood_cells(img_red):
    ret, img_bin = cv.threshold(img_red, 120, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    opening = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)

    img, contours, hierarchy = cv.findContours(opening, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    areas = []
    for c in contours:
        areas.append(cv.contourArea(c))
    mean_area = sum(areas) / len(areas)

    filtered = []
    additional = 0
    for c in contours:
        if cv.contourArea(c) > 0.5 * mean_area:
            if cv.contourArea(c) > 8 * mean_area:
                additional += 2
            filtered.append(c)

    image = img_red.copy()
    cv.drawContours(image, filtered, -1, (255, 0, 0), 1)
    plt.imshow(image)
    plt.show()
    return len(filtered) + additional

