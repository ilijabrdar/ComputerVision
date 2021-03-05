import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 200, 255, cv2.THRESH_BINARY)
    #ret, image_bin = cv2.threshold(image_gs, 200, 255, cv2.THRESH_OTSU)
    return image_bin


def invert(image):
    return 255-image


def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    resized = cv2.resize(region, (110, 110), interpolation=cv2.INTER_NEAREST)
    return resized


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def convert_output(outputs):
    return np.eye(len(outputs))


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def find_contours(image_orig, image_bin, area_size):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im = image_orig.copy()
    regions_array = []

    sum_y = 0
    max_h = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        sum_y += y
        max_h = max(h, max_h)
    mean_y = sum_y / len(contours)

    for cont_index in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[cont_index])
        center = y + h / 2
        area = cv2.contourArea(contours[cont_index])
        if area > area_size and hierarchy[0][cont_index][3] == -1: #and ((mean_y - max_h / 2) < center) and (y < mean_y):  # and 100 > h > 15 and w > 20:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            if len(region) == 0:
                continue
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    merged = []
    used = []
    for i in range(len(regions_array)):
        if i in used:
            continue
        xi, yi, wi, hi = regions_array[i][1]
        for j in range(len(regions_array)):
            if j in used:
                continue
            xj, yj, wj, hj = regions_array[j][1]
            if (i != j) and (xi < xj) and ((xi + wi) > (xj + wj)) and (yj < yi):
                region = image_bin[yj:yi + hi + 1, xi:xi + wi + 1]
                merged.append([resize_region(region), (xi, yj, wi, hi + hj)])
                cv2.rectangle(image_orig, (xi, yj), (xi + wi, yi + hi), (255, 0, 0), 2)
                used.append(j)
                used.append(i)
                continue
        if i not in used:
            merged.append(regions_array[i])

    regions_array = sorted(merged, key=lambda item: item[1][0])
    return regions_array


def select_roi(image_orig, image_bin, area_size):
    regions_array = find_contours(image_orig, image_bin, area_size)
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    # for i in range(1, len(regions_array)):
    #     xi, yi, wi, hi = regions_array[i][1]
    #     xj, yj, wj, hj = regions_array[i - 1][1]
    #     if (xj + wj) > xi:
    #         region = image_bin[yi:yi + hi + 1, xj + wj:xi + wi + 1]
    #         regions_array[i][0] = resize_region(region)

    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions, region_distances


def straighten_image(image_orig, image_bin, area_size):
    regression_x = []
    regression_y = []
    regions_array = find_contours(image_orig, image_bin, area_size)
    for region in regions_array:
        x, y, w, h = region[1]
        regression_x.append(x + w/2)
        regression_y.append(y + h/2)
    angle = linear_regression(regression_x, regression_y)
    print(angle)
    center = (regression_x[-1], regression_y[-1])
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    result_orig = cv2.warpAffine(image_orig, rot_mat, image_orig.shape[1::-1], flags=cv2.INTER_LINEAR)
    result_bin = cv2.warpAffine(image_bin, rot_mat, image_bin.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result_orig, result_bin


def linear_regression(x, y):
    X = np.array(x).reshape((-1, 1))
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    return np.arctan(reg.coef_) * 180 / 3.14
