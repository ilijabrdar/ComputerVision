from imutils import face_utils
import numpy as np
import argparse
import imutils
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import dlib
import cv2
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier
# from keras.models import Sequential
# from keras.layers.core import Dense,Activation
# from keras.optimizers import SGD
# from keras.models import model_from_json
from numpy import random


def load_image(path):
    #return cv2.imread(path),50,100)
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def hog_desc(shape):
    nbins = 90
    cell_size = (5, 5)
    block_size = (3, 3)

    hog = cv2.HOGDescriptor(_winSize=(shape[1] // cell_size[1] * cell_size[1],
                                      shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog

def extract_face_features_train(imgs):
    hog = hog_desc((200, 200))
    features = []
    for img in imgs:
        features.append(hog.compute(img))

    features = np.array(features)
    features = reshape_data(features)
    return features

def extract_face(img):
    detector = dlib.get_frontal_face_detector()
    rects = detector(img, 1)
    if len(rects) == 0:
        print('ops')
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
        return img
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_NEAREST)
        return face

# def extract_features_gender(imgs):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#     features = []
#     for img in imgs:
#         rects = detector(img, 1)
#         if len(rects) == 0:
#             continue
#         for (i, rect) in enumerate(rects):
#             shape = predictor(img, rect)
#             shape = face_utils.shape_to_np(shape)
#             mouth_ratio_first = (shape[51][1] - shape[62][1]) / (shape[66][1] - shape[57][1])
#             mouth_ratio_second = (shape[50][1] - shape[61][1]) / (shape[67][1] - shape[58][1])
#             mouth_nose = (shape[33][1] - shape[51][1])
#             eyebrows = (shape[19][1] - shape[17][1]) + (shape[24][1] - shape[26][1])
#             # reg_nose = LinearRegression().fit([[shape[31][0]], [shape[32][0]], [shape[33][0]]],
#             #                                   [[shape[31][1]], [shape[32][1]], [shape[33][1]]])
#             # reg_jaw = LinearRegression().fit([[shape[3][0]], [shape[4][0]], [shape[5][0]], [shape[6][0]], [shape[7][0]], [shape[8][0]]],
#             #                                  [[shape[3][1]], [shape[4][1]], [shape[5][1]], [shape[6][1]], [shape[7][1]], [shape[8][1]]])
#             reg_nose = make_pipeline(PolynomialFeatures(2), LinearRegression())
#             reg_nose.fit([[shape[31][0]], [shape[32][0]], [shape[33][0]]],
#                         [[shape[31][1]], [shape[32][1]], [shape[33][1]]])
#             features.append([mouth_ratio_first, mouth_ratio_second, mouth_nose, eyebrows, reg_nose.coef_])
#     features = np.array(features)
#     return features


def extract_features_race(imgs):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    features = []
    for img in imgs:
        rects = detector(img, 1)
        if len(rects) == 0:
            print('ops')
            continue
        for (i, rect) in enumerate(rects):
            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape)

            # # konvertovanje pravougaonika u bounding box koorinate
            # (x, y, w, h) = face_utils.rect_to_bb(rect)
            # # crtanje pravougaonika oko detektovanog lica
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # for (x, y) in shape:
            #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            #
            # plt.imshow(img, 'gray')
            # plt.show()
            # if x < 0:
            #     w -= x
            #     x = 0
            # if y < 0:
            #     h -= y
            #     y = 0
            #
            # face = img[x:x+w, y:y+w]
            # cv2.resize(face, (150, 150), interpolation=cv2.INTER_NEAREST)
            # # plt.imshow(face, 'gray')
            # # plt.show()
            # rects2 = detector(face, 1)
            # rect2 = rects2[0]
            # shape = predictor(face, rect2)
            # shape = face_utils.shape_to_np(shape)
            # for (x, y) in shape:
            #     cv2.circle(face, (x, y), 1, (0, 0, 255), -1)
            # plt.imshow(face, 'gray')
            # plt.show()

            eye_nose = (shape[27][0] - shape[39][0]) / (shape[42][0] - shape[27][0])
            eye_mouth = (shape[48][1] - shape[40][1]) / (shape[54][1] - shape[47][1])
            left_eye_nose_mouth = (shape[27][0] - shape[39][0]) / (shape[48][1] - shape[40][1])
            right_eye_nose_mouth = (shape[42][0] - shape[27][0]) / (shape[54][1] - shape[47][1])
            eye_nose_mouth = (shape[42][0] - shape[39][0]) / (shape[51][1] - shape[27][1])
            eye_eye_mouth = (shape[42][0] - shape[39][0]) / (shape[48][1] - shape[40][1])
            eye_eye_nose = (shape[42][0] - shape[39][0]) / (shape[27][0] - shape[39][0])
            eye_nose_nose = (shape[42][0] - shape[27][0]) / (shape[33][1] - shape[27][1])
            nose_mouth = (shape[33][1] - shape[27][1]) / (shape[51][1] - shape[27][1])
            random_feature1 = (shape[35][0] - shape[31][0]) / (shape[33][1] - shape[27][1])
            random_feature2 = (shape[41][1] - shape[37][1]) / (shape[39][0] - shape[36][0])
            random_feature3 = (shape[56][1] - shape[27][1]) / (shape[33][1] - shape[27][1])
            random_feature4 = (shape[42][0] - shape[39][0]) / (shape[45][0] - shape[36][0])
            random_feature5 = (shape[16][0] - shape[0][0]) / (shape[8][1] - shape[19][1])
            random_feature6 = (shape[22][0] - shape[21][0]) / (shape[26][0] - shape[17][0])
            random_feature7 = (shape[8][1] - shape[57][1]) / (shape[8][1] - shape[51][1])
            random_feature8 = (shape[8][1] - shape[33][1]) / (shape[8][1] - shape[19][1])
            random_feature9 = (shape[27][0] - shape[0][0]) / (shape[16][0] - shape[0][0])
            random_feature10 = (shape[42][0] - shape[39][0]) / (shape[8][1] - shape[39][1])
            features.append([eye_nose, eye_mouth, left_eye_nose_mouth, right_eye_nose_mouth, eye_nose_mouth,
                          eye_eye_mouth, eye_eye_nose, eye_nose_nose, nose_mouth, random_feature1,
                          random_feature2, random_feature3, random_feature4, random_feature5, random_feature6,
                             random_feature7, random_feature8, random_feature9, random_feature10
                          ])
    features = np.array(features)
    print(features)
    return features


def train_classification_model(features, labels):
    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(features, labels)
    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # clf.fit(features, labels)
    return clf_svm

def train_knn_model(features, labels):
    clf_knn = KNeighborsClassifier(n_neighbors=10)
    clf_knn = clf_knn.fit(features, labels)
    return clf_knn

def train_regression_model(features, labels):
    reg = LinearRegression().fit(features, labels)
    return reg

def train_or_load_age_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju


    # try:
    #     model = load('serialization_folder/gender.joblib')
    # except Exception:
    #model = generate_model(train_image_paths, train_image_labels)
        # dump(model, 'serialization_folder/gender.joblib')
    #model = generate_model_age(train_image_paths, train_image_labels)

    imgs = []
    for path in train_image_paths:
        imgs.append(cv2.resize(load_image(path), (200, 200), interpolation=cv2.INTER_NEAREST))
    labels = np.array(train_image_labels)
    model = train_classification_model(extract_face_features_train(imgs), labels)
    return model

    # imgs = []
    # for path in train_image_paths:
    #     imgs.append(cv2.resize(load_image(path), (150, 150), interpolation=cv2.INTER_NEAREST))
    # labels = np.array(train_image_labels)
    # features = extract_features_race(imgs)
    # model = train_classification_model(features, labels)
    # # serialize(model)
    return model


def train_or_load_gender_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    # try:
    #     model = load('serialization_folder/gender.joblib')
    # except Exception:
    #model = generate_model(train_image_paths, train_image_labels)
        # dump(model, 'serialization_folder/gender.joblib')
    # imgs = []
    # for path in train_image_paths:
    #     imgs.append(cv2.resize(load_image(path), (150, 150), interpolation=cv2.INTER_NEAREST))
    # model = train_classification_model(extract_features_gender(imgs), np.array(train_image_labels))
    # imgs = []
    # for path in train_image_paths:
    #     imgs.append(cv2.resize(load_image(path), (150, 150), interpolation=cv2.INTER_NEAREST))
    # labels = np.array(train_image_labels)
    # features = extract_features_race(imgs)
    # model = train_classification_model(features, labels)

    imgs = []
    for path in train_image_paths:
        imgs.append(cv2.resize(load_image(path), (200, 200), interpolation=cv2.INTER_NEAREST))
    labels = np.array(train_image_labels)
    model = train_classification_model(extract_face_features_train(imgs), labels)
    return model


def train_or_load_race_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    # try:
    #     model = load('serialization_folder/gender.joblib')
    # except Exception:
    #model = generate_model(train_image_paths, train_image_labels)
        # dump(model, 'serialization_folder/gender.joblib')
    #model = None
    #ann = load()

    # if ann is not None:
    #     return ann
    # imgs = []
    # for path in train_image_paths:
    #     imgs.append(cv2.resize(load_image(path), (150, 150), interpolation=cv2.INTER_NEAREST))
    # labels = np.array(train_image_labels)
    # features_resampled, labels_resampled = SMOTE().fit_resample(extract_features_race(imgs), labels)
    # model = train_classification_model(features_resampled, labels_resampled)
    #model = train_ann(features_resampled, convert_output(labels_resampled))
    #model = train_classification_model(features_resampled, labels_resampled)
    #serialize(model)

    imgs = []
    for path in train_image_paths:
        imgs.append(cv2.resize(load_image(path), (200, 200), interpolation=cv2.INTER_NEAREST))
    labels = np.array(train_image_labels)
    features = extract_face_features_train(imgs)
    features, labels = SMOTE().fit_resample(features, labels)
    model = train_classification_model(features, labels)
    return model


def predict_age(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje godina i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati godine.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje godina
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati godine lica
    :return: <Int> Prediktovanu vrednost za goinde  od 0 do 116
    """
    age = 0
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    # img = cv2.resize(find_edges(image_path), (150, 150), interpolation=cv2.INTER_NEAREST)
    try:
        #img = cv2.resize(find_edges(image_path), (500, 500), interpolation=cv2.INTER_NEAREST)
        img = load_image(image_path)
        age = trained_model.predict(extract_face_features_train([extract_face(img)]))
    except Exception:
        age = [28]
    return int(float(age[0]))
    # try:
    #     #img = cv2.resize(load_image(image_path), (150, 150), interpolation=cv2.INTER_NEAREST)
    #     #features = extract_features_age([img])
    #     img = load_image(image_path)
    #     age = trained_model.predict(extract_features_race([img]))
    #     #age = trained_model.predict(features)
    # except Exception:
    #     age = [28]


def predict_gender(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje pola na osnovu lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati pol.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa pola (0 - musko, 1 - zensko)
    """
    gender = 1
    # try:
    #     # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    #     img = cv2.resize(load_image(image_path), (150, 150), interpolation=cv2.INTER_NEAREST)
    #     gender = trained_model.predict(extract_features_gender([img]))
    # except Exception:
    #     gender = [0]

    # img = cv2.resize(load_image(image_path), (150, 150), interpolation=cv2.INTER_NEAREST)
    # hog = hog_desc(img.shape)
    # gender = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    # return gender[0]

    try:
        # img = cv2.resize(load_image(image_path), (150, 150), interpolation=cv2.INTER_NEAREST)
        # img = load_image(image_path)
        # gender = trained_model.predict(extract_features_race([img]))
        # hog = hog_desc(img.shape)
        # race = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
        img = load_image(image_path)
        gender = trained_model.predict(extract_face_features_train([extract_face(img)]))
    except Exception:
        gender = [0]
    return gender[0]

def predict_race(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje rase lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati rasu.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa (0 - Bela, 1 - Crna, 2 - Azijati, 3- Indijci, 4 - Ostali)
    """
    race = 0
    #return random.choice([0, 1, 2, 3, 4])
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    try:
        #img = cv2.resize(load_image(image_path), (150, 150), interpolation=cv2.INTER_NEAREST)
        # img = load_image(image_path)
        # race = trained_model.predict(extract_features_race([img]))
        img = load_image(image_path)
        race = trained_model.predict(extract_face_features_train([extract_face(img)]))
    #hog = hog_desc(img.shape)
    #race = trained_model.predict(reshape_data(np.array([hog.compute(img)])))
    except Exception:
        race = [0]
    return race[0]
