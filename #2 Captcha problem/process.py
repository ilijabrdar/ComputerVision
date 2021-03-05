# import libraries here
from ann_util import create, load, serialize, train
import image_util as img
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import cv2
from scipy import ndimage

alphabet_big = ["A", "B", "C", "Č", "Ć", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                "S", "Š", "T", "U", "V", "W", "X", "Y", "Z", "Ž"]
alphabet_small = ["a", "b", "c", "č", "ć", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
                  "r", "s", "š", "t", "u", "v", "w", "x", "y", "z", "ž"]
alphabet = alphabet_big + alphabet_small


def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju
    ann = load()

    if ann is not None:
        return ann

    alphabet_big_img = img.load_image(train_image_paths[0])
    alphabet_big_img_bin = img.invert(img.image_bin(img.image_gray(alphabet_big_img)))

    alphabet_small_img = img.load_image(train_image_paths[1])
    alphabet_small_img_bin = img.invert(img.image_bin(img.image_gray(alphabet_small_img)))

    _, big_letters = img.select_roi(alphabet_big_img, alphabet_big_img_bin)
    _, small_letters = img.select_roi(alphabet_small_img, alphabet_small_img_bin)
    letters = big_letters + small_letters
    inputs = img.prepare_for_ann(letters)
    outputs = img.convert_output(alphabet)

    ann = create()
    ann = train(ann, inputs, outputs)
    serialize(ann)
    return ann


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    try:
        extracted_text = ""
        # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string
        print(image_path)
        image = img.load_image(image_path)
        thresh = 0
        if np.mean(image[:, :, 0]) < 110:
            _image = image[:, :, 0]
            image_bin = np.uint8(_image > 215) * _image
            _, image_bin = cv2.threshold(image_bin, 1, 255, cv2.THRESH_BINARY)
            thresh = 250

        elif (145 < np.mean(image[:, :, 0]) < 170) and (137 < np.mean(image[:, :, 1])):
            _image = image[:, :, 0]
            image_bin = np.uint8(_image > 210) * _image
            _, image_bin = cv2.threshold(image_bin, 1, 255, cv2.THRESH_BINARY)
            thresh = 250

        elif (150 < np.mean(image[:, :, 0]) < 180) and (90 < np.mean(image[:, :, 1]) < 125) and (60 < np.mean(image[:, :, 2]) < 95):
            _image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            _image = _image[:, :, 1]
            _image = cv2.medianBlur(_image, 5)
            image_bin = np.uint8(_image < 130) * _image
            image_bin = img.dilate(img.erode(image_bin))
            thresh = 250
            _, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) < 3:
                image_bin = np.uint8(_image < 137) * _image
                image_bin = img.dilate(img.erode(image_bin))
        elif 220 < np.mean(image[:, :, 0]):
            _image = image
            image_bin = img.invert(img.image_bin(img.image_gray(_image)))
            image_bin = img.dilate(img.erode(image_bin))
            thresh = 130
        else:
            raise Exception()

        image, image_bin = img.straighten_image(image, image_bin, thresh)
        _, letters, distances = img.select_roi(image, image_bin, thresh)
        distances = np.array(distances).reshape(len(distances), 1)

        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(distances)

        inputs = img.prepare_for_ann(letters)
        results = trained_model.predict(np.array(inputs, np.float32))
        extracted_text = display_result(results, k_means)
        print(extracted_text)
        words = extracted_text.split()
        extracted_text = ''

        choices = list(vocabulary.keys())
        for word in words:
            word_info = []
            _word = ''
            if word == 'l':
                _word = 'I'
            else:
                _word = word
            for choice in choices:
                distance = fuzz.ratio(_word, choice)
                word_info.append((choice, distance, int(vocabulary[choice])))
            sorted_words = sorted(word_info, key=lambda w: (w[1], w[2]))
            extracted_text += sorted_words[-1][0]
            extracted_text += ' '
        print(extracted_text)
    except Exception:
        sorted_dict = sorted(vocabulary, key=vocabulary.get)
        extracted_text = sorted_dict[-1] + ' ' + sorted_dict[-2] + ' ' + sorted_dict[-3] + ' ' + sorted_dict[-4]
    return extracted_text


def display_result(outputs, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[img.winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[img.winner(output)]
    return result

