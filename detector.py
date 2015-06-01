# http://python-gazo.blog.jp/opencv/%E9%A1%94%E6%A4%9C%E5%87%BA

import cv2


def detect_image(image_file):

    image = cv2.imread(image_file)

    face = __detect_face_image(image)

    __show_result(image, face)


def detect_image_with_gray(image_file):

    image = cv2.imread(image_file)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray_image, gray_image)

    face = __detect_face_image(gray_image)

    __show_result(image, face)


def __detect_face_image(image):
    cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    face = cascade.detectMultiScale(image, 1.1, 3)

    return face


def __show_result(image, face):
    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 50, 255), 3)

    cv2.imshow("Show Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
