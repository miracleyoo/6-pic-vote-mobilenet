import functools
import os
import random
import shutil

import PIL.Image as Image
import cv2
import numpy as np
from PIL import ImageEnhance
from matplotlib import pyplot as plt


def violent_resize(img, short_len):
    return img.resize((short_len, short_len))


def resize_by_short(img, short_len=128, crop=False):
    """按照短边进行所需比例缩放"""
    (x, y) = img.size
    if x > y:
        y_s = short_len
        x_s = int(x * y_s / y)
        x_l = int(x_s / 2) - int(short_len / 2)
        x_r = int(x_s / 2) + int(short_len / 2)
        img = img.resize((x_s, y_s))
        if crop:
            box = (x_l, 0, x_r, short_len)
            img = img.crop(box)
    else:
        x_s = short_len
        y_s = int(y * x_s / x)
        y_l = int(y_s / 2) - int(short_len / 2)
        y_r = int(y_s / 2) + int(short_len / 2)
        img = img.resize((x_s, y_s))
        if crop:
            box = (0, y_l, short_len, y_r)
            img = img.crop(box)
    return img


def get_center_img(img, short_len=128):
    img = resize_by_short(img, short_len=short_len * 2)
    (x, y) = img.size
    box = (
        x // 2 - short_len * 3 // 4, y // 2 - short_len * 3 // 4, x // 2 + short_len * 3 // 4,
        y // 2 + short_len * 3 // 4)
    img = img.crop(box).resize((short_len, short_len))
    return img


def divide_4_pieces(img, short_len=128, pick=None):
    (x, y) = img.size
    boxs = []
    boxs.append((0, 0, x // 2, y // 2))
    boxs.append((0, y // 2, x // 2, y))
    boxs.append((x // 2, 0, x, y // 2))
    boxs.append((x // 2, y // 2, x, y))
    if pick is not None:
        return img.crop(boxs[pick]).resize((short_len, short_len))
    else:
        imgs = [img.crop(i).resize((short_len, short_len)) for i in boxs]
        return imgs


def get_6_pics(img, short_len=128):
    imgs = []
    imgs.append(violent_resize(img, short_len=short_len))
    imgs.append(get_center_img(img, short_len=short_len))
    imgs.extend(divide_4_pieces(img, short_len=short_len))
    return imgs


def divide_func(index):
    if index == 0:
        return violent_resize
    elif index == 1:
        return get_center_img
    elif 2 <= index <= 5:
        return functools.partial(divide_4_pieces, pick=index - 2)


def div_6_pic(img_path):
    prefix = "./source/temp"
    new_root = os.path.join(prefix, img_path.split("/")[-2])
    shutil.rmtree(prefix)
    os.makedirs(new_root)
    img = Image.open(img_path)
    imgs = get_6_pics(img, short_len=128)
    return imgs


def random_crop(image):
    """
    Make the shorter side equal to 1600 by scaling the img, then randomly
    crop the img to a 1600*1600 square.
    :param image: The matrix of img.
    :return: The cropped img.
    """
    h, w, c = image.shape
    kh, kw = 1600 / h, 1600 / w
    k = max(kh, kw)
    image = cv2.resize(image, (max(int(w * k), 1600), max(1600, int(h * k))))
    h, w, c = image.shape
    dh, dw = (h - 1600) / 2, (w - 1600) / 2
    x = random.randint(int(w / 2 - dw), int(w / 2 + dw))
    y = random.randint(int(h / 2 - dh), int(h / 2 + dh))
    image = image[(y - 800):(y + 800), (x - 800):(x + 800), :]
    return image


def image_show(image):
    """
    Show img by inputting matrix.
    :param image: The matrix of img.
    :return: None
    """
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def image_show_in_jupyter(image):
    """
    Show img in jupyter output block.
    :param image: The matrix of img.
    :return: None.
    """
    image = image[:, :, ::-1]
    plt.imshow(image)


def rotate_and_resize(image, alpha, output_size, beta=1):
    """
    Rotate the square img and resize it to a output_size*output_size square.
    :param image: The matrix of a square card.
    :param alpha: The angle of clockwise rotation.
    :param beta: The ratio of output square side to input square side, except padding.
    :param output_size: The length of output square side, include padding.
    :return: Rotated and resized img matrix.
    """
    size = int(image.shape[0] * beta)
    pad = int(output_size / 2 - size / 2)
    image = cv2.resize(image, (size, size))
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    M = cv2.getRotationMatrix2D((image.shape[0] / 2, image.shape[1] / 2), alpha, 1)
    image = cv2.warpAffine(image, M, image.shape[:2])
    image = cv2.resize(image, (output_size, output_size))
    return image


def get_brightness(image):
    """
    Get the brightness of the img.
    :param image: The matrix of img.
    :return: The brightness of the img.
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(image_hsv[:, :, 2])
    return brightness


def adjust_brightness(image, to_value):
    """
    Adjust the img brightness.
    :param image: The matrix of img.
    :param to_value: The brightness of output img.
    :return: The matrix of img.
    """
    bri = get_brightness(image)
    k = to_value / bri
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    enhance = ImageEnhance.Brightness(image)
    image_brighted = enhance.enhance(k)
    return cv2.cvtColor(np.asarray(image_brighted), cv2.COLOR_RGB2BGR)


def affine(image):
    pts1 = np.float32([[50, 50], [400, 50], [50, 400]])
    rand = np.random.randint(-100, 100, size=pts1.shape)
    M = cv2.getAffineTransform(pts1, (pts1 + rand.astype(np.float32)))
    ret = cv2.warpAffine(image, M, image.shape[:2])
    return ret


def mask_processing(mask):
    mask[mask > 0] = 1
    mask = np.bitwise_or(np.bitwise_or(mask[:, :, 0], mask[:, :, 1]), mask[:, :, 2])
    return np.transpose(np.stack([mask, mask, mask]), [1, 2, 0])


def random_cut(image):
    """
    Cut off a quadrilateral.
    :param image: The matrix of img.
    :return: The matrix of img.
    """
    cut = np.random.choice([True, False], 1, p=[0.4, 0.6])
    if cut:
        alpha = random.randint(1, 4) * 90
        center = (image.shape[0] / 2, image.shape[1] / 2)
        M = cv2.getRotationMatrix2D(center, alpha, 1)
        image = cv2.warpAffine(image, M, image.shape[:2])
        x = random.randint(200, 400)
        y = random.randint(200, 400)
        a1, a2 = (0, x), (y, 0)
        x = random.randint(200, x)
        y = random.randint(200, y)
        a3 = (y, x)
        roi = np.array([[[0, 0], a1, a3, a2]], dtype=np.int32)
        cv2.fillPoly(image, roi, [0, 0, 0])
        M = cv2.getRotationMatrix2D(center, -alpha, 1)
        image = cv2.warpAffine(image, M, image.shape[:2])
    return image


def random_cut2(image):
    """
    Cut off a ellipse.
    :param image: The matrix of img.
    :return: The matrix of img.
    """
    cut = np.random.choice([True, False], 1, p=[0.4, 0.6])
    if cut:
        alpha = random.randint(1, 4) * 90
        center = (image.shape[0] / 2, image.shape[1] / 2)
        M = cv2.getRotationMatrix2D(center, alpha, 1)
        image = cv2.warpAffine(image, M, image.shape[:2])
        pad = round(0.5 * image.shape[0])
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        ellipse_center = (random.randint(0.75 * pad, image.shape[0] - 0.75 * pad),
                          random.randint(image.shape[0] - 1.25 * pad, image.shape[0] - pad))
        ellipse_axis = (random.randint(0.25 * pad, 0.5 * pad), random.randint(0.75 * pad, pad))
        ellipse_alpha = random.randint(30, 150)
        image = cv2.ellipse(image, ellipse_center, ellipse_axis, ellipse_alpha, 0, 360, 0, -1)
        image = image[image.shape[0] - 3 * pad:image.shape[0] - pad, image.shape[0] - 3 * pad:image.shape[0] - pad, :]
        M = cv2.getRotationMatrix2D(center, -alpha, 1)
        image = cv2.warpAffine(image, M, image.shape[:2])
    return image


def motion_blur(image, degree: int = 100, angle: int = 0):
    """
    Blur the input img with motion-blur kernel.
    :param image: The matrix of img.
    :param degree: The degree of blur.
    :param angle: The clockwise degree of blur.
    :return: The blurred img.
    """
    angle += 45
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred_img = np.array(blurred, dtype=np.uint8)
    return blurred_img


def gaussian_blur(image, kx=10, ky=10, sx=0, sy=0):
    return cv2.GaussianBlur(image, ksize=(2 * kx + 1, 2 * ky + 1), sigmaX=sx, sigmaY=sy)


def gaussian_noise(image, percentage=0.05, instead=255):
    """
    Add gaussian noise to the img.
    :param image: The matrix of img.
    :param percentage: The percentage of the instead pixel.
    :param instead: The instead pixel, like (255, 255, 255).
    :return: The img which added noise.
    """
    g_noise_img = image
    g_noise_num = int(percentage * image.shape[0] * image.shape[1])
    for i in range(g_noise_num):
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        g_noise_img[temp_x][temp_y] = [instead, instead, instead]
    return g_noise_img


def img_sharpen(image, kernel_choose=1):
    """
    Sharpen the input img by using convolution kernel.
    :param image: The matrix of img.
    :param kernel_choose: 1、2、others for 3 different kernels.
    :return: Sharpened img.
    """
    if kernel_choose == 1:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif kernel_choose == 2:
        kernel = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    else:
        kernel = np.array([
            [-1, -1, -1, -1, -1],
            [-1, 2, 2, 2, -1],
            [-1, 2, 8, 2, -1],
            [-1, 2, 2, 2, -1],
            [-1, -1, -1, -1, -1]]) / 8.0
    dst = cv2.filter2D(image, -1, kernel)
    return dst


def combine(image, card, alpha=0, beta=1):
    brightness = get_brightness(image)
    card = random_cut(card)
    card = adjust_brightness(card, brightness)
    image = random_crop(image)
    card = rotate_and_resize(card, alpha, beta, 1600)
    card = affine(card)
    x = random.randint(600, 1000)
    y = random.randint(600, 1000)
    card_ = np.zeros_like(card)
    deltax = min(x, 1600 - x)
    deltay = min(y, 1600 - y)
    if x <= 800 and y <= 800:
        card_[0:(y + 800), 0:(x + 800), :] = card[(800 - deltay):1600, (800 - deltax):1600, :]
    elif x <= 800 and y >= 800:
        card_[(y - 800):1600, 0:(x + 800), :] = card[0:(800 + deltay), (800 - deltax):1600, :]
    elif x >= 800 and y >= 800:
        card_[(y - 800):1600, (x - 800):1600, :] = card[0:(800 + deltay), 0:(800 + deltax), :]
    else:
        card_[0:(y + 800), (x - 800):1600, :] = card[(800 - deltay):1600, 0:(800 + deltax), :]
    mask = card_.copy()
    mask = mask_processing(mask)
    res1 = np.multiply(image, (1 - mask)) + card_
    kernel = np.ones((5, 5), np.uint8)
    res = cv2.erode(cv2.dilate(res1, kernel, 1), kernel, 1)
    res = cv2.resize(res, (448, 448))
    return res


def image_generation():
    card_folder = "./card/"
    des_folder = "./generated_images"
    try:
        os.mkdir(des_folder)
    except Exception as e:
        print(e)
    dirs = os.listdir(card_folder)
    card_path_list = []
    for dir in dirs:
        card_path = os.listdir(card_folder + dir)
        card_path_list.append(dir + "/" + card_path[0])
        try:
            os.mkdir(des_folder + "/" + dir)
        except Exception as e:
            pass
    raw_image_list = []
    raw_folder = "./raw_image"
    raw_image_list = os.listdir(raw_folder)
    count = 25000
    for epoch in range(100):
        print("epoch: ", epoch)
        for card_path in card_path_list:
            try:
                index = random.randint(0, len(raw_image_list) - 1)
                image = cv2.imread(raw_folder + "/" + raw_image_list[index])
                card = cv2.imread(card_folder + "/" + card_path)
                alpha = random.randint(0, 360)
                beta = random.randint(50, 100) / 100
                res = combine(image, card, alpha, beta)
                count += 1
                res = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                res = res.resize((448, 448))
                res.save(des_folder + "/" + card_path.split("/")[0] + "/" + str(count) + ".jpg")
            except Exception as e:
                print(e)
    print(raw_image_list)


if __name__ == "__main__":
    image_generation()
