import cv2
import numpy as np
import os
import shutil


def remove_blue1(img):
    # Конвертация изображения в HSV для анализа
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Создание маски для выделения синего цвета
    blue_mask = (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 140)

    # Маска для защиты текста и линий
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edge_mask = cv2.Canny(gray, 100, 200) > 0

    # Объединение масок для защиты
    preservation_mask = np.bitwise_or(text_mask, edge_mask)

    # Применяем маски, чтобы заменить синий цвет на белый, исключая защищенные области
    img[(blue_mask) & (~preservation_mask)] = [255, 255, 255]  # Замена на белый цвет

    return img


def test_4(img):
    lo = np.array([80, 0, 0], dtype=np.uint8)
    hi = np.array([255, 200, 200], dtype=np.uint8)
    mask = cv2.inRange(img, lo, hi)
    mask_inv = cv2.bitwise_not(mask)

    return cv2.bitwise_and(img, img, mask=mask)


def remove_blue2(img):

    denoised = cv2.medianBlur(img, 5)
    hls = cv2.cvtColor(denoised, cv2.COLOR_BGR2HLS)

    # выделение текста и линий
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bg_mask = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite('bg_mask.png', bg_mask)
    edge_mask = cv2.Canny(gray, 200, 255) > 0
    preservation_mask = np.logical_or(bg_mask.astype(bool), edge_mask)

    b, g, r = cv2.split(img)
    mask_r = np.zeros_like(r, dtype=np.uint8)
    mask_g = np.zeros_like(g, dtype=np.uint8)
    mask_r[b > r+10] = 255
    mask_g[b > g+10] = 255
    mask = mask_r + mask_g
    mask[mask > 255] = 255
    mask = cv2.bitwise_or(mask_r, mask_g)
    lower_blue = np.array([60, 70, 80], dtype=np.uint8)
    upper_blue = np.array([180, 230, 255], dtype=np.uint8)
    blue_mask2 = cv2.inRange(hls, lower_blue, upper_blue)
    cv2.imwrite('blue_mask.png', blue_mask2)
    combined_mask = np.logical_xor(preservation_mask, blue_mask2 != 0)
    # Применение маски: замена синего цвета на белый, исключая места с текстом
    img[~combined_mask] = [255, 255, 255]
    return img


def get_only_black(img):
    print('get_only_black')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = int(gray.shape[1] * 11)
    height = int(gray.shape[0] * 11)
    resized_up = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized_up, (17, 17), 0)

    gray = cv2.resize(blurred, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    gray[gray < 141] = 0
    if not is_background_white(gray):
        return img
    gray[gray > 155] = 255
    return gray


def get_only_black_old(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    # enhanced_l = clahe.apply(enhanced_l)
    lab[l & ~enhanced_l] = 255
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    _, l, s = cv2.split(hls)
    img[s > 10] = [255, 255, 255]
    img[l < 135] = [0, 0, 0]
    img[l > 136] = [255, 255, 255]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bg_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img[~bg_mask.astype(bool)] = [255, 255, 255]
    return img


def increase_contrast_clahe(img, clipLimit=2.0, tileGridSize=(8, 8)):
    # Конвертируем BGR - LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Разделяем каналы
    l, a, b = cv2.split(lab)

    # Создаем объект CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    # Применяем CLAHE к каналу L
    cl = clahe.apply(l)

    # Собираем обратно в LAB
    limg = cv2.merge((cl, a, b))

    # Конвертируем обратно в BGR
    final_img = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

    return final_img


def increase_contrast_linear(img):
    # Конвертируем изображение в float32 для предотвращения переполнения
    img_float = img.astype(np.float32)
    # Нормализуем изображение для диапазона [0, 1]
    img_normalized = cv2.normalize(img_float, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Растягиваем гистограмму
    img_contrasted = cv2.convertScaleAbs(img_normalized, alpha=255.0 / (img_normalized.max() - img_normalized.min()),
                                         beta=-255.0 * img_normalized.min() / (
                                                     img_normalized.max() - img_normalized.min()))
    return img_contrasted


def clean_image2(img):
    denoised = cv2.medianBlur(img, 3)
    hls = cv2.cvtColor(denoised, cv2.COLOR_BGR2HLS)

    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    _, bg_mask = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imwrite('bg_mask.png', bg_mask)
    edge_mask = cv2.Canny(gray, 200, 255) > 0
    preservation_mask = np.logical_or(bg_mask.astype(bool), edge_mask)

    lower_blue = np.array([40, 60, 70], dtype=np.uint8)
    upper_blue = np.array([180, 230, 255], dtype=np.uint8)

    blue_mask = cv2.inRange(hls, lower_blue, upper_blue)

    final_mask = np.logical_xor(blue_mask != 0, preservation_mask)
    img[~final_mask] = [255, 255, 255]

    return img


def clean_document_image(img):

    # Можно выбрать либо медианный фильтр, либо билатеральный в зависимости от типа шума
    blur = cv2.medianBlur(img, 9)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cleaned_img = cv2.bilateralFilter(cleaned_img, d=9, sigmaColor=75, sigmaSpace=75)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    cleaned_img = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)

    # clahe = cv2.createCLAHE(clipLimit=14.0, tileGridSize=(3, 3))
    # cleaned_img = clahe.apply(cleaned_img)

    # cv2.imwrite('cleaned_img.png', cleaned_img.astype(np.uint8))

    # blur = cv2.bilateralFilter(cleaned_img, 4, 32, 4)
    # cv2.imwrite('blur_img.png', blur.astype(np.uint8))

    _, bg_mask = cv2.threshold(cleaned_img, 5, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imwrite('bg_mask.png', bg_mask)

    edge_mask = cv2.Canny(gray, 5, 255) > 0
    # cv2.imwrite('edge_mask.png', edge_mask.astype(np.uint8))

    preservation_mask = np.logical_xor(bg_mask.astype(bool), edge_mask)
    # cv2.imwrite('preservation_mask.png', preservation_mask.astype(np.uint8))
    # img[~combined_mask] = [255, 255, 255]
    cleaned_img = np.ones_like(img, dtype=np.uint8) * 255
    cleaned_img[preservation_mask == 1] = img[preservation_mask == 1]

    # Дополнительная очистка с морфологическими операциями
    # kernel = np.ones((1, 1), np.uint8)
    # cleaned_img = cv2.morphologyEx(cleaned_img, cv2.MORPH_OPEN, kernel)

    return cleaned_img


def toGray(numpy_img):
    if not is_gray_colors(numpy_img):
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY)
    return numpy_img


def canny_edge_detection(numpy_img):
    gray = toGray(numpy_img)
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)
    edges = cv2.Canny(blurred, 70, 135)
    return edges


def is_background_white(numpy_img, white_level=0.8):
    gray = toGray(numpy_img)
    white_pixels = np.sum(gray >= 254)
    total_pixels = gray.size
    ratio = round(white_pixels / total_pixels, 4)
    print('white pixels balance:', ratio, '| is valid:', ratio >= white_level)
    return white_pixels / total_pixels >= white_level


def not_noise(numpy_img):
    print('not_noise')
    # Подсчет количества строк с яркостью ниже порогового значения
    gray = toGray(numpy_img)
    dark_points_per_row = np.sum(gray < 50, axis=1)
    total_rows = gray.shape[0]
    threshold = gray.shape[1] * 0.1
    dark_rows = np.sum(dark_points_per_row > threshold)
    print('noise balance: dark_rows / total_rows', dark_rows, total_rows, round(dark_rows / total_rows, 4))
    return dark_rows / total_rows < 0.25


def get_only_text(img):
    print('get_only_text')
    gray = toGray(img.copy())
    cleaned_img = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    blur = cv2.bilateralFilter(cleaned_img, 6, 12, 3)
    tmp = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (255, 9, 255))
    nzmask = cv2.inRange(hsv, (0, 0, 10), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((3, 3)))
    mask = mask & nzmask
    gray[np.where(mask)] = 255
    gray[gray < 100] = 0
    if not is_background_white(gray):
        return img
    img[np.where(mask)] = 255
    return img


def get_only_text_rgb(img):
    print('get_only_text_rgb')
    img_float = img.astype(np.float32)
    x1, x2 = 93, 151
    k = 255/(x2 - x1)
    m = -k * x1  # y = k*x + m
    # Пиксели от 0 до c1 приводятся к 0
    img_float[img_float < x1] = 0
    # Пиксели от c1 до c2 масштабируются линейно от 0 до 255
    mask = (img_float >= x1) & (img_float <= x2)
    # Линейное преобразование
    img_float[mask] = k * img_float[mask] + m
    # Пиксели от c2 до 255 устанавливаются в 255
    img_float[img_float > x2] = 255
    # Преобразование обратно в uint8
    new_img = np.clip(img_float, 0, 255).astype(np.uint8)
    return new_img


    # h, l, s = cv2.split(new_img)
    # blue_mask = np.logical_and(h > 60, s > 0)
    # s[blue_mask] = 0
    # l[blue_mask] = 255
    # new_img = cv2.cvtColor(cv2.merge((h, l, s)), cv2.COLOR_HLS2BGR)

def get_only_text_soft(img):
    print('get_only_text_soft')
    gray = toGray(img)
    cleaned_img = cv2.fastNlMeansDenoising(gray, None, 0, 5, 10)
    blur = cv2.bilateralFilter(cleaned_img, 6, 12, 3)
    tmp = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 150), (255, 5, 255))
    nzmask = cv2.inRange(hsv, (0, 0, 40), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((5, 5)))
    mask = mask & nzmask
    img[np.where(mask)] = 255
    # img = gamma_correction(img)
    img[gray < 100] = 0
    return img


def clear_noise(img):
    # добавить к маске очистки от фона и шума маску синего и вычесть маску текста и линий
    if is_gray_colors(img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cleaned_img = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    _, bg_mask = cv2.threshold(cleaned_img, 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img[np.where(~bg_mask)] = 255
    return img


def is_gray_colors(img):
    if len(img.shape) < 3:
        return True  # Изображение в оттенках серого
    elif img.shape[2] == 1:
        return True  # Изображение в оттенках серого
    else:
        return False  # Цветное изображение


def clear_spot(img):
    img = toGray(img)
    # Повышение контраста и проверка на деление на ноль
    img_float = img.astype(np.float32)
    min_val, max_val = np.min(img_float), np.max(img_float)
    if max_val - min_val > 0:
        img_normalized = cv2.normalize(img_float, None, alpha=0, beta=2.7, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_contrasted = cv2.convertScaleAbs(img_normalized, alpha=255.0 / (max_val - min_val),
                                             beta=-255.0 * min_val / (max_val - min_val))
    else:
        # Если изображение уже имеет одинаковые значения, пропускаем нормализацию
        img_contrasted = cv2.convertScaleAbs(img_float)

    # Бинаризация с использованием метода Оцу для выделения текста
    _, bin_img = cv2.threshold(img_contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return bin_img


def gamma_correction(img, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def normalization(img):
    img = toGray(img)
    cleaned_img = cv2.convertScaleAbs(img, alpha=1.0, beta=0)
    # img_contrasted = gamma_correction(cleaned_img, gamma=0.3)
    img_float = cleaned_img.astype(np.float32)
    img_normalized = cv2.normalize(img_float, None, alpha=0, beta=2.7, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_contrasted = cv2.convertScaleAbs(img_normalized, alpha=255.0 / (img_normalized.max() - img_normalized.min()),
                                         beta=-255.0 * img_normalized.min() / (
                                                     img_normalized.max() - img_normalized.min()))
    # _, bin_img = cv2.threshold(img_contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_contrasted


def clear_blue(img):
    print('clear_blue')
    # добавить к маске очистки от фона и шума маску синего и вычесть маску текста и линий
    if is_gray_colors(img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    denoised = cv2.medianBlur(img, 3)
    hls = cv2.cvtColor(denoised, cv2.COLOR_BGR2HLS)

    # выделение текста и линий
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cleaned_img = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    _, bg_mask = cv2.threshold(cleaned_img, 2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # выделение синего
    lower_blue = np.array([50, 30, 80], dtype=np.uint8)
    upper_blue = np.array([180, 230, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hls, lower_blue, upper_blue)

    # вычитание масок
    combined_mask = np.logical_and(bg_mask.astype(bool), blue_mask == 0)
    img[np.where(~combined_mask)] = 255
    return img


def clear_color(img, color='b'):
    print('clear_color')
    if is_gray_colors(img):
        return img

    def implement_color(arr, mask, lim):
        for col in arr:
            c[col][mask >= 1] = lim
        return arr

    b, g, r = cv2.split(img)
    c = {'b': b, 'g': g, 'r': r}

    colors = list(c.keys() - color)
    sum_arr = c[colors[0]] + c[colors[1]]
    mean_arr = sum_arr // 2

    mask_white = ((c[color] >= c[colors[1]] + 10) & (c[color] >= c[colors[0]] + 10) & (c[color] > 30) & (mean_arr >= 30)).astype(np.uint8)
    c = implement_color(c, mask_white, 255)

    mean = np.mean(sum_arr) // 2
    mask_black = ((c[color] >= c[colors[1]] + 10) & (c[color] >= c[colors[0]] + 10) & (c[color] <= 130) & (mean_arr < 40)).astype(np.uint8)
    c = implement_color(c, mask_black, mean)

    return cv2.merge([c['b'], c['g'], c['r']])


def fix(image):
    image = clear_color(image, 'b')
    image_copy = image.copy()
    image_copy = get_only_text_soft(image_copy)

    if not is_background_white(image_copy):
        print('get_only_text')
        image = get_only_text(image)

    else:
        print('clear_blue & soft')
        image = clear_blue(image)
        image = get_only_text_soft(image)

    return image
