import cv2
import numpy as np
import os
from color_preprocessing import (fix, is_gray_colors, get_only_text, get_only_text_soft, clear_color,
                                 is_background_white, gamma_correction, get_only_black, get_only_black_old)
from sobel_morph import sobel_process

dataset_path = 'dataset'
output_path = 'cleaned_images'
files = ['10.jpg']

# files = os.listdir(dataset_path)

for file in files:
    fn = os.path.join(dataset_path, file)

    src = cv2.imread(fn)
    # src = clear_color(src)
    if not is_background_white(src):
        print('get_only_black')
        src = get_only_black(src)
    else:
        src = get_only_text_soft(src)

    # blur = cv2.GaussianBlur(src, (5, 5), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if not is_gray_colors(src) else src

    fn = os.path.join(output_path, file)
    cv2.imwrite(fn.replace('.', '_fix.'), gray)

    dst = cv2.bitwise_not(gray)
    cv2.imwrite(fn.replace('.', '_fix_inv.'), dst)
    flag, first = sobel_process(gray)
    if flag:
        cv2.imwrite(fn.replace('.', '_sobel.'), first)

    # Конвертация в цветное изображение
    color_dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)



    height, width = src.shape[:2]
    print(width, height)

    # Нахождение линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 80, None, width//2, 30)

    # Рисование найденных линий
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(color_dst, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(fn.replace('.', '_geom.'), color_dst)


