import cv2
import numpy as np
import os
from color_preprocessing import fix, is_gray_colors

dataset_path = 'dataset'
output_path = 'cleaned_images'
files = ['11.jpg']

# files = os.listdir(dataset_path)

for file in files:
    fn = os.path.join(dataset_path, file)
    src = fix(cv2.imread(fn))
    blur = cv2.GaussianBlur(src, (5, 5), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if not is_gray_colors(blur) else blur
    fn = os.path.join(output_path, file)
    cv2.imwrite(fn.replace('.', '_fix.'), gray)

    dst = cv2.bitwise_not(gray)
    cv2.imwrite(fn.replace('.', '_fix_inv.'), dst)

    height, width = src.shape[:2]
    print(width, height)

    # Нахождение линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 80, None, width//2, 30)

    # нахожу самые широкие линии
    x1, y1, x2, y2, lines_widths = {}, {}, {}, {}, {}
    lines_indexes, lines_y1 = [], []
    if lines is not None:
        idx = 0
        for line in lines:
            x1[idx], y1[idx], x2[idx], y2[idx] = line[0]
            w = abs(x2[idx] - x1[idx])
            h = abs(y2[idx] - y1[idx])
            if w > h and w >= width * 0.6:
                lines_indexes.append(idx)
                lines_widths[idx] = w
                lines_y1.append((idx, y1[idx]))  # индексы и y-координаты начальных точек линий
                print(idx, 'append', w, 'x,y', x1[idx], y1[idx])
                idx += 1

    # сортировка сверху вниз по y-координатам начальных точек
    sorted_lines_y1 = sorted(lines_y1, key=lambda y: y[1])

    # Пересортированные массивы индексов линий и их ширин
    sorted_lines_indexes = [idx for idx, _ in sorted_lines_y1]
    sorted_lines_widths = [lines_widths[idx] for idx, _ in sorted_lines_y1]

    # Вывод отсортированных индексов и ширин линий
    print('Вывод отсортированных индексов и ширин линий')
    for idx, w in zip(sorted_lines_indexes, sorted_lines_widths):
        print('idx:', idx, 'width/y:', w, y1[idx])

    # Из самых длинных линий берем самую верхнюю и самую нижнюю и получаю углы их наклона и углы их ближайших по У
    # соседей, берем средние углы, если они примерно одинаковые, то нужно только повернуть документ, если разные -
    # захватываем сверху и снизу области, равные половине остатка от линии до края - аппроксимируем перспективу
    # вверх и вниз, находим у0 и уМах для обеих линий в точках х=0 и х=Мах - это будут точки искажения перспективы,
    # а точки вписания = средние между у0 и уМах в точках х=0 и х=Мах

    # Определение количества линий для анализа сверху и снизу
    top_line_count = 40  # Количество верхних линий для рассмотрения
    bottom_line_count = 30  # Количество нижних линий для рассмотрения

    # Фильтрация самых верхних и самых нижних линий
    top_lines = sorted_lines_y1[:top_line_count]
    bottom_lines = sorted_lines_y1[-bottom_line_count:]

    # Выбор самой верхней и самой нижней линии из отфильтрованных групп
    top_most_line = top_lines[0]  # Самая верхняя линия
    bottom_most_line = bottom_lines[-1]  # Самая нижняя линия

    # Получение координат для самой верхней и самой нижней линии
    top_line_coords = (x1[top_most_line[1]], y1[top_most_line[1]], x2[top_most_line[1]], y2[top_most_line[1]])
    bottom_line_coords = (
    x1[bottom_most_line[1]], y1[bottom_most_line[1]], x2[bottom_most_line[1]], y2[bottom_most_line[1]])

    print("Координаты самой верхней линии:", top_line_coords)
    print("Координаты самой нижней линии:", bottom_line_coords)

    x_1, x_2, y_1, y_2 = x1[sorted_lines_indexes[idx]], x2[sorted_lines_indexes[idx]], y1[sorted_lines_indexes[idx]], y2[sorted_lines_indexes[idx]]
    # print('LT-idx', idx, x_1, y_1, 'w_max_top', w_max_top)
    #
    # idx = sorted_widths[-20:].index(w_max_bottom)
    # x_3, x_4, y_3, y_4 = x1[wide_lines[idx]], x2[wide_lines[idx]], y1[wide_lines[idx]], y2[wide_lines[idx]]
    # print('RB-idx', idx, x_4, y_4, 'w_max_bottom', w_max_bottom)

    # Определение исходных точек на изображении, которые формируют искаженную перспективу
    # pts1 = np.float32([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]])
    # Определение точек на выходном изображении, куда должны быть перенесены исходные точки
    # y = int(min(y_1, y_2))
    # x = int(min(x_1, x_2))
    # width = int(max(x_2, x_4) - min(x_1, x_3))
    # height = int(max(y_3, y_4) - min(y_1, y_2))
    # print('width, height', width, height)
    # print(y_1, y_2, y_3, y_4)

    # Получение матрицы преобразования перспективы
    # pts2 = np.float32([[x, y], [x+width, y], [x, y+height], [x+width, y+height]])
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    #
    # warped_img = cv2.warpPerspective(src, M, (width, height))
    # cropped_img = warped_img[y:y + height, x:x + width]
    # cv2.imwrite(fn.replace('.', '_geom.'), cropped_img)


