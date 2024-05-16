import cv2
import numpy as np


def sobel_process(gray):
    # Применение фильтра Собеля для выделения границ
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)

    # Нормализация градиентного изображения для конвертации в CV_8U
    sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    sobel_8u = np.uint8(sobel_norm)

    # Бинаризация результата
    _, binary = cv2.threshold(sobel_8u, 2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Нахождение контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Предполагаем, что наибольший контур - это контур документа
    if contours:
        # Сортировка контуров по убыванию площади
        print('Sobel process!')
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        document_contour = contours[0]

        # Аппроксимация контура четырехугольником
        perimeter = cv2.arcLength(document_contour, True)
        approx = cv2.approxPolyDP(document_contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            # Углы найдены, можно вырезать документ
            # Подготовка точек для преобразования перспективы
            pts1 = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            width = max(np.linalg.norm(pts1[0] - pts1[3]), np.linalg.norm(pts1[1] - pts1[2]))
            height = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
            print('pts1', pts1)

            print('approx', width, height)
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

            # Преобразование перспективы
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(gray, matrix, (int(width), int(height)))

            return True, result

    return False, None
