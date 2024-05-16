import cv2
import numpy as np
from color_preprocessing import toGray


def getFrame(img):
    # Преобразование изображения в массив NumPy
    np_img = np.array(img)
    # Определение границ текста на холсте
    rows = np.any(np_img < 235, axis=1)
    cols = np.any(np_img < 235, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    # Обрезка холста по границам текста
    trimmed_img = np_img[min_row:max_row + 1, min_col:max_col + 1]
    # Преобразование обрезанного изображения в целочисленный тип, затем в PIL Image
    img = trimmed_img.astype(np.uint8)
    return img


def orderPoints(pts):
    # Инициализируем список координат, который будет упорядочен
    # таким образом: [верхний левый, верхний правый, нижний правый, нижний левый]
    rect = np.zeros((4, 2), dtype="float32")

    # Верхний левый угол будет иметь наименьшую сумму, тогда как
    # нижний правый угол будет иметь наибольшую сумму
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Теперь, вычисляем разницу между точками,
    # верхний правый угол будет иметь наименьшую разницу,
    # нижний левый будет иметь наибольшую разницу
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Возвращаем упорядоченные координаты
    return rect


def getAVGSides(rect):
    # Вычисление длин сторон
    top_edge_length = np.linalg.norm(rect[0] - rect[1])
    bottom_edge_length = np.linalg.norm(rect[2] - rect[3])
    left_edge_length = np.linalg.norm(rect[0] - rect[3])
    right_edge_length = np.linalg.norm(rect[1] - rect[2])

    # Вычисление средних значений
    w = (top_edge_length + bottom_edge_length) / 2
    h = (left_edge_length + right_edge_length) / 2

    return int(w), int(h)


def resizeIMG(img, scaling_factor=0.97, border_color=(128, 128, 128)):
    # Исходные размеры
    original_height, original_width = img.shape[:2]

    # Новые размеры
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Изменение размера изображения
    resized_img = cv2.resize(img, (new_width, new_height))

    # Создание нового изображения с исходными размерами и цветом фона
    new_img = np.full((original_height, original_width, 3), border_color, dtype=np.uint8)

    # Вычисление позиции для вставки уменьшенного изображения
    x_offset = (original_width - new_width) // 2
    y_offset = (original_height - new_height) // 2

    # Вставка уменьшенного изображения в новое
    new_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img

    return new_img


# def calculate_average_angles(approx):
#     # Предполагаем, что approx - это четырехугольник, аппроксимированный cv2.approxPolyDP
#     slopes = [calculate_angle(approx[i][0], approx[(i + 1) % 4][0]) for i in range(4)]
#
#     # Конвертация углов наклона в градусы и корректировка относительно горизонтали
#     angles = [math.degrees(slope) for slope in slopes]
#     angles = [(angle + 360) % 180 for angle in angles]  # Приведение углов к диапазону [0, 180)
#
#     # Вычисление среднего угла для "горизонтальных" сторон (должно быть близко к 0 или 180)
#     avg_horizontal_angle = np.mean([angles[0], angles[2]])
#     # Вычисление среднего угла для "вертикальных" сторон (должно быть близко к 90)
#     avg_vertical_angle = np.mean([angles[1], angles[3]])
#
#     return avg_horizontal_angle, avg_vertical_angle

def calculate_average_angles(rect):
    # Упорядочивание точек: [верхняя левая, верхняя правая, нижняя правая, нижняя левая]
    tl, tr, br, bl = rect

    # Горизонтальные углы (верхняя и нижняя стороны)
    top_angle_deg = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    bottom_angle_deg = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    # Вертикальные углы (левая и правая стороны)
    left_angle_deg = np.degrees(np.arctan2(bl[1] - tl[1], bl[0] - tl[0]))
    right_angle_deg = np.degrees(np.arctan2(br[1] - tr[1], br[0] - tr[0]))

    # Средние углы
    avg_h = (top_angle_deg + bottom_angle_deg) / 2
    avg_v = (left_angle_deg + right_angle_deg) / 2

    # Нормализация средних углов к ближайшей оси
    avg_h = abs(avg_h - avg_h//90*90)
    avg_h -= 90*int(avg_h > 45)
    avg_v = abs(avg_v - avg_v//90*90)
    avg_v -= 90*int(avg_v > 45)

    return avg_h, avg_v


def perspectiveCorrection(src_numpy_img):
    perspectiveCorrection.i = 0 if not hasattr(perspectiveCorrection, 'i') else perspectiveCorrection.i + 1

    # numpy_img = resizeIMG(src_numpy_img)
    numpy_img = src_numpy_img
    h, w, _ = src_numpy_img.shape
    # Конвертация в градации серого и применение фильтра Canny для детектирования границ

    gray = toGray(numpy_img)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    tink = cv2.erode(binary, kernel, iterations=1)
    # edged = cv2.Canny(tink, 0, 255, apertureSize=3)
    sigma = 0.33
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(tink, lower, upper)

    # Нахождение контуров
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Перебор контуров и аппроксимация форм
    quadrilateral_contours = []
    view_img = numpy_img.copy()
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            quadrilateral_contours.append(approx)

            cv2.drawContours(view_img, [approx], -1, (0, 255, 0), 2)
            cv2.polylines(view_img, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

    resized_img = cv2.resize(view_img, (int(w*900/h), 900))
    cv2.imshow("Detected Quadrilaterals", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'{perspectiveCorrection.i}_tmp.png', view_img)

    # Если не найдено ни одного подходящего контура
    if not quadrilateral_contours:
        print("Не найдены контуры, аппроксимируемые четырехугольником.")
        return None, None, None

    # Выбор контура с наибольшей площадью
    largest_contour = max(quadrilateral_contours, key=cv2.contourArea)
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    # Расчет углов
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    rect = orderPoints(approx.reshape(4, 2))
    avg_horizontal_angle, avg_vertical_angle = calculate_average_angles(rect)

    print(f"Средний горизонтальный угол: {avg_horizontal_angle}")
    print(f"Средний вертикальный угол: {avg_vertical_angle}")

    # принимаем решение о необходимости коррекции перспективы
    if not(abs(avg_horizontal_angle) > 5 or abs(avg_vertical_angle) > 0.2):
        print("Коррекция перспективы не требуется.")
        return src_numpy_img, None, None

    print("Требуется коррекция перспективы.")
    # Вычисляем матрицу преобразования и применяем преобразование перспективы
    width, height = getAVGSides(rect)
    print(width, height, 'from', w, h)
    if w - width > w*0.3 or h - height > h*0.3:
        print('Неверно найдены координаты искажения!')
        return None, None, None

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    print('matrix', matrix)
    warped = cv2.warpPerspective(numpy_img, matrix, (width, height))
    result = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
    return result, matrix, (width, height)


def applyPerspectiveCorrection(src_numpy_img, matrix, size):
    width, height = size
    numpy_img = cv2.cvtColor(resizeIMG(src_numpy_img), cv2.COLOR_BGR2RGB)
    warped = cv2.warpPerspective(numpy_img, matrix, (width, height))
    result = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
    return result


def searchPerspectiveMore(numpy_img):
    # найти горизонтальные линии, взять 10 первых линий, посчитать углы наклона, посчитать средний у тех, где он
    # больше 0.2, вычислить у верхних углов и то же самое сделать с нижними линиями, создать rect, если углы~=0
    return None


def endswithPNG(suffix, fn):
    if '.' not in fn: return fn
    return fn.replace('.', suffix+'.').split('.')[-2]+'.png'

# approx = cv2.approxPolyDP(largest_contour, epsilon, True)
# angles = []
# for i in range(4):
#     angle = calculate_angle(approx[i][0], approx[(i + 1) % 4][0])
#     angles.append(angle)
# Горизонтальные углы (между верхней левой/правой и нижней левой/правой)
# horizontal_angles = [angles[0], angles[2]]
# Вертикальные углы (между верхней левой/нижней левой и верхней правой/нижней правой)
# vertical_angles = [angles[1], angles[3]]

# Средние углы для горизонтальных и вертикальных граней
# avg_horizontal_angle = sum(horizontal_angles) / len(horizontal_angles)
# avg_vertical_angle = sum(vertical_angles) / len(vertical_angles)
