import cv2
import os
from perspective_normalization import perspectiveCorrection, applyPerspectiveCorrection, endswithPNG
from color_preprocessing import (get_only_text, get_only_text_soft, clear_color,
                                 is_background_white, get_only_black,
                                 get_only_text_rgb, clear_blue, canny_edge_detection)

dataset_path = 'dataset'
output_path = 'cleaned_images'

# files = os.listdir(dataset_path)
# files = [f for f in os.listdir(dataset_path) if '.j' in f]
files = ['06.jpg']

used_funcs = []
for file in files:
    fn = os.path.join(dataset_path, file)
    print('\n', '--------------------------------------\n', fn)
    src = cv2.imread(fn)
    copy_src = src.copy()

    img_p, matrix, size = perspectiveCorrection(src)
    if img_p is None:
        src = clear_blue(copy_src)
        img_p, matrix, size = perspectiveCorrection(src)
        if img_p is None:
            print('Сложное выравнивание:(')

    src = clear_blue(copy_src)
    used_funcs.append(clear_blue.__name__)

    base = src.copy()
    if not is_background_white(base):
        src = get_only_text_soft(copy_src)
        if not is_background_white(src, 0.91):
            src = get_only_text_rgb(copy_src)
            if not is_background_white(src, 0.91):
                src = get_only_black(copy_src)
                if not is_background_white(src, 0.91):
                    src = get_only_text(base)
                    if is_background_white(src):
                        used_funcs.append(get_only_text.__name__)
                else:
                    used_funcs.append(get_only_black.__name__)
            else:
                used_funcs.append(get_only_text_soft.__name__)
        else:
            used_funcs.append(get_only_text_rgb.__name__)
    else:
        print('clear_blue is ok')
        copy_src = get_only_text_soft(base)
        if not is_background_white(copy_src, 0.91):
            copy_src = get_only_text_rgb(base)
            if not is_background_white(copy_src, 0.91):
                src = get_only_text(base)
                if is_background_white(src):
                    used_funcs.append(get_only_text.__name__)
            else:
                src = copy_src
                used_funcs.append(get_only_text_rgb.__name__)
        else:
            src = copy_src
            used_funcs.append(get_only_text_soft.__name__)

    src = clear_color(src)
    fn = os.path.join(output_path, file)
    fn = endswithPNG('_c', fn)
    cv2.imwrite(fn, src)

    fn_p = endswithPNG('_p', fn)
    if matrix is not None:
        img_p = applyPerspectiveCorrection(src, matrix, size)
        cv2.imwrite(fn_p, img_p)
    else:
        if img_p is not None:
            #  => матрица не вычислена, потому что исходный документ не требует
            print('Чистый документ не требует выравнивания - сохраняю')
            cv2.imwrite(fn_p, src)

    # print('\nLast operation:\n', fn, '>>>>>>>>', used_funcs[-1])

# used_funcs.append(clear_color.__name__)
# used_funcs.append(is_background_white.__name__)

modul_name = 'color_preprocessing'
with (open(__file__, 'r', encoding='utf-8') as script):
    imported = script.read().split(f'from {modul_name} import (')[1].split(')')[0]
    imported_funcs = imported.replace('\n', '').replace(' ', '').split(',')

print('Used funcs:\n', set(used_funcs))
print('Not used funcs:\n', set(imported_funcs) - set(used_funcs))

