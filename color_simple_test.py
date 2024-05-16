import cv2
from cv2 import bitwise_not as invert
import numpy as np
import os
from color_preprocessing import (fix, is_gray_colors, get_only_text, get_only_text_soft, clear_color,
                                 is_background_white, gamma_correction, get_only_black, get_only_black_old,
                                 get_only_text_rgb, clear_blue, not_noise, canny_edge_detection, toGray)
from perspective_normalization import perspectiveCorrection, getFrame

dataset_path = 'dataset'
output_path = 'cleaned_images'
file = 'invoice_1.png'

fn = os.path.join(dataset_path, file)
print('--------------------------------------\n', fn)
src = cv2.imread(fn)
orig_img = src.copy()

src = clear_blue(src)
# src = get_only_black(src)
# src = get_only_text(src)
# src = get_only_text_rgb(src)
# src = get_only_text_soft(src)
# src = get_only_text(src)

# is_background_white(src)
# src = clear_color(src)
# src = getFrame(toGray(src))
# print('edge>>>')
# edge = canny_edge_detection(src)
# inv_edge = cv2.bitwise_not(edge)

# cv2.imwrite(fn.replace('.', '_edge.'), inv_edge)


p = perspectiveCorrection(src)

fn = os.path.join(output_path, file)
cv2.imwrite(fn.replace('.', '_p.'), p)
cv2.imwrite(fn.replace('.', '_test.'), src)
