import easyocr
import cv2

import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

reader = easyocr.Reader(['ru', 'en'], gpu=True)
dataset_path = 'dataset'
output_path = 'cleaned_images'
fn = output_path+'\\_invoice_9_p.png'
gray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
results = reader.readtext(gray)

r = None
for result in results:
    print(result[1])  # Вывод распознанного текста
    r = result
