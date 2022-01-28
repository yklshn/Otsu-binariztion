import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as pl
import time

start_time = time.time()

img = cv2.imread('images/image030-57.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# pl.show()

img_filter = cv2.bilateralFilter(gray, 11, 15, 15) #убираем шум на изображении
edges = cv2.Canny(img_filter, 30, 200) #алгоритм поиска контуров изображения
cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #находим контуры изображения
cont = imutils.grab_contours(cont) #считываем и сортируем контуры изображения описывающие номерной знак
cont = sorted(cont, key = cv2.contourArea, reverse = True) #в обратном порядке сортируем контуры

# pos = None #перебираем контуры и находим похожие на номерной знак
# for c in cont:
#     approx = cv2.approxPolyDP(c, 15, True)
#     if len(approx) == 4:
#         pos = approx
#         break
# mask = np.zeros(gray.shape, np.uint8)
# new_img = cv2.drawContours(mask, [pos], 0, 255, -1) #рисуем контуры
# bitwise_img = cv2.bitwise_and(img, img, mask=mask) #побитовое вычисление
#
# (x, y) = np.where(mask==255) #выделение белых пикселей
# (x1, y1) = np.min(x), np.min(y) #нахождение минимальных элементов координаты x, y
# (x2, y2) = np.max(x), np.max(y) #нахождение максимальных элементов координаты x, y
# crop = gray[x1:x2, y1:y2]
#
# text = easyocr.Reader(['en']) #чтение номерного знака
# text = text.readtext(crop)
# print(text)
#
# res = text[0][-2]
# final_image = cv2.putText(img, res, (x1 - 200, y2 + 160), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)


pl.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
print("--- %s seconds ---" % (time.time() - start_time))
pl.show()