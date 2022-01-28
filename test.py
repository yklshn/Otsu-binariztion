import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

start_time = time.time()


img = cv2.imread("images/image030-57.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
row, col = np.shape(gray)
hist_dist = 256 * [0]
# Вычисление частоты каждого пикселя в изображении
for i in range(row):
    for j in range(col):
        hist_dist[gray[i, j]] += 1
# Нормализация частот для получения вероятностей
hist_dist = [c / float(row * col) for c in hist_dist]
plt.plot(hist_dist)


# Вычисление межсегментной дисперсии
def var_c1_c2_func(hist_dist, t):
    u1, u2, p1, p2, u = 0, 0, 0, 0, 0
    for i in range(t + 1):
        u1 += hist_dist[i] * i
        p1 += hist_dist[i]
    for i in range(t + 1, 256):
        u2 += hist_dist[i] * i
        p2 += hist_dist[i]
    for i in range(256):
        u += hist_dist[i] * i
    var_c1_c2 = p1 * (u1 - u) ** 2 + p2 * (u2 - u) ** 2
    return var_c1_c2


# Итеративный проход по всем значениям интенсивности пикселей
# в интервале от 0 до 255 и выбор значения, максимизирующего
# дисперсию
variance_list = []
for i in range(256):
    var_c1_c2 = var_c1_c2_func(hist_dist, i)
    variance_list.append(var_c1_c2)
# Извлечение порогового значения, , максимизирующего дисперсию
t_hat = np.argmax(variance_list)
# Вычисление сегментированного изображения на основе
# порогового значения t_hat
gray_recons = np.zeros((row, col))
for i in range(row):
    for j in range(col):
        if gray[i, j] <= t_hat:
            gray_recons[i, j] = 255
    else:
        gray_recons[i, j] = 0
plt.imshow(gray_recons, cmap='gray')
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
#print("--- %s seconds ---" % (time.time() - start_time))
#
# img_filter = cv2.bilateralFilter(gray, 11, 15, 15) #убираем шум на изображении
# edges = cv2.Canny(img_filter, 30, 200) #находим контуры изображения
# cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #находим контуры изображения
# cont = imutils.grab_contours(cont) #считываем и сортируем контуры изображения описывающие номерной знак
# cont = sorted(cont, key = cv2.contourArea, reverse = True) #в обратном порядке сортируем контуры

# pos = None #перебираем контуры и находим похожие на номерной знак
# for c in cont:
#     approx = cv2.approxPolyDP(c, 15, True)
#     if len(approx) == 4:
#         pos = approx
#         break
# mask = np.zeros(gray.shape, np.uint8)
# new_img = cv2.drawContours(mask, [pos], 0, 255, -1) #рисуем контуры
# bitwise_img = cv2.bitwise_and(img, img, mask=mask) #побитовая операция
#
# (x, y) = np.where(mask==255) #выделение белых пикселей
# (x1, y1) = np.min(x), np.min(y) #нахождение минимальных элементов координаты x, y
# (x2, y2) = np.max(x), np.max(y) #нахождение максимальных элементов координаты x, y
# crop = gray[x1:x2, y1:y2]
#
# text = easyocr.Reader(['ru']) #чтение номерного знака
# text = text.readtext(crop)
# print(text)
#
# res = text[0][-2]
# final_image = cv2.putText(img, res, (x1 + 100, y2 + 1000), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)


# pl.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
# pl.show()
