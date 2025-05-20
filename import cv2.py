import cv2
import numpy as np
from matplotlib import pyplot as plt

def add_impulse_noise(image, noise_ratio=0.15):
    """Добавление импульсного шума (salt & pepper) с модифицированными параметрами"""
    noisy = image.copy()
    h, w = image.shape[:2]
    
    # Генерация шума с разными вероятностями для соли и перца
    salt_prob = noise_ratio * 0.6  # 60% шума - соль
    pepper_prob = noise_ratio * 0.4  # 40% шума - перец
    
    # Создаем случайные маски
    salt_mask = np.random.random((h, w)) < salt_prob
    pepper_mask = np.random.random((h, w)) < pepper_prob
    
    # Применяем шум
    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0
    
    return noisy

def adaptive_median_filter(image, max_kernel_size=7):
    """Адаптивный медианный фильтр с переменным размером ядра"""
    h, w = image.shape
    result = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            kernel_size = 3
            while kernel_size <= max_kernel_size:
                half_size = kernel_size // 2
                # Проверка границ
                if i < half_size or i >= h - half_size or j < half_size or j >= w - half_size:
                    break
                
                window = image[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
                median = np.median(window)
                min_val = np.min(window)
                max_val = np.max(window)
                
                # Адаптивная логика
                if min_val < median < max_val:
                    if min_val < image[i,j] < max_val:
                        result[i,j] = image[i,j]
                    else:
                        result[i,j] = median
                    break
                else:
                    kernel_size += 2
            else:
                result[i,j] = median
    
    return result

def improved_canny_edge_detection(image, sigma=1.0):
    """Улучшенный детектор границ Канни с автоматическим подбором порогов"""
    # Размытие по Гауссу
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    
    # Вычисление градиентов
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Вычисление величины и направления градиента
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    
    # Нормализация
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Автоматический подбор порогов
    high_threshold = np.percentile(magnitude, 90)
    low_threshold = high_threshold * 0.5
    
    return cv2.Canny(image, low_threshold, high_threshold)

def percentile_filter(image, kernel_size=3, percentile=50):
    """Перцентильный фильтр (обобщение медианного)"""
    h, w = image.shape
    result = np.zeros_like(image)
    half_size = kernel_size // 2
    
    for i in range(half_size, h - half_size):
        for j in range(half_size, w - half_size):
            window = image[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            result[i,j] = np.percentile(window, percentile)
    
    return result

def main():
    # Загрузка изображения 
    img = cv2.imread('lol.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Ошибка загрузки изображения.")
        return
    
    # Изменение размера для удобства
    img = cv2.resize(img, (600, 400))
    
    # 1. Добавление импульсного шума и фильтрация
    noisy_img = add_impulse_noise(img, noise_ratio=0.2)
    
    # Стандартный медианный фильтр
    median_filtered = cv2.medianBlur(noisy_img, 5)
    
    # Адаптивный медианный фильтр
    adaptive_median = adaptive_median_filter(noisy_img)
    
    # 2. Выделение краев (разные методы)
    canny_edges = improved_canny_edge_detection(img)
    
    # 3. Перцентильная фильтрация с разными параметрами
    percentiles = [25, 50, 75]
    perc_filtered = [percentile_filter(noisy_img, 5, p) for p in percentiles]
    
    # Визуализация результатов
    plt.figure(figsize=(18, 12))
    
    plt.subplot(331), plt.imshow(img, cmap='gray'), plt.title('Оригинальное изображение')
    plt.subplot(332), plt.imshow(noisy_img, cmap='gray'), plt.title('С импульсным шумом (20%)')
    plt.subplot(333), plt.imshow(median_filtered, cmap='gray'), plt.title('Медианный фильтр 5x5')
    
    plt.subplot(334), plt.imshow(adaptive_median, cmap='gray'), plt.title('Адаптивный медианный фильтр')
    plt.subplot(335), plt.imshow(canny_edges, cmap='gray'), plt.title('Улучшенный Канни')
    plt.subplot(336), plt.imshow(perc_filtered[0], cmap='gray'), plt.title('25-й перцентиль')
    
    plt.subplot(337), plt.imshow(perc_filtered[1], cmap='gray'), plt.title('50-й перцентиль (медиана)')
    plt.subplot(338), plt.imshow(perc_filtered[2], cmap='gray'), plt.title('75-й перцентиль')
    
    # Сохранение результатов
    cv2.imwrite('noisy_image.jpg', noisy_img)
    cv2.imwrite('median_filtered.jpg', median_filtered)
    cv2.imwrite('adaptive_median.jpg', adaptive_median)
    cv2.imwrite('canny_edges.jpg', canny_edges)
    for i, p in enumerate(percentiles):
        cv2.imwrite(f'percentile_{p}.jpg', perc_filtered[i])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()