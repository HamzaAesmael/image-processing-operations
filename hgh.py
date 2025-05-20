import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_enhanced_fourier_transform(image):
    """
    Улучшенное вычисление спектра Фурье с дополнительной обработкой
    
    Args:
        image: входное изображение (grayscale)
    
    Returns:
        tuple: (сдвинутый Фурье-образ, спектр магнитуд, фазовая информация)
    """
    # Применяем оконную функцию для уменьшения краевых эффектов
    rows, cols = image.shape
    window = np.outer(np.hanning(rows), np.hanning(cols))
    windowed_img = image * window
    
    # Вычисляем Фурье-преобразование
    f_transform = np.fft.fft2(windowed_img)
    f_shift = np.fft.fftshift(f_transform)
    
    # Вычисляем спектр магнитуд (логарифмический)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-10)
    
    # Вычисляем фазовый спектр
    phase_spectrum = np.angle(f_shift)
    
    return f_shift, magnitude_spectrum, phase_spectrum

def inverse_fourier_transform_enhanced(f_shift):
    """
    Улучшенное обратное преобразование Фурье
    
    Args:
        f_shift: сдвинутый Фурье-образ
    
    Returns:
        восстановленное изображение
    """
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Нормализация для 8-битного изображения
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return img_back

def create_gaussian_filter(shape, cutoff, high_pass=False):
    """
    Создает гауссовский фильтр (вместо идеального)
    
    Args:
        shape: размеры изображения (rows, cols)
        cutoff: частота среза
        high_pass: если True, создает ФВЧ
    
    Returns:
        маска фильтра
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Создаем гауссовский фильтр
    x = np.linspace(-0.5, 0.5, cols) * cols
    y = np.linspace(-0.5, 0.5, rows) * rows
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    
    if high_pass:
        mask = 1 - np.exp(-(radius**2) / (2 * (cutoff**2)))
    else:
        mask = np.exp(-(radius**2) / (2 * (cutoff**2)))
    
    return mask

def apply_band_pass_filter(f_shift, low_cut, high_cut):
    """
    Полосовой фильтр на основе гауссовских ФНЧ и ФВЧ
    
    Args:
        f_shift: сдвинутый Фурье-образ
        low_cut: нижняя частота среза
        high_cut: верхняя частота среза
    
    Returns:
        tuple: (отфильтрованный спектр, маска)
    """
    rows, cols = f_shift.shape
    
    # Создаем ФНЧ для верхней частоты
    lpf = create_gaussian_filter((rows, cols), high_cut, False)
    
    # Создаем ФВЧ для нижней частоты
    hpf = create_gaussian_filter((rows, cols), low_cut, True)
    
    # Комбинируем для получения полосового фильтра
    mask = lpf * hpf
    
    # Применяем фильтр
    f_shift_filtered = f_shift * mask
    
    return f_shift_filtered, mask

def visualize_results(original, spectrum, filtered_images, filtered_spectra, masks):
    """
    Визуализация результатов с улучшенным оформлением
    
    Args:
        original: оригинальное изображение
        spectrum: спектр Фурье
        filtered_images: словарь отфильтрованных изображений
        filtered_spectra: словарь отфильтрованных спектров
        masks: словарь масок
    """
    plt.figure(figsize=(18, 12))
    
    # Оригинал и спектр
    plt.subplot(3, 4, 1), plt.imshow(original, cmap='gray')
    plt.title('Оригинальное изображение'), plt.axis('off')
    
    plt.subplot(3, 4, 2), plt.imshow(spectrum, cmap='jet')
    plt.title('Спектр Фурье (логарифм магнитуд)'), plt.axis('off')
    plt.colorbar()
    
    # ФВЧ
    plt.subplot(3, 4, 3), plt.imshow(masks['HPF_30'], cmap='gray')
    plt.title('Маска ФВЧ (σ=30)'), plt.axis('off')
    
    plt.subplot(3, 4, 4), plt.imshow(filtered_images['HPF_30'], cmap='gray')
    plt.title('ФВЧ (σ=30)'), plt.axis('off')
    
    # ФНЧ
    plt.subplot(3, 4, 5), plt.imshow(masks['LPF_60'], cmap='gray')
    plt.title('Маска ФНЧ (σ=60)'), plt.axis('off')
    
    plt.subplot(3, 4, 6), plt.imshow(filtered_images['LPF_60'], cmap='gray')
    plt.title('ФНЧ (σ=60)'), plt.axis('off')
    
    # Полосовой фильтр
    plt.subplot(3, 4, 7), plt.imshow(masks['BPF_30_60'], cmap='gray')
    plt.title('Маска полосового фильтра (30-60)'), plt.axis('off')
    
    plt.subplot(3, 4, 8), plt.imshow(filtered_images['BPF_30_60'], cmap='gray')
    plt.title('Полосовой фильтр (30-60)'), plt.axis('off')
    
    # Дополнительная визуализация спектров
    plt.subplot(3, 4, 9), plt.imshow(filtered_spectra['HPF_30'], cmap='jet')
    plt.title('Спектр после ФВЧ'), plt.axis('off')
    plt.colorbar()
    
    plt.subplot(3, 4, 10), plt.imshow(filtered_spectra['LPF_60'], cmap='jet')
    plt.title('Спектр после ФНЧ'), plt.axis('off')
    plt.colorbar()
    
    plt.subplot(3, 4, 11), plt.imshow(filtered_spectra['BPF_30_60'], cmap='jet')
    plt.title('Спектр после полосового фильтра'), plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def main():
    # Загрузка нового изображения (текстура дерева)
    img_path = 'virus_1.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Ошибка загрузки изображения {img_path}")
        return
    
    # Изменение размера и улучшение контраста
    img = cv2.resize(img, (512, 512))
    img = cv2.equalizeHist(img)
    
    # Вычисление спектра Фурье
    f_shift, magnitude_spectrum, _ = compute_enhanced_fourier_transform(img)
    
    # Параметры фильтрации
    hpf_cutoff = 30
    lpf_cutoff = 60
    bpf_low = 30
    bpf_high = 60
    
    # Применение фильтров
    # ФВЧ
    hpf_mask = create_gaussian_filter(img.shape, hpf_cutoff, True)
    f_shift_hpf = f_shift * hpf_mask
    img_hpf = inverse_fourier_transform_enhanced(f_shift_hpf)
    
    # ФНЧ
    lpf_mask = create_gaussian_filter(img.shape, lpf_cutoff, False)
    f_shift_lpf = f_shift * lpf_mask
    img_lpf = inverse_fourier_transform_enhanced(f_shift_lpf)
    
    # Полосовой фильтр
    f_shift_bpf, bpf_mask = apply_band_pass_filter(f_shift.copy(), bpf_low, bpf_high)
    img_bpf = inverse_fourier_transform_enhanced(f_shift_bpf)
    
    # Подготовка данных для визуализации
    filtered_images = {
        'HPF_30': img_hpf,
        'LPF_60': img_lpf,
        'BPF_30_60': img_bpf
    }
    
    filtered_spectra = {
        'HPF_30': 20 * np.log(np.abs(f_shift_hpf) + 1e-10),
        'LPF_60': 20 * np.log(np.abs(f_shift_lpf) + 1e-10),
        'BPF_30_60': 20 * np.log(np.abs(f_shift_bpf) + 1e-10)
    }
    
    masks = {
        'HPF_30': hpf_mask,
        'LPF_60': lpf_mask,
        'BPF_30_60': bpf_mask
    }
    
    # Визуализация
    visualize_results(img, magnitude_spectrum, filtered_images, filtered_spectra, masks)
    
    # Сохранение результатов
    cv2.imwrite('original.jpg', img)
    cv2.imwrite('spectrum.jpg', cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    cv2.imwrite('hpf_30.jpg', img_hpf)
    cv2.imwrite('lpf_60.jpg', img_lpf)
    cv2.imwrite('bpf_30_60.jpg', img_bpf)

if __name__ == "__main__":
    main()