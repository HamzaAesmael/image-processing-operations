import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    # 1. Loading and Preparing the Image
    img = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: The image could not be uploaded 'building.jpg'")
        print("Please make sure that the file exists in the same directory as the script.")
        return
    
    # resizes the image
    img = cv2.resize(img, (600, 400))
    
    # 2. Edge Detection using Sobel Filter
    # Vertical edges
    #cv2.Sobel(): Applies the Sobel operator.
    #64F means 64-bit floating-point numbers
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # Horizontal edges
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # Combined result
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalization for display
    #cv2.normalize(): This function scales the pixel values.
    sobel_x_norm = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sobel_y_norm = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sobel_combined_norm = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Displaying results
    plt.figure(figsize=(15,5))
    plt.subplot(131), plt.imshow(sobel_x_norm, cmap='gray'), plt.title('(Sobel X)')
    plt.subplot(132), plt.imshow(sobel_y_norm, cmap='gray'), plt.title(' (Sobel Y)')
    plt.subplot(133), plt.imshow(sobel_combined_norm, cmap='gray'), plt.title('combined (Sobel)')
    plt.show()
    
    # 3. Edge Detection using Laplacian Filter
    #The Laplacian filter is another edge detector. It's a second-order derivative, meaning it looks for areas where the rate of change of intensity itself changes rapidly. It's good at finding fine details but can be more sensitive to noise than Sobel.
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    plt.figure(figsize=(10,5))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(122), plt.imshow(laplacian_norm, cmap='gray'), plt.title('Edges')
    plt.show()
    
    # 4. Image Blurring
    # Simple blur / Averaging
    blur = cv2.blur(img, (7,7))
    # Box filter
    box = cv2.boxFilter(img, -1, (7,7), normalize=True)
    # Gaussian blur
    gaussian = cv2.GaussianBlur(img, (7,7), 0)
    
    plt.figure(figsize=(15,5))
    plt.subplot(131), plt.imshow(blur, cmap='gray'), plt.title('Simple blur')
    plt.subplot(132), plt.imshow(box, cmap='gray'), plt.title('Box filter')
    plt.subplot(133), plt.imshow(gaussian, cmap='gray'), plt.title('Gaussian blur')
    plt.show()
    
    # 5. Custom Filter
    # Kernel for detecting diagonal edges
    #Creates a NumPy array representing the custom kernel. This specific 3x3 kernel is designed to highlight diagonal edges or features that have a similar pattern.
    kernel = np.array([[-1, -1, 2],
                       [-1, 2, -1],
                       [2, -1, -1]])
    
    custom_filter = cv2.filter2D(img, -1, kernel)
    custom_filter_norm = cv2.normalize(custom_filter, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    plt.figure(figsize=(10,5))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(122), plt.imshow(custom_filter_norm, cmap='gray'), plt.title('Custom Filter')
    plt.show()
    
    # 6. Overlaying Edges on the Original Image
    # Create a color image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Add edges in red color
    #This is a bit more advanced NumPy slicing and OpenCV operation.
    img_color[:,:,2] = cv2.add(img_color[:,:,2], sobel_combined_norm)
    
    plt.figure(figsize=(10,5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)), plt.title('Edges on image')
    plt.subplot(122), plt.imshow(img, cmap='gray'), plt.title('The original for comparison')
    plt.show()
    
    # Saving Results
    cv2.imwrite('sobel_x.jpg', sobel_x_norm)
    cv2.imwrite('sobel_y.jpg', sobel_y_norm)
    cv2.imwrite('sobel_combined.jpg', sobel_combined_norm)
    cv2.imwrite('laplacian.jpg', laplacian_norm)
    cv2.imwrite('blur.jpg', blur)
    cv2.imwrite('box_filter.jpg', box)
    cv2.imwrite('gaussian_blur.jpg', gaussian)
    cv2.imwrite('custom_filter.jpg', custom_filter_norm)
    cv2.imwrite('edges_on_image.jpg', cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    
    print("Обработка изображения завершена. Результаты сохранены в файлы.")

if __name__ == "__main__":
    main()