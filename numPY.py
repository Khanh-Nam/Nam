import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu giả định: Diện tích (m2) và giá nhà (ngàn đô la)
area = np.array([50, 60, 80, 100, 120, 140, 160, 180, 200, 220])
price = np.array([150, 180, 240, 300, 360, 420, 480, 540, 600, 660])

# Bước 1: Vẽ biểu đồ phân tán để quan sát dữ liệu
plt.scatter(area, price, color='blue')
plt.title('Diện tích nhà (m2) vs Giá nhà (ngàn đô la)')
plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá nhà (ngàn đô la)')
plt.show()


# Bước 2: Tính toán các thông số của phương trình hồi quy tuyến tính
def linear_regression(x, y):
    n = len(x)

    # Tính các giá trị trung bình
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Tính hệ số w (slope) và b (intercept)
    numer = np.sum((x - mean_x) * (y - mean_y))  # Tử số của công thức w
    denom = np.sum((x - mean_x) ** 2)  # Mẫu số của công thức w
    w = numer / denom
    b = mean_y - (w * mean_x)

    return w, b


# Áp dụng hàm linear_regression
w, b = linear_regression(area, price)

print(f"Hệ số w (slope): {w}")
print(f"Hệ số b (intercept): {b}")


# Bước 3: Hàm dự đoán giá nhà dựa trên diện tích
def predict_price(area, w, b):
    return w * area + b


# Dự đoán giá cho diện tích 150 m2
predicted_price = predict_price(150, w, b)
print(f"Dự đoán giá cho ngôi nhà có diện tích 150 m2: {predicted_price:.2f} ngàn đô la")

# Bước 4: Vẽ biểu đồ với đường hồi quy
plt.scatter(area, price, color='blue', label='Dữ liệu thực tế')

# Dự đoán giá trị cho các điểm dữ liệu
predicted_prices = predict_price(area, w, b)

# Vẽ đường hồi quy
plt.plot(area, predicted_prices, color='red', label='Đường hồi quy')
plt.title('Diện tích nhà (m2) vs Giá nhà (ngàn đô la)')
plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá nhà (ngàn đô la)')
plt.legend()
plt.show()
