import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu giả định: Diện tích (m2) và giá nhà (ngàn đô la)
area = np.array([50, 60, 80, 100, 120, 140, 160, 180, 200, 220])
price = np.array([150, 180, 240, 300, 360, 420, 480, 540, 600, 660])


# Bước 1: Tính toán các thông số của phương trình hồi quy tuyến tính
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


# Bước 2: Hàm dự đoán giá nhà dựa trên diện tích
def predict_price(area, w, b):
    return w * area + b


# Dự đoán giá trị cho các điểm dữ liệu
predicted_prices = predict_price(area, w, b)

# Bước 3: Sử dụng subplot để hiển thị hai biểu đồ

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Biểu đồ 1: Biểu đồ phân tán dữ liệu thực tế
ax1.scatter(area, price, color='blue')
ax1.set_title('Biểu đồ phân tán: Diện tích vs Giá nhà')
ax1.set_xlabel('Diện tích (m2)')
ax1.set_ylabel('Giá nhà (ngàn đô la)')

# Biểu đồ 2: Biểu đồ với đường hồi quy
ax2.scatter(area, price, color='blue', label='Dữ liệu thực tế')
ax2.plot(area, predicted_prices, color='red', label='Đường hồi quy')
ax2.set_title('Hồi quy tuyến tính: Diện tích vs Giá nhà')
ax2.set_xlabel('Diện tích (m2)')
ax2.set_ylabel('Giá nhà (ngàn đô la)')
ax2.legend()

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Dự đoán giá cho diện tích 150 m2
predicted_price = predict_price(150, w, b)
print(f"Dự đoán giá cho ngôi nhà có diện tích 150 m2: {predicted_price:.2f} ngàn đô la")
