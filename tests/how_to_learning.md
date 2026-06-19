# Quá trình Model học bằng Gradient Descent (Ví dụ dễ hiểu)

## Tập dữ liệu

| Diện tích (m²) | Giá nhà (triệu) |
| -------------: | --------------: |
|             50 |             350 |
|            100 |             600 |
|            150 |             850 |

Mục tiêu của AI:

Tìm công thức:

$$y = w_1 x + b$$

Trong đó:

* `x`: diện tích
* `y`: giá nhà dự đoán
* `w1`: độ ảnh hưởng của diện tích
* `b`: giá trị nền

---

# Ý tưởng đơn giản

Ban đầu AI **không biết gì**.

AI sẽ:

1. Đoán công thức
2. Dự đoán giá
3. So sánh với giá thật
4. Tính sai số
5. Sửa công thức
6. Lặp lại nhiều lần

Flow:

```text
Đoán
↓
Tính sai
↓
Tìm hướng sửa
↓
Sửa công thức
↓
Lặp lại
↓
Sai số giảm dần
```

---

# Epoch 0

## Bước 1: Khởi tạo

Ban đầu:

```text
w1 = 0
b = 0
```

Công thức:

$$y = 0x + 0$$

AI chưa biết gì cả.

---

## Bước 2: Dự đoán

| Diện tích | Giá thật | Giá dự đoán |
| --------: | -------: | ----------: |
|        50 |      350 |           0 |
|       100 |      600 |           0 |
|       150 |      850 |           0 |

---

## Bước 3: Tính sai số

$$Residual = prediction - y$$

| Diện tích | Sai số |
| --------: | -----: |
|        50 |   -350 |
|       100 |   -600 |
|       150 |   -850 |

---

## Bước 4: Tính Loss

Công thức:

$$
Loss = \frac{(-350)^2 + (-600)^2 + (-850)^2}{3}
$$

Kết quả:

$$
Loss = 401666.67
$$

Loss rất lớn.

Điều này nghĩa là:

```text
AI dự đoán rất tệ
```

---

## Bước 5: Gradient tìm hướng sửa

Gradient giống như một chiếc la bàn:

```text
"Nên sửa công thức theo hướng nào?"
```

Kết quả:

```text
dw = -136666.67
db = -1200
```

Ý nghĩa:

```text
w1 cần tăng
b cần tăng
```

---

## Bước 6: Cập nhật weights

Công thức:

$$
weight = weight - lr \times gradient
$$

Giả sử:

```text
Learning rate = 0.00001
```

Tính:

$$
w_1 = 0 - 0.00001 \times (-136666.67) = 1.3667
$$

$$
b = 0 - 0.00001 \times (-1200) = 0.012
$$

Công thức mới:

$$
y = 1.3667x + 0.012
$$

---

# Epoch 1

Dự đoán:

| Diện tích | Giá thật | Giá dự đoán |
| --------: | -------: | ----------: |
|        50 |      350 |       68.35 |
|       100 |      600 |      136.68 |
|       150 |      850 |      205.01 |

Sai số:

| Diện tích | Residual |
| --------: | -------: |
|        50 |  -281.65 |
|       100 |  -463.32 |
|       150 |  -644.99 |

Loss:

$$
Loss = 236779
$$

Loss đã giảm:

```text
401666
↓
236779
```

AI đang học tốt hơn.

---

Gradient mới:

```text
dw=-103888
db=-926.64
```

Cập nhật:

$$
w_1 = 2.4056
$$

$$
b = 0.0213
$$

Công thức mới:

$$
y = 2.4056x + 0.0213
$$

---

# Epoch tiếp theo

Model tiếp tục:

```text
Epoch 2
→ sửa tiếp

Epoch 3
→ sửa tiếp

Epoch 4
→ sửa tiếp
```

Ví dụ:

| Epoch | Công thức     |   Loss |
| ----: | ------------- | -----: |
|     0 | y=0           | 401666 |
|     1 | y=1.37x+0.012 | 236779 |
|     2 | y=2.41x+0.021 | 139000 |
|     5 | y=4x+50       |  20000 |
|    10 | y=4.8x+95     |   1500 |
|    50 | y=5x+100      |     ≈0 |

---

# Kết quả cuối cùng

Sau khi học xong:

$$
y = 5x + 100
$$

Nếu nhập:

```text
Diện tích = 120m²
```

Model dự đoán:

$$
y = 5(120) + 100 = 700
$$

Kết quả:

```text
Giá nhà dự đoán = 700 triệu
```

---

# Kết luận

Gradient Descent thực chất làm việc theo quy trình:

```text
Khởi tạo công thức ngẫu nhiên
↓
Dự đoán
↓
Tính sai số
↓
Tính hướng sửa (Gradient)
↓
Cập nhật weights
↓
Lặp lại nhiều lần
↓
Tìm công thức tối ưu
```

Model không học thuộc dữ liệu.

Model học ra **quy luật giữa diện tích và giá nhà** để dự đoán dữ liệu mới.
