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

Gradient tính như sau:

$$
gradient = \frac{2.0}{n} \times (X^T \cdot residuals)
$$

Trong đó:

* `n`: số mẫu dữ liệu (`n = 3`)
* `X`: ma trận thiết kế (cột 1 là bias, cột 2 là diện tích)
* `residuals`: vector sai số ở Bước 3
* `X^T`: ma trận chuyển vị của `X`
* `@` hoặc `·`: phép nhân ma trận

### Tại sao có công thức này? (Chứng minh từ Loss)

Công thức trên **không phải tự nhiên có sẵn** — nó được **suy ra** từ việc lấy đạo hàm của Loss (MSE) theo từng trọng số.

#### 1. Viết model dạng ma trận

Thay vì tính từng mẫu, gộp tất cả vào ma trận:

$$
predictions = X \cdot weights
$$

Với:

$$
weights =
\begin{bmatrix}
b \\
w_1
\end{bmatrix},
\quad
X =
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n
\end{bmatrix}
$$

Mỗi hàng của $X$ là một mẫu: cột 1 luôn là $1$ (cho bias $b$), cột 2 là diện tích $x_i$ (cho $w_1$).

#### 2. Định nghĩa Loss (MSE)

$$
Loss = \frac{1}{n} \sum_{i=1}^{n} residual_i^2
$$

Trong đó:

$$
residual_i = prediction_i - y_i, \quad
residuals =
\begin{bmatrix}
residual_1 \\
residual_2 \\
\vdots \\
residual_n
\end{bmatrix}
= predictions - y
$$

#### 3. Đạo hàm Loss theo **một** trọng số

Lấy đạo hàm theo quy tắc chuỗi (chain rule). Với mỗi trọng số $w_j$ (ví dụ $j=0$ là $b$, $j=1$ là $w_1$):

$$
\frac{\partial Loss}{\partial w_j}
= \frac{1}{n} \sum_{i=1}^{n} 2 \cdot residual_i \cdot \frac{\partial prediction_i}{\partial w_j}
$$

Vì $prediction_i = \sum_k X_{i,k} \cdot w_k$, nên:

$$
\frac{\partial prediction_i}{\partial w_j} = X_{i,j}
$$

Thay vào:

$$
\frac{\partial Loss}{\partial w_j}
= \frac{2}{n} \sum_{i=1}^{n} X_{i,j} \cdot residual_i
$$

**Ý nghĩa:** gradient của trọng số $w_j$ = trung bình có trọng số của tất cả residuals, nhân với giá trị cột $j$ của mẫu đó.

#### 4. Gộp tất cả trọng số thành vector

Phần tử thứ $j$ của vector gradient chính là $\frac{\partial Loss}{\partial w_j}$ ở trên.

Nhận xét: $\sum_{i=1}^{n} X_{i,j} \cdot residual_i$ chính là **phần tử thứ $j$** của phép nhân $X^T \cdot residuals$.

Vì vậy, gộp cho **tất cả** trọng số cùng lúc:

$$
gradient =
\begin{bmatrix}
\frac{\partial Loss}{\partial b} \\
\frac{\partial Loss}{\partial w_1}
\end{bmatrix}
= \frac{2}{n} \times (X^T \cdot residuals)
$$

Đây chính là công thức trong code `model.py`:

```python
gradient = (2.0 / n_samples) * (X.T @ residuals)
```

#### 5. Kiểm tra với $b$ và $w_1$

| Trọng số | Cột trong $X$ | Công thức riêng | Kết quả ví dụ |
| -------- | ------------- | --------------- | ------------- |
| $b$ | cột 1 = toàn $1$ | $db = \frac{2}{n}\sum residual_i$ | $db = -1200$ |
| $w_1$ | cột 2 = $x_i$ | $dw = \frac{2}{n}\sum x_i \cdot residual_i$ | $dw = -136666.67$ |

Hai công thức riêng lẻ ở trên chính là hai phần tử của vector $X^T \cdot residuals$ nhân với $\frac{2}{n}$.

#### 6. Tóm tắt luồng suy ra

```text
Loss = trung bình (residual²)
  ↓ lấy đạo hàm theo từng weight
gradient_j = (2/n) × Σ (cột j của X) × (residual)
  ↓ gộp thành ma trận
gradient = (2/n) × X^T · residuals
```

$$
\boxed{
gradient = \frac{2}{n} \times (X^T \cdot residuals)
\quad \leftarrow \text{đạo hàm của MSE Loss theo } weights
}
$$

### Dữ liệu đầu vào

$$
X =
\begin{bmatrix}
1 & 50 \\
1 & 100 \\
1 & 150
\end{bmatrix},
\quad
residuals =
\begin{bmatrix}
-350 \\
-600 \\
-850
\end{bmatrix},
\quad
n = 3
$$

### Bước 1: Tính $X^T \cdot residuals$

$$
X^T =
\begin{bmatrix}
1 & 1 & 1 \\
50 & 100 & 150
\end{bmatrix}
$$

Nhân từng hàng với `residuals`:

$$
\begin{aligned}
\text{hàng 1 (bias)} &= (-350) + (-600) + (-850) = -1800 \\
\text{hàng 2 } (w_1) &= 50 \times (-350) + 100 \times (-600) + 150 \times (-850) \\
&= -17500 - 60000 - 127500 = -205000
\end{aligned}
$$

$$
X^T \cdot residuals =
\begin{bmatrix}
-1800 \\
-205000
\end{bmatrix}
$$

### Bước 2: Nhân với $\frac{2}{n}$

$$
gradient = \frac{2}{3}
\begin{bmatrix}
-1800 \\
-205000
\end{bmatrix}
=
\begin{bmatrix}
-1200 \\
-136666.67
\end{bmatrix}
$$

### Kết quả

$$
db = -1200, \quad dw = -136666.67
$$

(`db` là gradient của bias `b`, `dw` là gradient của trọng số `w_1`)

### Tại sao $db = -1200$? (Chứng minh chi tiết)

#### 1. Bắt đầu từ Loss

Loss (MSE) là trung bình bình phương sai số:

$$
Loss = \frac{1}{n} \sum_{i=1}^{n} residual_i^2
$$

Với mỗi mẫu $i$:

$$
\hat{y}_i = b + w_1 \cdot x_i, \quad residual_i = \hat{y}_i - y_i
$$

#### 2. Đạo hàm Loss theo $b$

Muốn biết $b$ nên tăng hay giảm, ta lấy đạo hàm:

$$
\frac{\partial Loss}{\partial b}
= \frac{1}{n} \sum_{i=1}^{n} 2 \cdot residual_i \cdot \frac{\partial \hat{y}_i}{\partial b}
$$

Vì $\hat{y}_i = b + w_1 x_i$ nên:

$$
\frac{\partial \hat{y}_i}{\partial b} = 1
$$

Thay vào:

$$
\frac{\partial Loss}{\partial b}
= \frac{2}{n} \sum_{i=1}^{n} residual_i
$$

Đây chính là công thức $db$ — phần tử đầu tiên của vector gradient.

#### 3. Liên hệ với ma trận $X$

Cột đầu tiên của $X$ luôn là $1$ vì mỗi dự đoán đều cộng thêm $b \times 1$:

$$
X =
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
1 & x_3
\end{bmatrix}
$$

Nên hàng đầu tiên của $X^T \cdot residuals$ chính là tổng tất cả residuals:

$$
(X^T \cdot residuals)_0
= 1 \cdot residual_1 + 1 \cdot residual_2 + 1 \cdot residual_3
= \sum_{i=1}^{n} residual_i
$$

Do đó:

$$
db = \frac{2}{n} \times \sum_{i=1}^{n} residual_i
$$

#### 4. Thay số cụ thể (Epoch 0)

Ở Epoch 0, model dự đoán toàn $0$ ($b=0$, $w_1=0$), giá thật là $350, 600, 850$:

| Mẫu | $x_i$ | $y_i$ | $\hat{y}_i$ | $residual_i = \hat{y}_i - y_i$ |
| --: | ----: | ----: | ----------: | -------------------------------: |
|   1 |    50 |   350 |           0 |                             -350 |
|   2 |   100 |   600 |           0 |                             -600 |
|   3 |   150 |   850 |           0 |                             -850 |

Tổng residuals:

$$
\sum residual_i = (-350) + (-600) + (-850) = -1800
$$

Thay vào công thức $db$:

$$
db = \frac{2}{3} \times (-1800) = -1200
$$

#### 5. Giải thích trực quan

```text
Dự đoán = 0  <  Giá thật (350, 600, 850)
→ Sai số âm ở TẤT CẢ các mẫu
→ Tổng residuals = -1800 (âm)
→ db = -1200 (âm)
```

Gradient âm nghĩa là: nếu **tăng** $b$ thì Loss sẽ **giảm**.

Kiểm tra bằng công thức cập nhật ở Bước 6:

$$
b_{mới} = b - lr \times db = 0 - 0.00001 \times (-1200) = 0.012
$$

$b$ tăng từ $0$ lên $0.012$ → đường dự đoán dịch lên → gần giá thật hơn. Đúng với trực giác.

#### 6. Tóm tắt công thức $db$

$$
\boxed{
db = \frac{2}{n} \sum_{i=1}^{n} (prediction_i - y_i)
= \frac{2}{n} \times (\text{tổng tất cả sai số})
}
$$

Với ví dụ này: $db = \frac{2}{3} \times (-1800) = -1200$.

### Ý nghĩa

Gradient âm nghĩa là loss sẽ giảm nếu **tăng** trọng số đó:

```text
w1 cần tăng  (vì dw < 0)
b cần tăng   (vì db < 0)
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
