# Layout trình bày — Linear Regression Visualizer

> Slide trình bày **kiến thức Linear Regression** cho giáo viên và các bạn học.  
> Không giải thích code · Không đưa code vào slide.  
> **3 người đầu** trình bày lý thuyết · **Người thứ 4** giới thiệu ứng dụng và **demo trực tiếp**.

---

## Tổng quan buổi present


| Người | Chủ đề                                            | Thời gian gợi ý        |
| ----- | ------------------------------------------------- | ---------------------- |
| **1** | Linear Regression là gì? — Khái niệm & mô hình    | 4–5 phút               |
| **2** | Cách model học — Loss, Gradient Descent, đánh giá | 4–5 phút               |
| **3** | Dữ liệu & tiền xử lý — Chuẩn bị cho hồi quy       | 4–5 phút               |
| **4** | Ứng dụng minh họa + **Demo** (Linear Regression)  | 2 phút + 6–8 phút demo |


**Tổng:** ~20–25 phút (không tính Q&A)

**Luồng:** Linear Regression là gì → Model học thế nào → Dữ liệu cần gì → Xem trực quan trên app

> **Lưu ý:** Ứng dụng hỗ trợ thêm **Polynomial Regression** (bậc > 1) như tính năng mở rộng — chỉ nhắc ngắn ở cuối, **không là trọng tâm**.

---

## Ký hiệu ghi chú ảnh

Trong tài liệu này, chỗ cần chèn ảnh được đánh dấu:

> **[ẢNH: mô tả ảnh cần chèn]**

Bạn tự chụp từ app, vẽ tay, hoặc lấy minh họa trên mạng phù hợp.

---

# NGƯỜI 1 — Linear Regression: Khái niệm & Mô hình (4–5 phút)

---

### Slide 1.1 — Giới thiệu

**Tiêu đề:** Linear Regression — Dự đoán bằng đường thẳng

**Nội dung slide:**

- Hồi quy tuyến tính là một trong những thuật toán Machine Learning **cơ bản nhất**.
- Mục tiêu: tìm **mối quan hệ tuyến tính** giữa biến đầu vào $x$ và biến mục tiêu $y$.
- Nhóm xây dựng **ứng dụng trực quan** giúp nhìn thấy model **học từng bước** — không chỉ xem kết quả cuối.

**[ẢNH: Logo nhóm / ảnh bìa — scatter plot có đường thẳng fit qua các điểm]**

**Nội dung nói:**

- Hôm nay trình bày về **Linear Regression** — phần cốt lõi của dự án.
- App còn hỗ trợ đa thức bậc cao hơn, nhưng hôm nay tập trung vào **đường thẳng** — dễ hiểu nhất.

---

### Slide 1.2 — Bài toán thực tế

**Tiêu đề:** Ví dụ: Diện tích nhà → Giá nhà

**Bảng minh họa:**


| Diện tích (m²) | Giá nhà (triệu) |
| -------------- | --------------- |
| 50             | 350             |
| 100            | 600             |
| 150            | 850             |


**Câu hỏi đặt ra:**

- Nếu nhà **120 m²**, giá khoảng bao nhiêu?
- Làm sao máy tính **tự tìm quy luật** từ bảng trên?

**[ẢNH: Biểu đồ scatter — trục X: diện tích, trục Y: giá — 3 điểm dữ liệu, chưa có đường fit]**

**Nội dung nói:**

- Đây là bài toán **dự đoán** (regression): cho $x$ → ước lượng $y$.
- Con người có thể "ước chừng" bằng mắt; AI cần **công thức toán học**.

---

### Slide 1.3 — Mô hình Linear Regression

**Tiêu đề:** Công thức hồi quy tuyến tính

**Công thức chính:**

$$y = w_1 \cdot x + b$$


| Ký hiệu | Ý nghĩa                                                    |
| ------- | ---------------------------------------------------------- |
| $x$     | Biến đầu vào (feature) — ví dụ: diện tích                  |
| $y$     | Biến mục tiêu (target) — ví dụ: giá nhà                    |
| $w_1$   | Hệ số góc — diện tích tăng 1 đơn vị thì giá tăng bao nhiêu |
| $b$     | Hệ số chặn — giá "nền" khi $x = 0$                         |


**Nội dung nói:**

- $w_1$ và $b$ là hai **trọng số** (weights) model cần học.
- Ban đầu model **chưa biết** $w_1$, $b$ — phải học từ dữ liệu.

**[ẢNH: Đồ thị có đường thẳng $y = w_1 x + b$ — chú thích $w_1$ (độ dốc) và $b$ (giao trục Y)]**

---

### Slide 1.4 — Dự đoán & Sai số

**Tiêu đề:** Model đoán đúng hay sai?

**Công thức:**

$$\text{Dự đoán} = w_1 \cdot x + b$$

$$\text{Sai số (residual)} = \text{Dự đoán} - y_{\text{thật}}$$

**Ví dụ** (ban đầu $w_1 = 0$, $b = 0$ → dự đoán toàn 0):


| Diện tích | Giá thật | Dự đoán | Sai số |
| --------- | -------- | ------- | ------ |
| 50        | 350      | 0       | -350   |
| 100       | 600      | 0       | -600   |
| 150       | 850      | 0       | -850   |


**Nội dung nói:**

- Sai số **âm** → dự đoán **thấp hơn** thực tế.
- Sai số **dương** → dự đoán **cao hơn** thực tế.
- Mục tiêu học: làm sai số **càng nhỏ càng tốt**.

**[ẢNH: Scatter plot — các điểm dữ liệu và đường dự đoán $y=0$ (nằm trên trục X) — vẽ đường residual dọc từ điểm xuống đường fit]**

---

### Slide 1.5 — Nhiều điểm dữ liệu → Một đường thẳng tốt nhất

**Tiêu đề:** "Best fit line" — đường khớp nhất

**Nội dung slide:**

- Với nhiều điểm, **không thể** đi qua tất cả — tìm đường **gần nhất** với toàn bộ data.
- Linear Regression tìm $w_1$, $b$ sao cho tổng sai số **nhỏ nhất**.
- Kết quả mong đợi (ví dụ giá nhà): $y \approx 5x + 100$

**[ẢNH: Scatter nhiều điểm + đường thẳng fit tốt — so sánh 2 đường: fit kém (xa điểm) vs fit tốt (gần điểm)]**

**Kết — chuyển sang Người 2:**

> "Đã biết model là một đường thẳng. Câu hỏi tiếp theo: **làm sao máy tìm ra $w_1$ và $b$?** — [Tên TV2] sẽ trình bày."

---

# NGƯỜI 2 — Cách Model Học: Loss & Gradient Descent (4–5 phút)

---

### Slide 2.1 — Loss: Đo mức độ "tệ" của model

**Tiêu đề:** Hàm Loss — Model tốt hay tệ?

**Công thức MSE (Mean Squared Error):**

$$\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (\text{dự đoán}*i - y_i)^2 = \frac{1}{n} \sum*{i=1}^{n} \text{residual}_i^2$$

**Nội dung nói:**

- Bình phương sai số → sai lớn bị **phạt nặng hơn**.
- Loss **càng nhỏ** → model **càng tốt**.
- Training = liên tục **giảm Loss**.

**Ví dụ số** (3 điểm giá nhà, dự đoán ban đầu = 0):

$$\text{Loss} = \frac{(-350)^2 + (-600)^2 + (-850)^2}{3} = 401{,}666.67$$

**[ẢNH: Biểu đồ Loss — trục Y là giá trị Loss, có thể vẽ 2 cột so sánh Loss cao vs Loss thấp]**

---

### Slide 2.2 — Gradient Descent: Thuật toán "học"

**Tiêu đề:** Gradient Descent — Đoán, sai, sửa, lặp lại

**Quy trình trên slide:**

```text
① Khởi tạo: w₁ = 0, b = 0  (chưa biết gì)
② Dự đoán giá cho từng mẫu
③ Tính sai số (residual)
④ Tính Loss
⑤ Tính Gradient — "nên sửa w₁, b theo hướng nào?"
⑥ Cập nhật: trọng số mới = trọng số cũ − η × gradient
⑦ Lặp lại nhiều lần (epochs)
```


| Thuật ngữ             | Ý nghĩa                                                             |
| --------------------- | ------------------------------------------------------------------- |
| **Gradient**          | Hướng làm Loss tăng nhanh nhất → đi **ngược** gradient để Loss giảm |
| **Learning rate (η)** | Bước nhảy mỗi lần sửa — quá lớn: nhảy loạn; quá nhỏ: học chậm       |
| **Epoch**             | Một vòng lặp đủ ①→⑥                                                 |


**[ẢNH: Sơ đồ "xuống dốc" — quả bóng lăn xuống đáy thung lũng (điểm Loss thấp nhất) — minh họa Gradient Descent]**

---

### Slide 2.3 — Công thức Gradient

**Tiêu đề:** Gradient tính như thế nào?

**Công thức:**

$$\text{gradient} = \frac{2}{n} \times \sum_{i=1}^{n} \text{residual}_i \times \text{(giá trị feature)}$$

**Riêng cho bias $b$** (feature luôn = 1):

$$\frac{\partial \text{Loss}}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} \text{residual}_i$$

**Ví dụ số** (3 mẫu giá nhà, dự đoán = 0):


| Gradient | Tính                                                  | Kết quả      | Ý nghĩa            |
| -------- | ----------------------------------------------------- | ------------ | ------------------ |
| $db$     | $\frac{2}{3} \times (-350 - 600 - 850)$               | $-1200$      | Cần **tăng** $b$   |
| $dw_1$   | $\frac{2}{3} \times \sum x_i \cdot \text{residual}_i$ | $-136{,}667$ | Cần **tăng** $w_1$ |


**Nội dung nói:**

- Gradient **âm** → tăng trọng số sẽ làm Loss **giảm**.
- Sau 1 lần cập nhật (η = 0.00001): $b$ từ 0 → 0.012, $w_1$ từ 0 → 1.37 — đường thẳng **dịch lên**, gần data hơn.

**[ẢNH: 2 đồ thị cạnh nhau — Epoch 0 (đường y=0) vs Epoch 1 (đường y=1.37x+0.012) — loss giảm từ 401666 → 236779]**

---

### Slide 2.4 — Quá trình học qua các Epoch

**Tiêu đề:** Model cải thiện dần qua từng epoch


| Epoch | Công thức gần đúng   | Loss        |
| ----- | -------------------- | ----------- |
| 0     | $y = 0$              | 401 666     |
| 1     | $y = 1.37x + 0.01$   | 236 779     |
| 5     | $y = 4x + 50$        | 20 000      |
| 50    | $y \approx 5x + 100$ | $\approx 0$ |


**Nội dung nói:**

- Mỗi epoch model **sửa nhẹ** trọng số — không nhảy một phát đúng ngay.
- Loss **giảm dần** → bằng chứng model đang học đúng hướng.
- Kết quả cuối: $y = 5x + 100$ → nhà 120 m² → $y = 5(120)+100 = 700$ triệu.

**[ẢNH: Đường cong Loss giảm dần theo epoch (loss curve) — trục X: epoch, trục Y: loss]**

---

### Slide 2.5 — Đánh giá model & Chuẩn hóa dữ liệu

**Tiêu đề:** Model tốt đến mức nào?

**Các chỉ số đánh giá:**


| Chỉ số   | Ý nghĩa đơn giản                                                 |
| -------- | ---------------------------------------------------------------- |
| **RMSE** | Sai số trung bình (cùng đơn vị với $y$) — càng nhỏ càng tốt      |
| **MAE**  | Trung bình khoảng cách tuyệt đối — dễ hiểu hơn RMSE              |
| **R²**   | Mức độ giải thích dữ liệu — gần **1** = rất tốt, gần **0** = kém |


**Tại sao cần chuẩn hóa dữ liệu?**

- Khi $x$ hoặc $y$ có giá trị **rất lớn** (vd: giá nhà tỷ VND) → gradient không ổn định, học chậm hoặc sai.
- Chuẩn hóa (đưa về thang tương đương) giúp model **học ổn định** hơn.

**Nhắc ngắn — Polynomial (phụ):**

- Linear Regression: đường **thẳng** (bậc 1).
- App còn hỗ trợ **Polynomial** bậc 2, 3… cho dữ liệu cong — nhưng **cùng thuật toán** Gradient Descent.

**[ẢNH: Bảng metrics mẫu — RMSE, MAE, R² sau khi train (chụp từ app hoặc tự tạo bảng)]**

**Kết — chuyển sang Người 3:**

> "Đã hiểu model học thế nào. [Tên TV3] trình bày **dữ liệu** cần chuẩn bị ra sao trước khi train."

---

# NGƯỜI 3 — Dữ liệu & Tiền xử lý (4–5 phút)

---

### Slide 3.1 — Dữ liệu trong hồi quy tuyến tính

**Tiêu đề:** Feature, Target & mẫu dữ liệu


| Khái niệm         | Ví dụ giá nhà     | Giải thích                       |
| ----------------- | ----------------- | -------------------------------- |
| **Feature** ($x$) | Diện tích (m²)    | Biến đầu vào — dùng để dự đoán   |
| **Target** ($y$)  | Giá nhà (VND)     | Biến cần dự đoán                 |
| **Mẫu (sample)**  | 1 căn nhà         | Một dòng dữ liệu                 |
| **Tập train**     | Nhiều căn nhà     | Dùng để model **học**            |
| **Tập test**      | Căn nhà chưa thấy | Dùng để **kiểm tra** sau khi học |


**Nội dung nói:**

- Model **không học thuộc** từng căn nhà — nó học **quy luật chung** giữa diện tích và giá.
- Tập test giúp kiểm tra model dự đoán **dữ liệu mới** có tốt không.

**[ẢNH: Bảng CSV giá nhà — cột Dien_Tich_m2 và Gia_Nha_VND (chụp từ app hoặc file mẫu)]**

---

### Slide 3.2 — Ba cách nhập dữ liệu trong ứng dụng

**Tiêu đề:** Nguồn dữ liệu linh hoạt


| Cách            | Mô tả                | Phù hợp khi                        |
| --------------- | -------------------- | ---------------------------------- |
| **Nhập tay**    | Gõ từng cặp $x, y$   | Thử nhanh vài điểm, học thuật toán |
| **Import CSV**  | File Excel/CSV thật  | Dự án thực tế, nhiều mẫu           |
| **Dataset mẫu** | Dữ liệu tổng hợp sẵn | Demo nhanh trên lớp                |


**Nội dung nói:**

- CSV thực tế thường có **nhiều cột** — cần chọn cột nào là feature, cột nào là target.
- Ví dụ dự án: `house_price_train_linear.csv` — dự đoán giá nhà theo diện tích.

**[ẢNH: Screenshot tab Manual / CSV / Sample trong app — 3 tab sidebar]**

---

### Slide 3.3 — Vấn đề chất lượng dữ liệu

**Tiêu đề:** Dữ liệu "bẩn" làm model học sai


| Vấn đề              | Ví dụ               | Hậu quả                    |
| ------------------- | ------------------- | -------------------------- |
| **Thiếu giá trị**   | Ô trống, N/A        | Không tính được, gây lỗi   |
| **Outlier**         | Nhà 50 m² giá 50 tỷ | Kéo lệch đường fit         |
| **Giá trị quá lớn** | Giá tỷ VND          | Gradient học không ổn định |
| **Sai đơn vị**      | m² vs km² lẫn lộn   | Quan hệ $x$-$y$ sai        |


**Nội dung nói:**

- "Garbage in, garbage out" — dữ liệu kém → model kém, dù thuật toán đúng.
- Cần **làm sạch** trước khi train.

**[ẢNH: Scatter plot có 1 điểm outlier xa biệt — đường fit bị kéo lệch so với đường fit khi bỏ outlier]**

---

### Slide 3.4 — Tiền xử lý dữ liệu

**Tiêu đề:** Làm sạch trước khi train


| Bước                   | Mô tả                                                  |
| ---------------------- | ------------------------------------------------------ |
| **Loại dòng thiếu**    | Bỏ các mẫu có giá trị trống hoặc không hợp lệ          |
| **Loại outlier (IQR)** | Bỏ điểm quá xa so với phần lớn dữ liệu                 |
| **Chuẩn hóa target**   | Chia giá nhà cho $10^9$ — đưa về thang nhỏ hơn, dễ học |


**Luồng trên slide:**

```text
CSV thô → Lọc missing → Lọc outlier → Chuẩn hóa → Train Linear Regression
```

**Nội dung nói:**

- Tiền xử lý **không thay đổi** bản chất quan hệ — chỉ giúp model học **ổn định và chính xác** hơn.
- Sau train, kết quả dự đoán được **đổi lại** đơn vị gốc (VND) để dễ hiểu.

**[ẢNH: Screenshot báo cáo preprocessing trong app — "X dòng gốc, Y dòng bỏ missing, Z dòng bỏ outlier, còn N dòng"]**

---

### Slide 3.5 — Train / Test — Đánh giá công bằng

**Tiêu đề:** Không đánh giá trên chính dữ liệu đã học

**Nội dung slide:**

- **Train set** — model nhìn thấy khi học.
- **Test set** — model **chưa từng thấy** — dùng để đánh giá thật.
- Nếu chỉ test trên train → có thể **overfit** (học thuộc, không hiểu quy luật).

**Ví dụ dự án:**

- Train: `house_price_train_linear.csv`
- Test: `house_price_test_linear.csv`
- Sau train → đánh giá RMSE, R² trên test → chứng minh model **generalize**.

**[ẢNH: Screenshot cửa sổ kết quả test trong app — RMSE, MAE, R²]**

**Kết — chuyển sang Người 4:**

> "Lý thuyết đã đủ. [Tên TV4] sẽ cho các bạn **thấy trực tiếp** Linear Regression học trên ứng dụng."

---

# NGƯỜI 4 — Ứng dụng minh họa + DEMO (2 phút + 6–8 phút demo)

---

### Slide 4.1 — Ứng dụng làm gì?

**Tiêu đề:** Linear Regression Visualizer — Nhìn thấy AI đang học

**Nội dung slide:**

- Desktop app giúp **quan sát trực quan** quá trình Gradient Descent.
- Không chỉ xem kết quả cuối — xem **từng epoch**: đường thẳng thay đổi, loss giảm dần.
- Phù hợp **học tập** — giáo viên và sinh viên hiểu bản chất, không chỉ dùng "hộp đen".

**[ẢNH: Screenshot toàn màn hình app sau khi train xong — có đường fit + loss curve]**

---

### Slide 4.2 — Giao diện ứng dụng

**Tiêu đề:** Các thành phần chính trên màn hình


| Khu vực             | Chức năng                                                      |
| ------------------- | -------------------------------------------------------------- |
| **Sidebar trái**    | Nhập dữ liệu · chỉnh Degree, Learning Rate, Epochs · nút Train |
| **Biểu đồ Fit**     | Điểm dữ liệu + đường dự đoán + đường sai số (residual)         |
| **Biểu đồ Loss**    | Loss giảm qua từng epoch                                       |
| **Thanh animation** | Play / Pause · kéo slider xem từng epoch                       |
| **Phương trình**    | Hiển thị $y = w_1 x + b$ đang thay đổi                         |
| **Bảng dữ liệu**    | $x$, $y$, dự đoán, sai số theo epoch hiện tại                  |


**[ẢNH: Screenshot app có chú thích mũi tên chỉ từng khu vực — sidebar, chart fit, chart loss, animation bar, equation]**

---

### Slide 4.3 — Kịch bản Demo (phần chính)

> **Tập trung Linear Regression** — đặt **Degree = 1** trong toàn bộ demo.

#### Demo A — Giá nhà thật (4–5 phút) ⭐ Chính

1. Mở app → tab **CSV** → chọn file giá nhà
2. Chọn cột diện tích (feature) và giá (target)
3. Bật tiền xử lý → xem báo cáo làm sạch
4. Đặt **Degree = 1**, chỉnh Epochs → **Train**
5. **Play animation** — nói trong lúc chạy:
  - *"Ban đầu đường thẳng chưa khớp — loss cao."*
  - *"Qua từng epoch, $w_1$ và $b$ được sửa — đường dịch lên, gần điểm hơn."*
  - *"Loss curve bên phải giảm dần — đúng như lý thuyết Người 2 trình bày."*
6. Kéo slider về **epoch 0** và **epoch cuối** — so sánh phương trình
7. Load file **test** → xem RMSE, R²

**[ẢNH: Chụp sẵn 3 mốc — epoch 0 / giữa / cuối — để chiếu nếu demo lỗi]**

---

#### Demo B — Nhập tay nhanh (2 phút)

1. Tab **Manual** — nhập 3–5 điểm (vd: diện tích & giá đơn giản)
2. Degree = 1 → Train → kéo slider
3. Chỉ bảng dữ liệu: cột **sai số giảm dần** qua các epoch

**[ẢNH: Screenshot tab Manual với vài điểm và đường fit]**

---

#### Demo C — Polynomial (phụ, ~1 phút nếu còn thời gian)

1. Tab **Sample** → chọn dataset cong → Degree = **3** → Train
2. Nói ngắn: *"Đây là mở rộng — khi dữ liệu không thẳng, cần đường cong bậc cao hơn. Linear Regression (bậc 1) vẫn là nền tảng."*

**[ẢNH: So sánh đường thẳng (degree 1) vs đường cong (degree 3) trên cùng data cong]**

---

### Slide 4.4 — Kết luận & Q&A (cả nhóm)

**Tiêu đề:** Tóm tắt

**Bullet:**

- **Linear Regression** tìm đường thẳng $y = w_1 x + b$ khớp nhất với dữ liệu.
- **Gradient Descent** học bằng cách liên tục giảm Loss qua từng epoch.
- **Dữ liệu sạch** và **train/test tách biệt** là điều kiện để model đáng tin.
- Ứng dụng giúp **nhìn thấy** toàn bộ quá trình — từ lý thuyết đến thực hành.

**[ẢNH: Ảnh nhóm / slide cảm ơn]**

---

## Phụ lục — Checklist trước khi present

### Chuẩn bị

- [ ] App chạy được, đã test demo giá nhà (Degree = 1)
- [ ] Chụp sẵn ảnh dự phòng (epoch 0, epoch cuối, loss curve, metrics test)
- [ ] File CSV train/test có trong máy demo
- [ ] Máy chiếu: zoom app ~100%, dark mode

### Phân công

- [ ] **Người 1:** Linear Regression là gì — công thức, ví dụ giá nhà, sai số
- [ ] **Người 2:** Loss, Gradient Descent, ví dụ số, metrics, chuẩn hóa
- [ ] **Người 3:** Feature/target, chất lượng data, tiền xử lý, train/test
- [ ] **Người 4:** Giới thiệu app ngắn → **demo Linear Regression** là phần chính

### Câu hỏi dự kiến


| Câu hỏi                                   | Người trả lời              |
| ----------------------------------------- | -------------------------- |
| Linear Regression khác gì Classification? | Người 1                    |
| Loss là gì, tại sao bình phương?          | Người 2                    |
| Gradient Descent hoạt động thế nào?       | Người 2                    |
| Tại sao cần tập test?                     | Người 3                    |
| Outlier ảnh hưởng thế nào?                | Người 3                    |
| R² = 0.95 nghĩa là gì?                    | Người 2                    |
| App có làm gì khác ngoài Linear?          | Người 4 (Polynomial — phụ) |


---

## Gợi ý đặt tên file slide


| File                            | Người   | Slide                   |
| ------------------------------- | ------- | ----------------------- |
| `01-05_Linear_Concept.pptx`     | Người 1 | 1.1 – 1.5               |
| `06-10_Learning_Algorithm.pptx` | Người 2 | 2.1 – 2.5               |
| `11-15_Data_Preprocessing.pptx` | Người 3 | 3.1 – 3.5               |
| `16-18_App_Demo.pptx`           | Người 4 | 4.1 – 4.2 (+ demo live) |


Hoặc **1 file chung** `Linear_Regression_Slides.pptx`.

---

## Danh sách ảnh cần chuẩn bị (tổng hợp)


| #   | Mô tả ảnh                                             | Slide | Ai chụp |
| --- | ----------------------------------------------------- | ----- | ------- |
| 1   | Scatter plot điểm dữ liệu giá nhà (chưa có đường fit) | 1.2   | Người 1 |
| 2   | Đường thẳng $y=w_1x+b$ — chú thích slope & intercept  | 1.3   | Người 1 |
| 3   | Residual lines từ điểm xuống đường fit                | 1.4   | Người 1 |
| 4   | So sánh fit kém vs fit tốt                            | 1.5   | Người 1 |
| 5   | Minh họa "quả bóng lăn xuống dốc" (Gradient Descent)  | 2.2   | Người 2 |
| 6   | Epoch 0 vs Epoch 1 — đường fit thay đổi               | 2.3   | Người 2 |
| 7   | Loss curve giảm theo epoch                            | 2.4   | Người 2 |
| 8   | Bảng metrics RMSE / MAE / R²                          | 2.5   | Người 2 |
| 9   | Bảng CSV giá nhà                                      | 3.1   | Người 3 |
| 10  | 3 tab nhập liệu trong app                             | 3.2   | Người 3 |
| 11  | Outlier kéo lệch đường fit                            | 3.3   | Người 3 |
| 12  | Báo cáo preprocessing                                 | 3.4   | Người 3 |
| 13  | Kết quả test RMSE / R²                                | 3.5   | Người 3 |
| 14  | Screenshot toàn app sau train                         | 4.1   | Người 4 |
| 15  | Screenshot có chú thích các khu vực UI                | 4.2   | Người 4 |
| 16  | Epoch 0 / giữa / cuối (dự phòng demo)                 | 4.3   | Người 4 |
| 17  | So sánh degree 1 vs degree 3 (phụ)                    | 4.3C  | Người 4 |


