# Polynomial AI Visualizer Project

Đây là skeleton dự án để tách code từ prototype `polynomial_regression_app.py`.

Mục tiêu cuối cùng:

- Chạy app bằng lệnh: `python main.py`.
- Người dùng nhập dữ liệu thủ công hoặc import CSV.
- App train Polynomial Regression bằng Gradient Descent.
- App visualize quá trình AI học theo từng epoch.
- App hiển thị đường dự đoán và biểu đồ loss.

Lưu ý quan trọng:

- Folder này chưa có logic thật.
- Mỗi file chỉ chứa hướng dẫn, interface, input/output và TODO.
- Các thành viên sẽ lấy code base hiện tại rồi đưa vào đúng module.
- Không dồn toàn bộ code vào một file.

## Phân Công Thành Viên

### Thành viên 1: Core AI / Model

Phụ trách:

- `core/model.py`
- `core/preprocessing.py`
- `core/training.py`
- `tests/test_model.py`

Nhiệm vụ:

- Viết class `PolynomialRegressionGD`.
- Viết hàm train Gradient Descent.
- Lưu frame training gồm epoch, weights, loss.
- Viết hàm predict.
- Viết hàm tạo equation text.

### Thành viên 2: Data Input / CSV / Sample

Phụ trách:

- `data/csv_loader.py`
- `data/sample_data.py`
- `tests/test_data.py`

Nhiệm vụ:

- Parse text dạng `x,y`.
- Đọc CSV.
- Lọc các cột numeric.
- Tạo sample dataset.
- Chuyển feature/target thành numpy array.

### Thành viên 3: Visualization / Plot / Animation

Phụ trách:

- `visualization/plotter.py`
- `visualization/animation.py`

Nhiệm vụ:

- Vẽ scatter data.
- Vẽ prediction curve.
- Vẽ loss curve.
- Cập nhật từng frame animation.
- Hỗ trợ slider frame.

### Thành viên 4: GUI / Integration

Phụ trách:

- `app/gui.py`
- `app/controller.py`
- `main.py`
- `requirements.txt`

Nhiệm vụ:

- Dựng giao diện Tkinter.
- Gọi module data để lấy `x`, `y`.
- Gọi module model để train.
- Gọi module visualization để vẽ.
- Quản lý state của app.

## Quy Định Code

- Tên hàm dùng `snake_case`.
- Tên class dùng `PascalCase`.
- Không dùng biến global cho dữ liệu train.
- Model không được import Tkinter hoặc Matplotlib.
- Data loader không được import Tkinter.
- Visualization không được đọc CSV.
- GUI là nơi kết nối các module, không viết thuật toán trong GUI.
- Mỗi hàm phải có type hint.
- Mỗi hàm public nên có docstring ngắn nói rõ input/output.

## Quy Trình Git

Mỗi người làm trên branch riêng:

- `feature/core-model`
- `feature/data-loader`
- `feature/visualization`
- `feature/gui-integration`

Lệnh mẫu:

```bash
git checkout main
git pull
git checkout -b feature/core-model
git add .
git commit -m "Add polynomial regression training core"
git push origin feature/core-model
```

Trước khi merge:

- Chạy được `python main.py` nếu phần sửa liên quan GUI.
- Test module của mình không lỗi.
- Không sửa file của thành viên khác nếu không cần.
- Commit message ngắn, rõ ý.

## Thứ Tự Làm

1. Thành viên 1 làm core model.
2. Thành viên 2 làm data loader.
3. Thành viên 3 làm visualization theo interface đã thống nhất.
4. Thành viên 4 làm GUI skeleton và tích hợp.
5. Cả nhóm test demo cuối.

