"""
Owner: Thành viên 4 - GUI / Integration.

File này là entry point của dự án.

Input:
    - Không nhận input trực tiếp.
    - Người dùng sẽ thao tác qua giao diện Tkinter.

Output:
    - Mở app desktop.
    - Trả về exit code 0 khi app đóng bình thường.

TODO:
    - Import `PolynomialRegressionApp` từ `app.gui`.
    - Tạo app.
    - Gọi `mainloop()`.
"""

from app.gui import PolynomialRegressionApp


def main() -> int:
    """Launch the Tkinter / CustomTkinter application."""
    app = PolynomialRegressionApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
