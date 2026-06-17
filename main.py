"""
Owner: Thành viên 4 - GUI / Integration.

Entry point for the Polynomial Regression Visualizer desktop app.
"""

from app.gui import PolynomialRegressionApp


def main() -> int:
    """Launch the Tkinter / CustomTkinter application."""
    app = PolynomialRegressionApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
