"""
Owner: Thành viên 3 - Visualization / Plot / Animation.

File này chứa helper liên quan đến animation state.

Lưu ý:
    - Tkinter `after` có thể vẫn nằm trong GUI vì nó gắn với widget.
    - File này nên chứa các hàm tính frame tiếp theo, validate index, speed.

Input:
    - frame_index hiện tại.
    - total_frames.
    - playing state.

Output:
    - frame_index mới.
    - playing state mới.

TODO:
    - Nếu nhóm muốn tách animation khỏi GUI, implement helper tại đây.
"""


def next_frame_index(frame_index: int, total_frames: int) -> tuple[int, bool]:
    """
    Tính frame tiếp theo.

    Input:
        frame_index: Frame hiện tại.
        total_frames: Tổng số frame.

    Output:
        next_index: Frame tiếp theo.
        should_continue: True nếu animation còn chạy, False nếu đã tới cuối.
    """
    raise NotImplementedError("Thành viên 3 implement nếu cần tách logic animation khỏi GUI.")

