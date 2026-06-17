"""Animation index helpers for training frame playback."""


def next_frame_index(frame_index: int, total_frames: int) -> tuple[int, bool]:
    """
    Compute the next animation frame index.

    Returns:
        next_index: Next frame to display.
        should_continue: False when the last frame has been reached.
    """
    if total_frames <= 0:
        return 0, False
    if frame_index >= total_frames - 1:
        return total_frames - 1, False
    return frame_index + 1, True
