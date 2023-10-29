from pathlib import Path
import cv2
from utils.config import Config
from utils.video_player import VideoPlayer

videos_dir = Config.var_path
im_size = Config.image_size
export_int_sec = Config.export_int_sec


def process_period(period: str):
    video_path = videos_dir / period
    frames_dir = videos_dir / \
        video_path.with_stem(f"{video_path.stem}_frames").stem
    Path.mkdir(frames_dir, exist_ok=False)

    player = VideoPlayer(video_path)

    is_alive = True
    frame_id = 0
    while is_alive:
        # is_alive, frame = player.get_next_frame()
        is_alive = player.cap.grab()

        frame_sec = frame_id / int(player.fps)
        if frame_sec % export_int_sec == 0:
            frame_img_id = int(frame_sec // export_int_sec)

            frame_path = frames_dir / \
                f"{video_path.stem}_frame_{frame_id:04d}.jpg"
            print(str(frame_path))

            is_alive, frame = player.cap.retrieve()
            frame = cv2.resize(frame, im_size)
            cv2.imwrite(str(frame_path), frame)

        frame_id += 1

    player.release()


if __name__ == "__main__":
    for period in ["var_p0.mp4", "var_p1.mp4"]:
        process_period(period)
