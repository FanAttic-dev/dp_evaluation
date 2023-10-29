from pathlib import Path
import cv2
from utils.config import Config
from utils.video_player import VideoPlayer

videos_dir = Config.var_path
im_size = Config.image_size
export_int_sec = Config.export_int_sec


def process_clip(clip_path: Path):
    frames_dir = clip_path.with_name(f"{clip_path.stem}_frames")
    Path.mkdir(frames_dir, exist_ok=False)

    player = VideoPlayer(clip_path)

    frame_id = 0
    is_alive = player.cap.grab()
    while is_alive:
        frame_sec = frame_id / int(player.fps)

        if frame_sec % export_int_sec == 0:
            frame_img_id = int(frame_sec // export_int_sec)

            frame_path = frames_dir / \
                f"{clip_path.stem}_frame_{frame_id:04d}.jpg"
            print(str(frame_path))

            _, frame = player.cap.retrieve()
            frame = cv2.resize(frame, im_size)
            cv2.imwrite(str(frame_path), frame)

        frame_id += 1
        is_alive = player.cap.grab()

    player.release()


if __name__ == "__main__":
    clips = videos_dir.glob("*.mp4")
    for clip in clips:
        process_clip(clip)
