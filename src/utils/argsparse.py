
import argparse


class EvalArgsNamespace(argparse.Namespace):
    record: bool
    mouse: bool
    video_name: str
    config_path: str
    hide_windows: bool
    export_frames: bool
    no_debug: bool


def parse_args():
    parser = argparse.ArgumentParser()
    ns = EvalArgsNamespace()
    parser.add_argument('--show', action='store_true')
    return parser.parse_args(namespace=ns)
