
import argparse


class EvalArgsNamespace(argparse.Namespace):
    show: bool
    fig_save: bool


def parse_args():
    parser = argparse.ArgumentParser()
    ns = EvalArgsNamespace()
    parser.add_argument('--show', action='store_true')
    return parser.parse_args(namespace=ns)
