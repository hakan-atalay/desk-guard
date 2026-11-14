from __future__ import annotations

import argparse
from dataclasses import replace

from deskguard.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deskguard", description="Smart desk security camera")
    parser.add_argument("--config", default=None, help="YAML config, defaults to the bundled one")
    parser.add_argument("--weights", default=None, help="Override model weights")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Live monitoring from a camera or video")
    run.add_argument("--source", default="0", help="Camera index or video path")
    run.add_argument("--record", default=None, help="Save the annotated video")

    detect = commands.add_parser("detect", help="Detect items in an image or a folder")
    detect.add_argument("source")
    detect.add_argument("--output", default="runs/detect")

    collect = commands.add_parser("collect", help="Capture training images from a camera")
    collect.add_argument("--output", default="data/raw")
    collect.add_argument("--camera", type=int, default=0)
    collect.add_argument("--interval", type=float, default=1.0)

    label = commands.add_parser("autolabel", help="Pseudo-label images with YOLOE")
    label.add_argument("--images", default="data/raw")
    label.add_argument("--output", default="data/desk")
    label.add_argument("--val-ratio", type=float, default=0.2)
    label.add_argument("--test-ratio", type=float, default=0.15)
    label.add_argument("--confidence", type=float, default=0.35)
    label.add_argument("--only", default=None, help="Every item in these images is this label")

    web = commands.add_parser("webdata", help="Download and merge public Roboflow datasets")
    web.add_argument("--api-key", default=None, help="Roboflow API key, or ROBOFLOW_API_KEY")
    web.add_argument("--skip-download", action="store_true")
    web.add_argument("--output", default="data/train")

    evaluate = commands.add_parser("evaluate", help="Compare models on a labelled split")
    evaluate.add_argument("models", nargs="+")
    evaluate.add_argument("--data", default="data/mytest/data.yaml")
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--output", default="runs/eval")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.weights:
        config = replace(config, model=replace(config.model, weights=args.weights))

    if args.command == "run":
        from deskguard.app import run as run_live

        run_live(config, args.source, args.record)
    elif args.command == "detect":
        from deskguard.app import detect_images

        print(detect_images(config, args.source, args.output))
    elif args.command == "collect":
        from deskguard.app import collect as collect_images

        print(collect_images(args.output, args.camera, args.interval))
    elif args.command == "autolabel":
        from deskguard.dataset import autolabel

        print(
            autolabel(
                config,
                args.images,
                args.output,
                args.val_ratio,
                args.test_ratio,
                args.confidence,
                args.only,
            )
        )
    elif args.command == "webdata":
        import os

        from deskguard.dataset import download_web_data, merge_web_data

        if not args.skip_download:
            key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
            if not key:
                parser.error("give --api-key or set ROBOFLOW_API_KEY")
            download_web_data(key)
        from deskguard.detector import Detector

        print(merge_web_data(config, output_dir=args.output, detector=Detector(config)))
    elif args.command == "evaluate":
        from deskguard.dataset import evaluate as evaluate_models

        print(evaluate_models(config, args.data, args.split, args.models, args.output))
    return 0
