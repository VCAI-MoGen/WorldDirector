import argparse
import torch
from internal.viewer import Viewer
import os
import json

def cli():
    # define arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--background_color", "--background_color", "--bkg_color", "-b",
                        type=str, nargs="+", default=["black"],
                        help="e.g.: white, black, 0 0 0, 1 1 1")
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--perturb", action="store_true")
    args = parser.parse_args()

    # arguments post process
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = (1., 1., 1.)
        else:
            args.background_color = (0., 0., 0.)
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    data_path = os.path.join(args.root_path, 'data.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
        args.model_paths = [data['scene']]
        print("TEXT: ", data['text'])

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(**viewer_init_args)

    # start viewer server
    viewer.start()

cli()