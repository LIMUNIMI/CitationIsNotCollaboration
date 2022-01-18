"""Utils for CLI scripts"""
import argparse
import logging
import importlib


class FeatgraphArgParse(argparse.ArgumentParser):
  """Argument parser for Featgraph CLI scripts

  Args:
    kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument("--jvm-path",
                      metavar="PATH",
                      help="The Java virtual machine full path")
    self.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        metavar="LEVEL",
        default="WARN",
        type=lambda s: str(s).upper(),
        help="The logging level. Default is 'WARN'",
    )
    self.add_argument(
        "--tqdm",
        action="store_true",
        help="Use tqdm progress bar (you should install tqdm for this)",
    )

  def custom_parse(self, argv):
    """Parse arguments, configure logger and eventually import :mod:`tqdm`"""
    args = self.parse_args(argv)
    try:
      log_level = int(args.log_level)
    except ValueError:
      log_level = args.log_level
    args.logging_kwargs = dict(
        level=log_level,
        format="%(asctime)s %(name)-12s %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.basicConfig(**args.logging_kwargs)
    args.tqdm = importlib.import_module("tqdm").tqdm if args.tqdm else None
    return args
