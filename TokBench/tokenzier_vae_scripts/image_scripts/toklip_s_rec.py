import argparse

from toklip_rec_common import get_args_parser, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokLIP-S image reconstruction", parents=[get_args_parser("toklip_s")])
    main(parser.parse_args())
