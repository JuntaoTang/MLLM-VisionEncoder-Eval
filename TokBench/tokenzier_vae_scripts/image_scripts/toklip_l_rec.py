import argparse

from toklip_rec_common import get_args_parser, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokLIP-L image reconstruction", parents=[get_args_parser("toklip_l")])
    main(parser.parse_args())
