import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Implementation of Semantic Image Synthesis with Spatially-Adaptive Normalization")

    # SPADE normalization layer
    parser.add_argument('--spade-filter', dest='spade_filter', default=128, type=int, 
                        help='The filter size to use in SPADE block')
    parser.add_argument('--sapde-kernel', dest='spade_kernel', default=3, type=int,
                        help='The kernel size to use in SPADE block')

    # SPADE ResBlk
    # You can add flags here depending on you want to do addition, concatenation of the ouputs
    parser.add_argument('spade-resblk-kernel', dest='spade_resblk_kernel', default=3, type=int,
                        help='The kernel size to be used for the conv layers in SPADE ResBlk')

    # SPADE Generator
    parser.add_argument('--gen-input-size', dest='gen_input_size', default=256, type=int,
                        help='The noise size to be given to generator')
    parser.add_argument('--gen-hidden-size', dest='gen_hidden_size', default=16384, type=int,
                        help='Hidden size for the first layer of generator')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.hidden_size%16 != 0:
        print('hidden-size should be multiple of 16. It is based on paper where first input', end=" ") 
        print('to SPADE is (4,4) in height and width. You can change this defualt in args.py')
        exit()

    print(args)