import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Implementation of Semantic Image Synthesis with Spatially-Adaptive Normalization")

    # Dataloader
    parser.add_argument('--path', required=True, help='path to the image folder')
    parser.add_argument('--img-size', dest='img_size', nargs='+', default=[256,256], type=int,
                        help='The size of images for training and validation')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
                        help='Batch size for the dataloaders for train and val set')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=4,
                        help='Number of CPU cores you want to use for data loading')

    # SPADE normalization layer
    parser.add_argument('--spade-filter', dest='spade_filter', default=128, type=int, 
                        help='The filter size to use in SPADE block')
    parser.add_argument('--sapde-kernel', dest='spade_kernel', default=3, type=int,
                        help='The kernel size to use in SPADE block')

    # SPADE ResBlk
    # You can add flags here depending on you want to do addition, concatenation of the ouputs
    parser.add_argument('--spade-resblk-kernel', dest='spade_resblk_kernel', default=3, type=int,
                        help='The kernel size to be used for the conv layers in SPADE ResBlk')

    # SPADE Generator
    parser.add_argument('--gen-input-size', dest='gen_input_size', default=256, type=int,
                        help='The noise size to be given to generator')
    parser.add_argument('--gen-hidden-size', dest='gen_hidden_size', default=16384, type=int,
                        help='Hidden size for the first layer of generator')

    # Training arguments
    parser.add_argument('--epochs', dest='epochs', type=int, default=100,
                        help='Number of epochs to run of training')
    parser.add_argument('--lr_gen', dest='lr_gen', type=int, default=0.0001,
                        help='Learning rate of generator')
    parser.add_argument('--lr_dis', dest='lr_dis', type=int, default=0.0004,
                        help='Learning rate of discriminator')

    return parser