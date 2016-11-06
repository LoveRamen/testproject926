import chainer
import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--org', help='Path to original image')
parser.add_argument('--var', help='xyz')

# recognize args
args = parser.parse_args()

if args.org:
    print('yes')
else:
    print('no')

