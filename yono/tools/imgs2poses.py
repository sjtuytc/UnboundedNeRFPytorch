# Modified from https://github.com/Fyusion/LLFF
import os
import sys
import glob

from colmap_utils.pose_utils import gen_poses


def check_structure(scenedir):
    source = os.path.join(scenedir, 'source')
    if not os.path.isdir(source):
        print('Invalid directory structure.')
        print('Please put all your images under', source, '!')
        sys.exit()
    if len(glob.glob(f'{source}/*[JPG\|jpg\|png\|jpeg\|PNG]')) == 0:
        print('Invalid directory structure.')
        print('No image in', source, '!')
        sys.exit()
    print('Directory structure check: PASS.')


if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_type', type=str,
                        default='exhaustive_matcher', help='type of matcher used.  Valid options: \
                        exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
    parser.add_argument('scenedir', type=str,
                        help='input scene directory')
    args = parser.parse_args()

    if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
        print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
        sys.exit()

    check_structure(args.scenedir)

    gen_poses(args.scenedir, args.match_type, factors=[2,4,8])

