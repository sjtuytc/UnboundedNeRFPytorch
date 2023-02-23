import pdb
from tqdm import tqdm
from pathlib import Path

from FourierGrid.hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from FourierGrid.hloc.visualization import plot_images, read_image
from FourierGrid.hloc.utils import viz_3d
from pathlib import Path
from pprint import pformat

from FourierGrid.hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from FourierGrid.hloc import colmap_from_nvm, triangulation, localize_sfm, visualization
from FourierGrid.hloc import extract_features, match_features, localize_inloc, visualization

def run_sfm(args, cfg, data_dict):
    dataset = Path('/content/drive/MyDrive/sep25_image')  # change this if your dataset is somewhere else
    images = dataset / 'mapping/'
    outputs = Path('outputs/zlzhao/')  # where everything will be saved
    sfm_pairs = outputs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
    loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
    reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
    results = outputs / 'Aachen_hloc_superpoint+superglue_netvlad20.txt'  # the result file
    sfm_dir = outputs / 'sfm_superpoint+superglue' # the SfM model we will build
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'
    # list the standard configurations available
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')
    # pick one of the configurations for image retrieval, local feature extraction, and matching
    # you can also simply write your own here!
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    # TODO: finish this.
    pdb.set_trace()
    pass