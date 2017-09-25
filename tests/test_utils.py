import os
import time
import glob
import pandas as pd
import numpy as np
from utils import rle_encode
from PIL import Image

carvana_dir = os.path.join(os.path.expanduser('~'), 'school/data science/kaggle/carvana')
img_dir = os.path.join(os.path.expanduser('~'), 'school/data science/kaggle/carvana/train')
mask_dir = os.path.join(os.path.expanduser('~'), 'school/data science/kaggle/carvana/train_masks')
csv_data = pd.read_csv(os.path.join(carvana_dir, 'train_masks.csv'))
total_fns = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(img_dir, '*.jpg'))]
fn_dict = {fn: [os.path.join(img_dir, fn + '.jpg'), os.path.join(mask_dir, fn + '_mask.gif')] for fn in total_fns}


def test_rle_encode(fn, fn_dict=fn_dict):
    mask_path = fn_dict[fn][1]
    now = time.time()
    out = rle_encode(np.array(Image.open(mask_path))[:,:,np.newaxis])
    print("rle_encode takes %2f seconds to complete" % (time.time()-now))
    truth = csv_data[csv_data.img == os.path.basename(fn_dict[fn][0])]['rle_mask'].values[0]
    if out == truth:
        print("test passed...")
    else:
        print("test failed...")
    return out, truth

if __name__ == "__main__":
    out, truth = test_rle_encode(list(fn_dict.keys())[0])