"""
This module contains Kaggle submission related code.

"""
import os
from functools import reduce

import numpy as np
import pandas as pd

# from tgssalt_challenge import features
# from tgssalt_challenge.io import IO


def rle_decode(rle_mask):
    """Decode a run length encoded string.

    Args:
        rle_mask: run-length as string formated (start length)

    Returns:
        mask (numpy array): 1 - mask, 0 - background
    """

    """
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    if type(rle_mask) == float:
        return np.zeros([101, 101])
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)


def rle_encode(im, order='F', format=True):
    """Decode a numpy array into a run length encoded string.

    Args:
        im: numpy array where 1 is a mask value and 0 is background.
        order: is down-then-right, i.e. Fortran
        format: determines if the order needs to be preformatted (according to submission rules) or not

    Returns:
        lre_encoded (string): run length as string formated
    """
    pixels = im.flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if format:
        return ' '.join(str(x) for x in runs)
    else:
        return runs


def ensemble_rle_submissions(submission_files, threshold):
    """Ensemble multiple rle submission files.

    Args:
        submission_files: list of files to ensemble
        threshold: number of matches needed to across submissions to accept a mask pixel

    Returns:
        result (dataframe): an ensembled result file
        df (dataframe): merged dataframe.
    """
    dataframes = []
    for file in submission_files:
        df = pd.read_csv(file).rename(
            columns={'rle_mask': 'rle_mask_' + str(len(dataframes))})
        dataframes.append(df)

    df = reduce(lambda left,right: pd.merge(left, right, on=['id']), dataframes)\

    result = dataframes[0].copy().rename(columns={'rle_mask_0': 'rle_mask'})

    for i in range(len(result)):
        nan_count = 0
        for mi in range(len(dataframes)):
            if str(df.loc[i, 'rle_mask_' + str(mi)]) == str(np.nan):
                nan_count += 1

        if nan_count == 0:
            decoded_mask_all = rle_decode(df.loc[i, 'rle_mask_0'])
            for mi in range(1, len(dataframes)):
                decoded_mask_all += rle_decode(df.loc[i, 'rle_mask_' + str(mi)])

            decoded_mask_all[decoded_mask_all < threshold] = 0
            decoded_mask_all[decoded_mask_all >= threshold] = 1

            mask = rle_encode(decoded_mask_all)

            result.loc[i, 'rle_mask'] = mask

        i = i + 1

    return result, df
