[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chipqa-no-reference-video-quality-prediction/video-quality-assessment-on-live-etri)](https://paperswithcode.com/sota/video-quality-assessment-on-live-etri?p=chipqa-no-reference-video-quality-prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chipqa-no-reference-video-quality-prediction/video-quality-assessment-on-live-livestream)](https://paperswithcode.com/sota/video-quality-assessment-on-live-livestream?p=chipqa-no-reference-video-quality-prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chipqa-no-reference-video-quality-prediction/video-quality-assessment-on-youtube-ugc)](https://paperswithcode.com/sota/video-quality-assessment-on-youtube-ugc?p=chipqa-no-reference-video-quality-prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chipqa-no-reference-video-quality-prediction/video-quality-assessment-on-konvid-1k)](https://paperswithcode.com/sota/video-quality-assessment-on-konvid-1k?p=chipqa-no-reference-video-quality-prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chipqa-no-reference-video-quality-prediction/video-quality-assessment-on-live-vqc)](https://paperswithcode.com/sota/video-quality-assessment-on-live-vqc?p=chipqa-no-reference-video-quality-prediction)

# CHIPQA: No-Reference Video Quality Assessment using Space-Time Chips

This repository contains code for ChipQA and ChipQA-0.

Please cite ChipQA if you use this work:
J. P. Ebenezer, Z. Shang, Y. Wu, H. Wei, S. Sethuraman and A. C. Bovik, "ChipQA: No-Reference Video Quality Prediction via Space-Time Chips," in IEEE Transactions on Image Processing, vol. 30, pp. 8059-8074, 2021, doi: 10.1109/TIP.2021.3112055.

The IEEE link is https://ieeexplore.ieee.org/document/9540785 and you can find a free preprint here: https://arxiv.org/abs/2109.08726.

## Requirements

See requirements.txt

## Feature extraction

To extract features on SDR 8 bit MP4 or AVI files, run
```

python3 chipqa.py --input_file path/to/input_file --results_folder path/to/output_file

```
`input_file` is the .mp4 video from which chipqa features will be extracted.
`output_file` will contain the features that are written out.

To extract features on SDR YUV or HDR YUV files, run
```

python3 chipqa_yuv.py --input_file path/to/input_file --results_folder path/to/output_file --width W --height H --bit_depth {8/10/12} --color_space {BT2020/BT709}

```

Note that metadata such as height, width, bit depth, and color space have to be specified for YUV files.

## Training with a generic database

Run 

```

python3 cleaner_svr.py --score_file /path/to/score.csv --feature_folder feature_folder --train_and_test

```

See `python clearner_svr.py --help` for more options and descriptions of arguments.

## Training with LIVE-APV Livestream VQA database

After extracting the features, run 
```

python3 zip_feats_and_scores.py input_folder csv_file output_file 

```
Here `input_folder` must contain the features generated from the previous step, `csv_file` must be a csv file with LIVE-APV database names and scores, and `output_file` will be where the combined features and scores will be stored.

After this, run 
```

python3 svr.py input_file output_folder

```
to evaluate. `input_file` must be the path where the zipped features and scores are saved. The predictions and the ground truth MOS for each of the runs will be stored in `output_folder`.

`all_srocc.m` can then be used to find the SROCC and LCC.

## Testing on a new database with a pretrained SVR

1. Run chipqa.py with the path to the folder of videos and the output directory.
2. Run `testing.py` with a path to the input feature file(s).
