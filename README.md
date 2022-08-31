# CHIPQA: No-Reference Video Quality Assessment using Space-Time Chips

This repository contains code for ChipQA and ChipQA-0.

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
