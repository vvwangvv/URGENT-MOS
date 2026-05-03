#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for NISQA."""

import argparse
import csv
import logging
import os
import sys

from tqdm import tqdm


# The following function(s) is(are) the same as in sheet.utils.utils
# copied here for installation-free data preparation
def read_csv(path, dict_reader=False, lazy=False):
    with open(path, newline="") as csvfile:
        if dict_reader:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
        else:
            reader = csv.reader(csvfile)
            fieldnames = None

        if lazy:
            contents = reader
        else:
            contents = [line for line in reader]

    return contents, fieldnames


def main():
    """Run data preprocessing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-path",
        required=True,
        type=str,
        help=("original csv file path."),
    )
    parser.add_argument(
        "--wavdir",
        required=True,
        type=str,
        help=(
            "directory of the waveform files. This is needed because wav paths in the metadata files, the wav dir is not contained."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help=("output csv file path."),
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help=("whether to perform resampling or not."),
    )
    parser.add_argument(
        "--target-sampling-rate",
        type=int,
        help=("target sampling rate."),
    )
    parser.add_argument(
        "--resample-backend",
        type=str,
        default="librosa",
        choices=["librosa"],
        help=("resample backend."),
    )
    parser.add_argument(
        "--target-wavdir",
        type=str,
        help=("directory of the resampled waveform files."),
    )
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # make resampled dir and dynamic import
    if args.resample:
        os.makedirs(args.target_wavdir, exist_ok=True)
        if args.resample_backend == "librosa":
            import librosa

    # read csv
    logging.info("Reading original csv file.")
    filelist, _ = read_csv(args.original_path, dict_reader=True)

    # prepare. header:
    # db,con,file,con_description,filename_deg,filename_ref,source,lang,votes,mos,noi,col,dis,loud,noi_std,col_std,dis_std,loud_std,mos_std,filepath_deg,filepath_ref,filter,timeclipping,wbgn,p50mnru,bgn,clipping,arb_filter,asl_in,asl_out,codec1,codec2,codec3,plcMode1,plcMode2,plcMode3,wbgn_snr,bgn_snr,tc_fer,tc_nburst,cl_th,bp_low,bp_high,p50_q,bMode1,bMode2,bMode3,FER1,FER2,FER3,asl_in_level,asl_out_level
    logging.info("Preparing metadata.")
    metadata = []
    for line in tqdm(filelist, desc="Processing CSV metadata"):
        if len(line) == 0:
            continue
        system_id = int(float(line["con"]))
        wav_path = line["filename_deg"]
        complete_wav_path = os.path.join(args.wavdir, wav_path)
        sample_id = os.path.splitext(wav_path)[0]

        score = float(line["mos"])
        score_std = float(line["mos_std"])

        if "TRAIN" not in args.original_path and "VAL" not in args.original_path:
            sample_id = sample_id.split("_", 1)[1]
        else:
            if sample_id.startswith("book"):
                sample_id = os.path.splitext(line["filename_ref"])[0]
                system_id = os.path.splitext(wav_path)[0].removeprefix(sample_id)[1:]

        # if resample and resample is necessary
        if args.resample and librosa.get_samplerate(complete_wav_path) != args.target_sampling_rate:
            # check whether soundfile has been imported
            if "soundfile" not in sys.modules:
                import soundfile as sf

            resampled_wav_path = os.path.join(args.target_wavdir, wav_path)
            # resample and write if not exist yet
            if not os.path.isfile(resampled_wav_path):
                if args.resample_backend == "librosa":
                    resampled_wav, _ = librosa.load(complete_wav_path, sr=args.target_sampling_rate)
                sf.write(
                    resampled_wav_path,
                    resampled_wav,
                    samplerate=args.target_sampling_rate,
                )
            complete_wav_path = resampled_wav_path

        item = {
            "wav_path": complete_wav_path,
            "score": score,
            "dimension": "mos_overall",
            "score_std": score_std,
            "system_id": system_id,
            "reference_id": sample_id,
        }
        metadata.append(item)

    # write csv
    logging.info("Writing output csv file.")
    fieldnames = ["wav_path", "system_id", "reference_id", "score", "dimension", "score_std"]
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in metadata:
            writer.writerow(line)


if __name__ == "__main__":
    main()
