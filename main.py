import argparse
import os

import desed
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
import yaml
from local.classes_dict import (classes_labels_maestro_real)
from resample_folder import resample_folder
from utils import (calculate_macs, generate_tsv_wav_durations,
                         process_tsvs)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import (StronglyAnnotatedSet, UnlabeledSet,
                                        WeakSet)
from encoder import CatManyHotEncoder, ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup


def resample_data_generate_durations(config_data, test_only=False, evaluation=False):
    if not test_only:
        dsets = ["real_maestro_train_folder",
            "real_maestro_val_folder",
            "test_folder",
        ]
    elif evaluation:
        dsets = ["eval_folder"]
    else:
        dsets = ["test_folder"]

    for dset in dsets:
        print(f"Resampling {dset} to 16 kHz.")
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
        )

    if not evaluation:
        for base_set in ["synth_val", "test"]:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(
                    config_data[base_set + "_folder"], config_data[base_set + "_dur"]
                )


def get_encoder(config):
    maestro_real_encoder = ManyHotEncoder(
        list(classes_labels_maestro_real.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    encoder = maestro_real_encoder

    return encoder
def split_maestro(config, maestro_dev_df):

    np.random.seed(config["training"]["seed"])
    split_f = config["training"]["maestro_split"]
    for indx, scene_name in enumerate(
        [
            "cafe_restaurant",
            "city_center",
            "grocery_store",
            "metro_station",
            "residential_area",
        ]
    ):

        mask = (
            maestro_dev_df["filename"].apply(lambda x: "_".join(x.split("_")[:-1]))
            == scene_name
        )
        filenames = (
            maestro_dev_df[mask]["filename"].apply(lambda x: x.split("-")[0]).unique()
        )
        np.random.shuffle(filenames)

        pivot = int(split_f * len(filenames))
        filenames_train = filenames[:pivot]
        filenames_valid = filenames[pivot:]
        if indx == 0:
            mask_train = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_train)
            )
            mask_valid = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_valid)
            )
            train_split = maestro_dev_df[mask_train]
            valid_split = maestro_dev_df[mask_valid]
        else:
            mask_train = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_train)
            )
            mask_valid = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_valid)
            )
            train_split = pd.concat(
                [train_split, maestro_dev_df[mask_train]], ignore_index=True
            )
            valid_split = pd.concat(
                [valid_split, maestro_dev_df[mask_valid]], ignore_index=True
            )

    return train_split, valid_split

  maestro_real_devtest = StronglyAnnotatedSet(
            config["data"]["real_maestro_val_folder"],
            maestro_real_devtest_tsv,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "maestro_real_dev"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_maestro_real,
            test=True,
        )
        devtest_dataset = torch.utils.data.ConcatDataset(
            [desed_devtest_dataset, maestro_real_devtest]
        )
