import glob
import os.path
import zipfile
import shutil  # Import shutil to copy files
from pathlib import Path

import desed
import intervaltree
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def count_chunks(inlen, chunk_size, chunk_stride):
    return int((inlen - chunk_size + chunk_stride) / chunk_stride)


def get_chunks_indx(in_len, chunk_size, chunk_stride, discard_last=False):
    i = -1
    for i in range(count_chunks(in_len, chunk_size, chunk_stride)):
        yield i * chunk_stride, i * chunk_stride + chunk_size
    if not discard_last and i * chunk_stride + chunk_size < in_len:
        if in_len - (i + 1) * chunk_stride > 0:
            yield (i + 1) * chunk_stride, in_len


def read_maestro_annotation(annotation_f):
    annotation = []
    with open(annotation_f, "r") as f:
        lines = f.readlines()

    for l in lines:
        if Path(annotation_f).suffix != ".csv":
            start, stop, event, confidence = l.rstrip("\n").split("\t")
            annotation.append(
                {
                    "onset": float(start),
                    "offset": float(stop),
                    "event_label": event,
                    "confidence": float(confidence),
                }
            )
        else:
            start, stop, event = l.rstrip("\n").split("\t")
            if start == "onset":
                continue
            annotation.append(
                {
                    "onset": float(start),
                    "offset": float(stop),
                    "event_label": event,
                    "confidence": 1.0,
                }
            )
    return annotation


def ann2intervaltree(annotation):
    tree = intervaltree.IntervalTree()
    for elem in annotation:
        tree.add(intervaltree.Interval(elem["onset"], elem["offset"], elem))

    return tree


def get_current_annotation(annotation, start, end):
    # use intervaltree here !
    overlapping = []
    for ann in annotation:
        if ann.overlaps(start, end):
            c_segment = ann.data
            # make it relative
            onset = max(0.0, c_segment["onset"] - start)
            offset = min(end - start, c_segment["offset"] - start)
            c_segment = {
                "onset": onset,
                "offset": offset,
                "event_label": c_segment["event_label"],
                "confidence": c_segment["confidence"],
            }
            overlapping.append(c_segment)

    overlapping = sorted(overlapping, key=lambda x: x["onset"])
    return overlapping


def split_maestro_single_file(output_audio_folder, audiofile, annotation, window_len=10, hop_len=1):
    audio, fs = sf.read(audiofile)
    annotation = read_maestro_annotation(annotation)
    annotation = ann2intervaltree(annotation)
    new_annotation = []
    for st, end in get_chunks_indx(len(audio), int(window_len * fs), int(hop_len * fs)):
        c_seg = audio[st:end]
        c_annotation = get_current_annotation(annotation, st / fs, end / fs)

        # save
        start = st / fs * 100
        end = end / fs * 100
        filename = Path(audiofile).stem + f"-{int(start):06d}-{int(end):06d}"
        sf.write(os.path.join(output_audio_folder, filename + ".wav"), c_seg, fs)
        for line in c_annotation:
            new_annotation.append(
                {
                    "filename": filename + Path(audiofile).suffix,
                    "onset": line["onset"],
                    "offset": line["offset"],
                    "event_label": line["event_label"],
                    "confidence": line["confidence"],
                }
            )  # tsv like desed

    return new_annotation


def split_maestro_real(download_folder, out_audio_folder, out_meta_folder):
    audiofiles = glob.glob(
        os.path.join(download_folder, "development_audio", "**/*.wav"), recursive=True
    )
    annotation_files = glob.glob(
        os.path.join(download_folder, "development_annotation", "**/*.txt"),
        recursive=True,
    )

    assert len(audiofiles) == len(annotation_files)
    assert len(audiofiles) > 0, (
        "You probably have the wrong folder as input to this script."
        f"Check {download_folder}, does it contain MAESTRO dev data ? "
        f"It must have development_annotation and development_audio as sub-folders."
    )
    for split in ["train", "validation"]:
        split_info = os.path.join(os.path.dirname(__file__), f"{split}_split.csv")
        if split == "validation":
            split_info = pd.read_csv(split_info)["val"]
            hop_len = 5
        else:
            split_info = pd.read_csv(split_info)[f"{split}"]
            hop_len = 1
        split_info = set([Path(x).stem for x in split_info])
        # filter audiofiles here now and annotation
        c_audiofiles = [x for x in audiofiles if Path(x).stem in split_info]
        c_annotation_files = [x for x in annotation_files if Path(x).stem in split_info]

        # get corresponding annotation files.
        c_audiofiles = sorted(c_audiofiles, key=lambda x: Path(x).stem)
        c_audiofiles = {Path(x).stem: x for x in c_audiofiles}
        # get all metadata
        c_annotation_files = {Path(x).stem: x for x in c_annotation_files}

        Path(os.path.join(out_audio_folder, f"maestro_real_{split}")).mkdir(
            parents=True, exist_ok=True
        )
        Path(out_meta_folder).mkdir(parents=True, exist_ok=True)

        # split here
        all_annotations = []
        for k in tqdm(c_audiofiles.keys()):
            c_path = c_audiofiles[k]
            c_metadata_f = c_annotation_files[k]
            c_annotations = split_maestro_single_file(
                os.path.join(out_audio_folder, f"maestro_real_{split}"),
                c_path,
                c_metadata_f,
                window_len=10,
                hop_len=hop_len,
            )
            all_annotations.extend(c_annotations)

        all_annotations = pd.DataFrame.from_dict(all_annotations)
        all_annotations = all_annotations.sort_values(by="filename", ascending=True)
        all_annotations.to_csv(
            os.path.join(out_meta_folder, f"maestro_real_{split}.tsv"),
            sep="\t",
            index=False,
        )
    (Path(download_folder) / "development_metadata.csv").rename(
        Path(out_meta_folder) / "maestro_real_durations.tsv"
    )


def split_maestro_synth(download_folder, out_audio_folder, out_meta_folder):
    audiofiles = glob.glob(
        os.path.join(download_folder, "audio", "**/*.wav"), recursive=True
    )
    annotation_files = glob.glob(
        os.path.join(download_folder, "estimated_strong_labels", "**/*.csv"),
        recursive=True,
    )

    assert len(audiofiles) == len(annotation_files)
    assert len(audiofiles) > 0, (
        "You probably have the wrong folder as input to this script."
        f"Check {download_folder}, does it contain MAESTRO dev data ? "
        f"It must have development_annotation and development_audio as sub-folders."
    )

    c_audiofiles = audiofiles
    c_annotation_files = annotation_files
    split = "train"
    # get corresponding annotation files.
    c_audiofiles = sorted(c_audiofiles, key=lambda x: Path(x).stem)
    c_audiofiles = {Path(x).stem: x for x in c_audiofiles}
    # get all metadata
    c_annotation_files = {Path(x).stem: x for x in c_annotation_files}

    Path(os.path.join(out_audio_folder, f"maestro_synth_{split}")).mkdir(
        parents=True, exist_ok=True
    )
    Path(out_meta_folder).mkdir(parents=True, exist_ok=True)

    # split here
    all_annotations = []
    for k in tqdm(c_audiofiles.keys()):
        c_path = c_audiofiles[k]
        c_metadata_f = c_annotation_files["mturk_" + k]
        c_annotations = split_maestro_single_file(
            os.path.join(out_audio_folder, f"maestro_synth_{split}"),
            c_path,
            c_metadata_f,
        )
        all_annotations.extend(c_annotations)

    all_annotations = pd.DataFrame.from_dict(all_annotations)
    all_annotations = all_annotations.sort_values(by="filename", ascending=True)
    all_annotations.to_csv(
        os.path.join(out_meta_folder, f"maestro_synth_{split}.tsv"),
        sep="\t",
        index=False,
    )


def prepare_maestro(dcase_dataset_folder, local_zip_paths):
    """
    Prepare the MAESTRO dataset using local ZIP files instead of downloading.

    Parameters:
    - dcase_dataset_folder: The main folder where the dataset will be organized.
    - local_zip_paths: A dictionary containing paths to your local ZIP files.
      Expected keys:
        - 'synth_audio': Path to 'audio.zip' for the synthetic data.
        - 'synth_meta': Path to 'meta.zip' for the synthetic metadata.
        - 'dev_audio': Path to 'development_audio.zip' for the development data.
        - 'dev_meta': Path to 'development_annotation.zip' for the development annotations.
        - 'dev_audio_durations': Path to 'development_metadata.csv' file.
    """

    synth_label_metadata_path = os.path.join(dcase_dataset_folder, "maestro_synth")

    def help_extract(main_dir, zip_path):
        Path(main_dir).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(main_dir)

    # Extract synthetic metadata
    help_extract(synth_label_metadata_path, local_zip_paths['synth_meta'])
    # Extract synthetic audio
    synth_audio_path = os.path.join(dcase_dataset_folder, "maestro_synth")
    help_extract(synth_audio_path, local_zip_paths['synth_audio'])

    # Extract development audio
    dev_audio_path = os.path.join(dcase_dataset_folder, "maestro_dev")
    help_extract(dev_audio_path, local_zip_paths['dev_audio'])
    # Extract development metadata
    dev_meta_path = os.path.join(dcase_dataset_folder, "maestro_dev")
    help_extract(dev_meta_path, local_zip_paths['dev_meta'])

    # Copy development metadata CSV
    shutil.copy(
        local_zip_paths['dev_audio_durations'],
        os.path.join(dev_meta_path, "development_metadata.csv")
    )


def get_maestro(dcase_dataset_folder, local_zip_paths):
    prepare_maestro(os.path.join(dcase_dataset_folder, "MAESTRO_original"), local_zip_paths)
    print(
        "Preparing MAESTRO real development and training sets."
        " Splitting it into 10s chunks."
    )

    split_maestro_real(
        os.path.join(dcase_dataset_folder, "MAESTRO_original", "maestro_dev"),
        os.path.join(dcase_dataset_folder, "audio"),
        os.path.join(dcase_dataset_folder, "metadata"),
    )

    print("Preparing MAESTRO synth training set. Splitting it into 10s chunks.")

    split_maestro_synth(
        os.path.join(dcase_dataset_folder, "MAESTRO_original", "maestro_synth"),
        os.path.join(dcase_dataset_folder, "audio"),
        os.path.join(dcase_dataset_folder, "metadata"),
    )


if __name__ == "__main__":
    # Define the paths to your local ZIP files
    local_zip_paths = {
        'synth_audio': '/path/to/your/local/audio.zip',
        'synth_meta': '/path/to/your/local/meta.zip',
        'dev_audio': '/path/to/your/local/development_audio.zip',
        'dev_meta': '/path/to/your/local/development_annotation.zip',
        'dev_audio_durations': '/path/to/your/local/development_metadata.csv'
    }

    # Set the main dataset folder
    dcase_dataset_folder = '/media/samco/Data1/MAESTRO_split'

    # Call the get_maestro function with the local ZIP paths
    get_maestro(dcase_dataset_folder, local_zip_paths)
