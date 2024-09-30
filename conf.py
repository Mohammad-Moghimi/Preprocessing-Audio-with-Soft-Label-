scaler:
  statistic: instance # instance or dataset-wide statistic
  normtype: minmax # minmax or standard or mean normalization
  dims: [1, 2] # dimensions over which normalization is applied
  savepath: ./scaler.ckpt # path to scaler checkpoint
data: # change with your paths if different.
  synth_maestro_train: "../../data/dcase/dataset/audio/maestro_synth_train_16k"
  synth_maestro_train_44k: "../../data/dcase/dataset/audio/maestro_synth_train"
  synth_maestro_tsv: "../../data/dcase/dataset/metadata/maestro_synth_train.tsv"
  real_maestro_train_folder: "../../data/dcase/dataset/audio/maestro_real_train_16k"
  real_maestro_train_folder_44k: "../../data/dcase/dataset/audio/maestro_real_train"
  real_maestro_train_tsv: "../../data/dcase/dataset/metadata/maestro_real_train.tsv"
  real_maestro_val_folder: "../../data/dcase/dataset/audio/maestro_real_validation_16k"
  real_maestro_val_folder_44k: "../../data/dcase/dataset/audio/maestro_real_validation"
  real_maestro_val_tsv: "../../data/dcase/dataset/metadata/maestro_real_validation.tsv"
  real_maestro_val_dur: "../../data/dcase/dataset/metadata/maestro_real_durations.tsv"
  test_folder: "../../data/dcase/dataset/audio/validation/validation_16k/"
  test_folder_44k: "../../data/dcase/dataset/audio/validation/validation/"
  test_tsv: "../../data/dcase/dataset/metadata/validation/validation.tsv"
  test_dur: "../../data/dcase/dataset/metadata/validation/validation_durations.tsv"
  eval_folder: "../../data/dcase/dataset/audio/eval24_16k"
  eval_folder_44k: "../../data/dcase/dataset/audio/eval24"
  audio_max_len: 10
  fs: 16000
  net_subsample: 4
feats:
  n_mels: 128
  n_filters: 2048
  hop_length: 256
  n_window: 2048
  sample_rate: 16000
  f_min: 0
  f_max: 8000
