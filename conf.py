training:
  #batch size: [maestro, synth, strong, weak, unlabel]
  batch_size: [12, 6, 6, 12, 24]
  batch_size_val: 24
  const_max: 2 # max weight used for self supervised loss
  n_epochs_warmup: 50 # num epochs used for exponential warmup
  epoch_decay: 100
  num_workers: 6 # change according to your cpu
  n_epochs: 300 # max num epochs
  early_stop_patience: 200 # Same as number of epochs by default, so no early stopping used
  accumulate_batches: 1
  gradient_clip: 5.0 # 0 no gradient clipping
  val_thresholds: [0.5] # thresholds used to compute f1 intersection in validation.
  n_test_thresholds: 50 # number of thresholds used to compute psds in test
  ema_factor: 0.999 # ema factor for mean teacher
  self_sup_loss: mse # bce or mse for self supervised mean teacher loss
  backend: dp # pytorch lightning backend, ddp, dp or None
  validation_interval: 10 # perform validation every X epoch, 1 default
  weak_split: 0.9
  maestro_split: 0.9
  seed: 42
  deterministic: False
  precision: 32
  mixup: soft # Soft mixup gives the ratio of the mix to the labels, hard mixup gives a 1 to every label present.
  mixup_prob: 0.5
  obj_metric_synth_type: intersection
  obj_metric_maestro_type: fmo
  enable_progress_bar: True
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
