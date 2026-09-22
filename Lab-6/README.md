# RNN vs LSTM vs GRU: Sequence Learning and Video Understanding

Comparing SimpleRNN, LSTM, and GRU on activity recognition from smartphone
sensor data, extending that into video action recognition with a CNN
front-end, and finishing with a small sequence-to-sequence demo.

## What this is

Three separate but connected pieces of work:

1. **Activity recognition from motion sensors.** Six activities (walking,
   walking upstairs/downstairs, sitting, standing, laying), classified from
   raw accelerometer and gyroscope windows using the UCI HAR dataset.
2. **Action recognition from video.** A small set of action classes
   classified from short video clips, using a frozen pretrained CNN to
   extract per-frame features and a recurrent layer to model how those
   features change over time.
3. **Sequence-to-sequence reversal.** A synthetic task (reverse a short
   sequence of digits) used to demonstrate an encoder–decoder setup on its
   own, separate from the classification work above.

## Results

**Activity recognition** — same architecture, optimizer, batch size, and
epoch count for all three recurrent cells, only the cell itself changed:

| Model | Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| SimpleRNN | ~76–82% | ~73% | 1,974 |
| LSTM | ~94–95% | ~95% | 6,006 |
| GRU | ~95–96% | ~95% | 4,758 |

The gated models (LSTM, GRU) beat plain RNN by a wide margin, mainly
because they can hold onto information across the full 128-step window —
plain RNN loses that over long sequences. GRU matched LSTM's accuracy with
about 20% fewer parameters. The main error both share is confusing
sitting with standing, which comes from the sensor data itself (those two
postures look nearly identical on the raw signals) rather than from either
model.

**Video action recognition** — 5 classes, ~200 short clips, 10 frames
sampled per video, MobileNetV2 features feeding a GRU classifier: reached
100% accuracy on a 32-video test split. Worth treating as a working
pipeline demo rather than a real benchmark, given how small that test set
is — a repeat run caught one misclassification.

**Sequence reversal** — an LSTM encoder–decoder trained on 5,000 random
4-digit sequences reached 99.77% per-token accuracy and 99.07%
whole-sequence accuracy, which lines up almost exactly with what you'd
expect mathematically (getting every position right at once is harder
than getting each position right on average).

## Pipeline overview

**Sensor data:** raw inertial signal files → windowed into `(128, 9)`
tensors (128 time steps, 9 channels) → normalized using training-set
statistics → split 70/15/15 → fed into a recurrent classifier (32 units →
dropout → dense(16, ReLU) → dense(6, softmax)).

**Video:** 10 frames sampled per clip → resized to 224×224 → passed
through a frozen, pretrained MobileNetV2 to get a 1280-dim feature vector
per frame → the resulting `(10, 1280)` sequence goes into a GRU
classifier.

**Seq2seq:** integer sequence → embedding → LSTM encoder compresses it
into a context vector → that context drives an LSTM decoder → dense
softmax layer predicts a digit at each output position.

## Datasets

- **UCI Human Activity Recognition Using Smartphones** — used in its raw
  inertial-signal form (not the pre-computed 561-feature vectors), so the
  model gets a proper sequence rather than a flat table.
- **UCF101** (subset) — the full official archive is a ~6.5GB single file,
  impractical for a handful of classes, so a much smaller pre-packaged
  subset (~171MB, 10 action classes) was used instead, narrowed down to 5
  classes for this project: Basketball, BasketballDunk, Archery,
  BenchPress, BabyCrawling.

## Stack

TensorFlow/Keras, scikit-learn, OpenCV (video frame extraction), NumPy,
Matplotlib. pipeline rather than a robust benchmark.
