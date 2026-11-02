# CIFAR-10 Image Classification

I built a CNN from scratch in TensorFlow/Keras to classify the 10 classes of the CIFAR-10 dataset. I trained a baseline model first, then added a simplified ResNet block (Part 2, Option A) and compared the two models.

## Project Structure

```
├── CIFAR10_Image_classification.ipynb   # the whole project: data, baseline, residual model, evaluation
├── requirements.txt                     # libraries needed to run the notebook
└── README.md
```

The notebook is organised in this order:

1. Mounting Google Drive (for saving checkpoints)
2. Loading the data and splitting it into train / validation / test
3. Preprocessing: normalisation, one-hot encoding, augmentation
4. Data loading with `tf.data`
5. Baseline model: building, training, evaluation, misclassified images
6. Part 2: residual block model: building, training, evaluation
7. Baseline vs residual comparison
8. Writing `requirements.txt`

Files saved to Google Drive during training:

| File | What it is |
|---|---|
| `best_model.keras` | Best baseline model (lowest validation loss) |
| `best_residual_model.keras` | Best residual model (lowest validation loss) |
| `residual_training_log.csv` | Loss and accuracy for every epoch of the residual model |

## How to Run

1. Open `CIFAR10_Image_classification.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type** and select a **T4 GPU**.
3. Run the first cell and allow Google Drive access. I save the checkpoints to Drive so they aren't lost if the Colab session resets.
4. Run all the cells (**Runtime → Run all**). On a free T4, each model takes about 45 minutes (around 54 seconds per epoch).

To run it outside Colab:

```
pip install -r requirements.txt
```

Then skip the Google Drive cell and change the two checkpoint paths (`/content/drive/MyDrive/...`) to a local folder.

## Data Pipeline

- **Dataset:** CIFAR-10, loaded with `keras.datasets.cifar10`. 60,000 colour images of size 32×32 in 10 classes.
- **Split:** 45,000 training images, 5,000 validation images (10% of the original training set) and 10,000 test images. The test set is only used at the end, to evaluate each model.
- **Normalisation:** I subtract the mean and divide by the standard deviation. Both are calculated from the training images only, so nothing leaks in from the validation or test sets.
- **Labels:** converted to one-hot vectors to match the 10 outputs of the model.
- **Augmentation (training data only):** random rotation (15°), width and height shifts (12%), zoom (10%), shear, horizontal flips, and small brightness and channel shifts, using `ImageDataGenerator`. This helps reduce overfitting.
- **Efficient loading:** the augmented batches are wrapped in a `tf.data` pipeline with `prefetch`, so the next batch is prepared while the GPU trains on the current one. The validation data is batched, cached and prefetched. Batch size is 64, which gives 704 batches per epoch.

## Model Design

### Baseline CNN

The baseline has 4 blocks. Each block has two 3×3 conv layers, each followed by BatchNorm, then 2×2 max pooling and dropout. At the end, a Flatten layer and a Dense layer with softmax give the 10 class probabilities.

| Block | Conv layers | Filters | Dropout |
|---|---|---|---|
| 1 | 2 | 32 | 0.2 |
| 2 | 2 | 64 | 0.3 |
| 3 | 2 | 128 | 0.4 |
| 4 | 2 | 256 | 0.5 |

Total parameters: **1,186,346**

Why I made these choices:

- **Filters grow as the image shrinks** (32×32 down to 2×2). The early layers learn simple patterns like edges, and the deeper layers learn more complex shapes, so they need more filters.
- **BatchNorm** after each conv layer keeps training stable and helps the model learn faster.
- **Dropout increases in deeper blocks** (0.2 to 0.5), because those layers have more parameters and overfit more easily.
- **L2 regularisation** (weight decay 0.0001) on every conv layer is another guard against overfitting.

### Residual Model 

The residual model has the same 4 blocks, filters and dropout rates as the baseline. The difference is that in each block, the second conv layer is replaced by a **simplified ResNet block**:

```
input → Conv → BatchNorm → ReLU → Conv → BatchNorm → (+ input) → ReLU
```

The block adds its own input back to its output. This is called a skip connection. It means the block only has to learn what to change, and it gives the training signal a shortcut path through the network, which makes deeper networks easier to train.

My version is simplified because it has **no 1×1 projection shortcut**. A full ResNet block uses one when the input and output have different shapes. I avoid that by putting a normal conv layer before each residual block, which sets the number of channels first, so the input and output of the block always have the same shape and can be added directly.

I built this model with the Keras **functional API**, because a skip connection can't be written as a `Sequential` model.

Since each residual block has 2 conv layers, the residual model has **12 conv layers** compared to 8 in the baseline. Total parameters: **1,972,106**.

## Training Setup

Both models were trained with exactly the same settings, so that the only difference is the residual blocks:

| Setting | Value |
|---|---|
| Epochs | 50 |
| Batch size | 64 |
| Optimizer | Adam, learning rate 0.0005 |
| Loss | Categorical cross-entropy |
| Metric | Accuracy |
| Learning rate scheduling | `ReduceLROnPlateau`: halves the learning rate if validation loss doesn't improve for 4 epochs (minimum 0.00001) |
| Early stopping | Stops if validation loss doesn't improve for 8 epochs, and restores the best weights |
| Checkpoints | `ModelCheckpoint` saves the best model to Google Drive whenever validation loss improves |
| Logging | `CSVLogger` saves every epoch's numbers for the residual model |

I used `model.fit` as the training loop. The loss, accuracy metric, learning rate scheduler and checkpoints are all passed into it.

I chose 50 epochs because of the GPU time limits on free Colab.

### Reproducibility

- `tf.keras.utils.set_random_seed(42)` at the start of the notebook (sets the Python, NumPy and TensorFlow seeds), and again before building the residual model.
- `random_state=101` for the train/validation split.
- `seed=42` for the augmentation generator.
- `requirements.txt` lists the libraries used.

## Results

| Model | Epochs | Best val accuracy | Test accuracy | Test loss | Parameters |
|---|---|---|---|---|---|
| Baseline | 50 | 87.84% | **86.62%** | 0.5149 | 1,186,346 |
| Residual | 50 | 87.46% | 86.10% | **0.5066** | 1,972,106 |

- The **baseline**'s best epoch was epoch 50, the last one, so its validation loss was still improving when training ended.
- The **residual model**'s best epoch was epoch 42. Early stopping triggered at epoch 50 and restored the epoch 42 weights.
- The baseline misclassified **1,338 of the 10,000** test images. The notebook shows 15 examples of these, with the true and predicted labels.
- As an extra check, I tested the baseline on a bird photo from the internet (resized to 32×32 and normalised the same way). It predicted **bird** with about 82% confidence.

### What the comparison shows

The two models performed almost the same. The baseline had slightly higher test accuracy (by about 0.5%), and the residual model had slightly lower test loss. Their validation loss and accuracy curves follow each other closely for all 50 epochs.

Why I think the residual block didn't make a big difference here:

- **The network is shallow.** Skip connections help most in very deep networks, where the training signal gets weaker as it passes back through many layers. With 8 to 12 conv layers, my baseline already trains well without them.
- **The difference is within noise.** Each model was trained once, with one seed. A gap of about half a percent can come from random variation alone.
- **Training was short.** The baseline was still improving at epoch 50, so a longer run might show a clearer difference between the two models.

## Limitations and Next Steps

- **Longer training and more runs:** train both models for 100+ epochs, and repeat each one with a few different seeds, to see whether the small difference is real.
- **Augmentation order:** the augmentation runs after normalisation, but the brightness and channel shift settings are meant for raw 0–255 pixel values. Moving augmentation before normalisation would be cleaner.
- **Augmentation inside tf.data:** the augmentation still runs in Python through `ImageDataGenerator`, and `tf.data` wraps it for batching and prefetching. Moving augmentation into Keras preprocessing layers inside the `tf.data` pipeline would be faster.
- **Normalisation per channel:** I use one mean and standard deviation for all three colour channels. Calculating them per channel is more common.
- **Deeper residual network:** skip connections should matter more in a deeper model.
- **Optional extensions not done:** mixed-precision training, distributed training, deployment (FastAPI / Flask / TF Serving), ONNX export and unit tests.
