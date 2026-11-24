# FaceNote

FaceNote is an experimental face-recognition project built around the FaceNet embedding model and MTCNN face detection. It includes scripts for aligning image datasets, training classifiers, and identifying faces from a camera or video stream.

## How it works

1. MTCNN detects and aligns faces in images.
2. FaceNet converts each face into a 128-dimensional embedding.
3. A classifier uses those embeddings to assign identities. The camera demo currently uses a scikit-learn K-nearest-neighbors (KNN) model.

## Requirements

This repository uses a legacy Python and TensorFlow stack. The checked-in `requirements.txt` pins NumPy 1.16.2 and SciPy 1.2.1, but does **not** install TensorFlow. The original project targeted TensorFlow 1.10; several dependencies and APIs used by the code are no longer available in current releases. Set up a compatible, isolated environment rather than installing these dependencies into a modern Python environment, and install the compatible TensorFlow version separately.

Install the listed packages from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

The repository does not include trained FaceNet checkpoints or a trained KNN classifier. Obtain a compatible FaceNet model separately and provide its directory when running the scripts.

## Prepare image data

The alignment script expects an input directory organized into one subdirectory per identity, and writes aligned face images in the same class-folder layout. For example, after downloading and extracting LFW:

```bash
mkdir -p ~/datasets
wget -O ~/datasets/lfw.tgz http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf ~/datasets/lfw.tgz -C ~/datasets
```

Run the aligner from `core/` so its relative reference to the bundled MTCNN data resolves correctly:

```bash
cd core
python3 align_dataset_mtcnn.py ~/datasets/lfw ~/datasets/lfw_mtcnnalign_160 --image_size 160 --margin 32
cd ..
```

For FaceScrub, `script/facescrub_download.py` downloads images using the metadata files under `data/`. The downloader writes to `../datasets/facescrub` relative to the current working directory; check that location and the dataset's terms before running it. Align the downloaded images in the same way, using a path that matches where the data was saved.

## Train a FaceNet classifier

`train_facenote.sh` runs `core.facenet_train_classifier` with the FaceScrub and LFW datasets. Before running it, edit the dataset, log, model, and pretrained-checkpoint paths in the script to match your local setup. The referenced checkpoint is not included in this repository.

```bash
bash train_facenote.sh
```

## Train the KNN model used by the demo

The KNN training script expects a training directory with **exactly two identity subdirectories**. Update the paths in `train_classifier.sh`, or run the script directly:

```bash
python3 knn_train_classifier.py \
  --model_dir ~/models/facenet/20190114-154656/ \
  --train_dir ~/data/facenote/sample/
```

It saves the classifier to `models/knn_classifier.model`. The training code assigns numeric labels according to the directory listing order; the demo maps label `0` to `Chen` and label `1` to `Reenie`. Use matching identities or update that mapping in `facenote.py`.

## Run the camera demo

After training or placing the KNN model at `models/knn_classifier.model` and supplying a compatible FaceNet checkpoint, run:

```bash
bash run_demo.sh
```

`run_demo.sh` points to `~/models/facenet/20190114-154656/`; edit it if your checkpoint is elsewhere. The video source is currently hard-coded in `facenote.py` as an IP-camera URL. Replace it with your camera, video file, or stream before running, and keep any camera credentials private. Press **Esc** to exit; press **Space** to save the current frame.

## Repository layout

| Path | Purpose |
| --- | --- |
| `facenote.py` | Face recognition demo for a camera/video stream |
| `knn_train_classifier.py` | Creates the KNN identity classifier used by the demo |
| `core/` | FaceNet utilities, MTCNN detection, alignment, training, and validation |
| `data/` | MTCNN detector weights and dataset/training metadata |
| `models/` | Network definitions; trained model artifacts are not included |
| `script/` | Dataset download and preparation utilities |
| `train_facenote.sh` | FaceNet classifier training command |
| `train_classifier.sh` | KNN training command |
| `run_demo.sh` | Camera demo launch command |

## Limitations

This is a research prototype with legacy dependencies and paths that need local configuration. Face-recognition results depend on the quality and compatibility of the model, training data, and camera input. Use only images and biometric data you are authorized to process, and obtain consent where required.