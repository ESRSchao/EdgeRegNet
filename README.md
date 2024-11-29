# EdgeRegNet: Edge Feature-based Multi-modal Registration Network between Images and LiDAR Point Clouds

This repository is the official implementation of [EdgeRegNet: Edge Feature-based Multi-modal
Registration Network between Images and LiDAR
Point Clouds]

## Environment
You can set up the Python environment using the following command:
``` python
conda create -n regis2D_y3D python==3.8.5 -y

conda activate regis2D_3D

pip install numpy pillow opencv-python scipy pandas matplotlib
```

---

If need GPU for training and testing, install the appropriate [PyTorch](https://pytorch.org/) version for your GPU drivers:

```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Test

Before the evaluation, you should download the KITTI dataset and nuScenes dataset, or use our proprocessed dataset by [this link](https://drive.google.com/file/d/1oadj5iqrW9XUMufQQB2nvVd1sucbapr3/view?usp=sharing), which is much more convenient.
In Addition, pretrained model also needs to be downloaded [here](https://drive.google.com/file/d/1Aj3a5sncsVISk-mGhEZVrZxggsTO6S3u/view?usp=sharing) and saved in ./ck folder.
After that, run test.py to start evalutation.
```python
python test.py
```
After this process, KITTI.csv will be created in the root folder, which contains the result of experment.

## Train
The details of training and testing will be provided after the paper is accepted.

## Contributing

MIT License

Copyright (c)
