
# RE-EDPM CT Denoiser

A course project implementing a lightweight residual-conditioned
U-Net for synthetic-noise CT image denoising.

## Features
- CT image upload
- Grayscale preprocessing
- Neural network denoising
- Original and denoised image comparison
- Downloadable output

## Model
- Lightweight residual-conditioned U-Net
- 273,361 parameters
- 96 x 96 input resolution
- PyTorch implementation
- Hybrid MSE + MAE + Total Variation loss

## Dataset
A public chest CT image dataset containing 1,000 images.
Synthetic Gaussian noise is added to generate training pairs.

## Evaluation
The model was evaluated on 315 test images.

| Method | PSNR (dB) | SSIM | MAE |
|---|---:|---:|---:|
| Noisy Input | 25.1449 | 0.6337 | 0.0437 |
| Gaussian Filter | 27.1073 | 0.7611 | 0.0328 |
| Median Filter | 28.3486 | 0.7873 | 0.0263 |
| Direct U-Net | 28.9781 | 0.8101 | 0.0263 |

## Disclaimer
This is an academic research prototype.
It is not intended for clinical diagnosis or medical decision-making.
