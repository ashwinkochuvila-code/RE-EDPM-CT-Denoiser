
# RE-EDPM CT Denoiser

A course project implementing a lightweight residual-conditioned U-Net for synthetic-noise CT image denoising, inspired by the paper *Low-dose CT imaging using a regularization-enhanced efficient diffusion probabilistic model*.

## Project Overview

The project explores deep-learning-based CT image denoising using a lightweight convolutional residual U-Net. A Streamlit application provides an interface for uploading CT images and viewing the denoised output.

This is a simplified, paper-inspired implementation, not a complete reproduction of the original RE-EDPM architecture or diffusion sampling procedure.

## Model Architecture

- Lightweight residual-conditioned U-Net
- 273,361 trainable parameters
- 96 × 96 training image resolution
- Time-conditioned residual blocks
- PyTorch implementation
- Hybrid MSE + MAE + Total Variation loss

## Dataset

- Public chest CT image dataset
- 1,000 images across four categories
- Training: 613 images
- Validation: 72 images
- Testing: 315 images
- Synthetic Gaussian noise used to construct noisy-clean training pairs

The dataset does not contain paired low-dose and full-dose CT acquisitions. Therefore, the experiments do not represent clinical low-dose CT reconstruction.

## Inference

The application uses:
- Grayscale image preprocessing
- Overlapping 96 × 96 patches
- Stride of 64 pixels
- Hann-window weighted overlap blending
- Direct U-Net inference at timestep 20

## Full Test-Set Evaluation

Evaluation was performed on all 315 held-out images using synthetic Gaussian noise with a standard deviation of 0.05.

| Method | PSNR (dB) | SSIM | MAE |
|---|---:|---:|---:|
| Noisy Input | 26.8840 | 0.4765 | 0.0338 |
| Gaussian Filter | 31.8669 | 0.7269 | 0.0196 |
| Median Filter | 32.3423 | 0.7570 | 0.0177 |
| Resize U-Net | 24.9754 | 0.5880 | 0.0382 |
| Patch U-Net | 27.5320 | 0.6704 | 0.0335 |

### Observations

- Patch-based inference outperformed resize-based inference across all three metrics.
- The U-Net improved SSIM and PSNR relative to the noisy input.
- Gaussian and median filtering achieved better results than the neural network.
- Median filtering achieved the strongest overall performance in this evaluation.

## Streamlit Application

The application supports:
- CT image upload
- Grayscale preprocessing
- Patch-based denoising
- Original and denoised image comparison
- Pixel-wise difference map
- Downloadable denoised output

## Repository Contents

- `app.py` – Streamlit application
- `best_model.pth` – Trained model checkpoint
- `requirements.txt` – Python dependencies
- `final_results.csv` – Earlier evaluation results
- `final_comparison.png` – Earlier comparison figure
- `training_loss.png` – Training loss plot
- `final_evaluation_backup.zip` – Full test-set experimental results and figures

## Limitations

- The implementation uses synthetic Gaussian noise rather than paired clinical LDCT–FDCT data.
- The network is a convolutional residual U-Net rather than the paper's Swin-based architecture.
- The original four-step reverse diffusion procedure has not been reproduced.
- Classical filtering methods outperformed the neural network in the full test-set evaluation.
- Fine anatomical detail preservation remains a limitation.

## Disclaimer

This application is an academic research prototype. It is not clinically validated and must not be used for medical diagnosis or clinical decision-making.
