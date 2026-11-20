
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import io
import math


# =========================================
# CONFIGURATION
# =========================================

st.set_page_config(
    page_title="RE-EDPM CT Denoiser",
    page_icon="🔬",
    layout="wide"
)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 96
MODEL_PATH = "best_model.pth"


# =========================================
# MODEL ARCHITECTURE
# =========================================

class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):

        half_dim = self.dim // 2

        frequencies = torch.exp(
            -math.log(10000) *
            torch.arange(half_dim, device=t.device) / half_dim
        )

        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=1
        )

        return self.mlp(embedding)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )

        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )

        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_projection = nn.Linear(
            time_dim, out_channels
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):

        h = self.conv1(x)
        h = self.norm1(h)

        h = h + self.time_projection(t_emb)[:, :, None, None]

        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return F.silu(h + self.shortcut(x))


class ResidualUNet(nn.Module):

    def __init__(self, base_channels=16, time_dim=128):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)

        self.enc1 = ResidualBlock(
            2, base_channels, time_dim
        )

        self.down1 = nn.Conv2d(
            base_channels, base_channels * 2,
            4, stride=2, padding=1
        )

        self.enc2 = ResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_dim
        )

        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 4,
            4, stride=2, padding=1
        )

        self.middle = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim
        )

        self.dec2 = ResidualBlock(
            base_channels * 6,
            base_channels * 2,
            time_dim
        )

        self.dec1 = ResidualBlock(
            base_channels * 3,
            base_channels,
            time_dim
        )

        self.output = nn.Conv2d(
            base_channels, 1, 1
        )

    def forward(self, x_t, x_ld, t):

        x = torch.cat([x_t, x_ld], dim=1)

        t_emb = self.time_embedding(t)

        e1 = self.enc1(x, t_emb)

        e2 = self.down1(e1)
        e2 = self.enc2(e2, t_emb)

        b = self.down2(e2)
        b = self.middle(b, t_emb)

        d2 = F.interpolate(
            b, size=e2.shape[2:],
            mode="bilinear", align_corners=False
        )

        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb)

        d1 = F.interpolate(
            d2, size=e1.shape[2:],
            mode="bilinear", align_corners=False
        )

        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb)

        return torch.sigmoid(self.output(d1))


# =========================================
# LOAD TRAINED MODEL
# =========================================

@st.cache_resource
def load_model():

    model = ResidualUNet(
        base_channels=16,
        time_dim=128
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model


# =========================================
# IMAGE PREPROCESSING
# =========================================

def preprocess_image(image):

    # Preserve original image dimensions
    image = image.convert("L")

    arr = np.array(image).astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        arr
    ).unsqueeze(0).unsqueeze(0)

    original_size = image.size

    return tensor, original_size



# =========================================
# DENOISING
# =========================================


def denoise_image(model, tensor):

    PATCH_SIZE = 96
    STRIDE = 64

    _, _, height, width = tensor.shape

    pad_h = max(0, PATCH_SIZE - height)
    pad_w = max(0, PATCH_SIZE - width)

    padded = F.pad(
        tensor,
        (0, pad_w, 0, pad_h),
        mode="replicate"
    )

    padded_h = padded.shape[2]
    padded_w = padded.shape[3]

    def get_positions(length):

        if length <= PATCH_SIZE:
            return [0]

        positions = list(range(
            0, length - PATCH_SIZE + 1, STRIDE
        ))

        final_position = length - PATCH_SIZE

        if positions[-1] != final_position:
            positions.append(final_position)

        return positions

    y_positions = get_positions(padded_h)
    x_positions = get_positions(padded_w)

    # Smooth blending window
    window_1d = torch.hann_window(
        PATCH_SIZE + 2,
        periodic=False
    )[1:-1]

    window = (
        window_1d[:, None] *
        window_1d[None, :]
    )

    window = window.unsqueeze(0).unsqueeze(0)

    window = window.to(
        device=tensor.device,
        dtype=tensor.dtype
    )

    output = torch.zeros_like(padded)
    weights = torch.zeros_like(padded)

    model.eval()

    with torch.no_grad():

        for y in y_positions:

            for x in x_positions:

                patch = padded[
                    :, :,
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                t = torch.full(
                    (patch.shape[0],),
                    20,
                    dtype=torch.long,
                    device=patch.device
                )

                prediction = model(
                    patch, patch, t
                )

                output[
                    :, :,
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ] += prediction * window

                weights[
                    :, :,
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ] += window

    output = output / weights.clamp_min(1e-8)

    return output[:, :, :height, :width]





# =========================================
# USER INTERFACE
# =========================================

st.title("🔬 RE-EDPM CT Denoiser")

st.markdown(
    """
    ### Residual-Guided Deep Learning for CT Image Denoising

    Upload a CT image to process it using our trained
    lightweight residual-conditioned U-Net.
    """
)

st.info(
    "Research demonstration only. This application uses a model "
    "trained with synthetic Gaussian noise and is not intended "
    "for clinical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload a CT image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    tensor, original_size = preprocess_image(image)

    st.subheader("Input CT image")

    if st.button("Denoise CT Image", type="primary"):

        with st.spinner("Processing image..."):

            try:
                model = load_model()

                output = denoise_image(model, tensor)

                output_array = (
                    output[0, 0].cpu().numpy() * 255
                ).clip(0, 255).astype(np.uint8)

                output_image = Image.fromarray(output_array)

                # Absolute difference map
                original_array = (
                    tensor[0, 0].cpu().numpy()
                )

                denoised_array = (
                    output[0, 0].cpu().numpy()
                )

                difference = np.abs(
                    original_array - denoised_array
                )

                # Enhance visibility of small changes
                difference_display = np.clip(
                    difference * 4, 0, 1
                )

                difference_image = Image.fromarray(
                    (difference_display * 255).astype(np.uint8)
                )

                st.success("Image processing completed!")

                st.subheader("Image comparison")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Original image**")
                    st.image(
                        image,
                        use_container_width=True
                    )

                with col2:
                    st.markdown("**Denoised image**")
                    st.image(
                        output_image,
                        use_container_width=True
                    )

                with col3:
                    st.markdown("**Difference map**")
                    st.image(
                        difference_image,
                        use_container_width=True
                    )

                st.caption(
                    "The difference map displays absolute pixel "
                    "changes, enhanced by a factor of four for visibility. "
                    "It is not an anatomical or diagnostic map."
                )

                st.divider()

                st.subheader("Download result")

                buffer = io.BytesIO()
                output_image.save(buffer, format="PNG")

                st.download_button(
                    "Download denoised CT image",
                    data=buffer.getvalue(),
                    file_name="denoised_ct.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Processing failed: {e}")

st.divider()

st.caption(
    "Course project prototype | Synthetic-noise CT denoising | "
    "Not for clinical use"
)
