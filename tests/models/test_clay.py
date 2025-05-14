# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from unittest.mock import patch # Added for potential future mocking if needed

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

# Corrected import: Assuming clay.py is in the same directory or accessible
# If clay.py is in a subdirectory (e.g., 'models'), use 'from .models.clay import ...'
from torchgeo.models import (
    ClayMAE, # Import the main class if needed directly
    ClayMAEBase_Weights,
    clay_mae_base,
    clay_mae_large,
)


# Helper fixture if torch.hub download needs mocking (not strictly needed by current mocked_weights)
@pytest.fixture
def load_state_dict_from_url() -> None:
    with patch("torch.hub.load_state_dict_from_url"):
        yield

class TestClayMAE:
    @pytest.fixture(params=[ClayMAEBase_Weights.PRETRAIN]) # Use the specific enum member
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        weights: WeightsEnum,
        # load_state_dict_from_url fixture is implicitly used by weights.get_state_dict
        # but we mock the URL itself below, making the direct network call unnecessary for this fixture
    ) -> WeightsEnum:
        """Mocks the weights object to load from a local file."""
        path = tmp_path / f'{weights}.pth'
        # Create a dummy state dict for the base model structure
        # Note: This won't have the actual pretrained weights, just the right keys
        model = clay_mae_base()
        torch.save(model.state_dict(), path)

        # Mock the URL to point to the local dummy file
        # The specific attribute might be 'url' or '_url' depending on torchvision version
        # Accessing weights.value accesses the Weights object within the Enum
        if hasattr(weights.value, 'url'):
            monkeypatch.setattr(weights.value, 'url', str(path.as_uri()))
        elif hasattr(weights, 'url'): # Fallback for older/different structures
             monkeypatch.setattr(weights, 'url', str(path.as_uri()))
        else:
            # If 'url' attribute isn't found directly, try mocking the internal loader if necessary
            # This part might need adjustment based on how WeightsEnum retrieves the state dict
            print(f"Warning: Could not directly patch 'url' for {weights}. Mocking might be incomplete.")
            # As a robust alternative, you could mock 'torch.hub.load_state_dict_from_url'
            # directly within tests that need it, using the 'load_state_dict_from_url' fixture.

        # Ensure the mock uses the file URI scheme expected by torch.hub
        monkeypatch.setattr(weights.value, 'url', str(path.as_uri()))


        # Ensure the mock uses the file URI scheme expected by torch.hub
        # Check if weights is an enum member or the enum class itself
        target = weights.value if isinstance(weights, WeightsEnum) else weights
        if hasattr(target, 'url'):
             monkeypatch.setattr(target, 'url', path.as_uri()) # Use file URI
        else:
             # Handle cases where the url might be stored differently or accessed via a method
             # For simplicity, we assume .url attribute exists on the Weights object
             # If direct patching fails, more involved mocking of torch.hub might be needed
             print(f"Warning: Could not patch 'url' for {weights}. Mocking might be incomplete.")

        return weights

    def test_clay_mae_base(self) -> None:
        """Test initializing the base model."""
        model = clay_mae_base()
        assert isinstance(model, ClayMAE)

    def test_clay_mae_large(self) -> None:
        """Test initializing the large model."""
        model = clay_mae_large()
        assert isinstance(model, ClayMAE)

    def test_clay_mae_forward_pass(self) -> None:
        """Test a single forward pass with dummy data."""
        model = clay_mae_base()
        model.train() # Ensure model is in training mode for dropout, etc.
        batch_size = 2
        channels = 3
        height = width = 224 # Standard test size

        # Create a sample datacube matching the expected input structure
        datacube = {
            "pixels": torch.randn(batch_size, channels, height, width),
            "waves": torch.linspace(400, 700, channels) # Example wavelengths matching channels
            # Add other keys if your model specifically uses them in forward
            # "time": torch.randn(batch_size, 2),
            # "latlon": torch.randn(batch_size, 2),
            # "platform": ["dummy-platform"] * batch_size,
            # "gsd": torch.tensor([10.0] * batch_size),
        }

        loss, rec_loss, rep_loss = model(datacube)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(rec_loss, torch.Tensor)
        assert isinstance(rep_loss, torch.Tensor)
        assert not torch.isnan(loss) # Check loss is valid

    def test_clay_mae_weights(self, mocked_weights: WeightsEnum) -> None:
        """Test loading mocked weights."""
        # This call will use the mocked_weights fixture to load from the temp file
        model = clay_mae_base(weights=mocked_weights)
        assert isinstance(model, ClayMAE)
        # Add checks if specific layers were loaded if necessary

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        """Test transforms associated with weights (if any)."""
        # In the provided clay.py, transforms=None for ClayMAEBase_Weights.PRETRAIN
        # So this test might not do much unless transforms are added later.
        # Access transforms as an attribute
        transforms_obj = mocked_weights.transforms
        if transforms_obj is not None:
            # The original test assumed meta['in_chans'], which wasn't in your meta def.
            # We need to determine the expected number of channels differently if needed.
            # For now, let's assume 3 channels for a generic test if transforms exist.
            c = 3 # Placeholder: Adjust if needed based on actual transforms
            sample = {
                "pixels": torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224),
                "waves": torch.linspace(400, 700, c) # Match channels
                # Add other necessary keys for the transform
            }
            # If transforms_obj is callable (like torchvision's standard approach)
            if callable(transforms_obj):
                transformed_sample = transforms_obj(sample)
            else:
                # If it's some other object requiring different handling
                # Adjust this part based on how transforms are actually implemented
                # For now, assume it should be callable if not None
                raise TypeError("Expected transforms to be callable or None")

            assert isinstance(transformed_sample, dict)
            assert "pixels" in transformed_sample
            # Add more specific checks based on what the transform should do
        else:
            # If transforms are None, the test passes vacuously or we can skip/assert None.
            assert transforms_obj is None

    @pytest.mark.parametrize("patch_size", [8, 16])
    def test_clay_mae_different_patch_sizes(self, patch_size: int) -> None:
        """Test initializing with different patch sizes."""
        model = clay_mae_base(patch_size=patch_size)
        assert model.patch_size == patch_size
        assert isinstance(model, ClayMAE)
        # You could add a forward pass test here too if patch size affects it significantly

    @pytest.mark.parametrize("mask_ratio", [0.5, 0.75, 0.9])
    def test_clay_mae_different_mask_ratios(self, mask_ratio: float) -> None:
        """Test initializing with different mask ratios."""
        model = clay_mae_base(mask_ratio=mask_ratio)
        assert model.mask_ratio == mask_ratio
        assert isinstance(model, ClayMAE)

    def test_clay_mae_teacher_freeze(self) -> None:
        """Test that teacher model parameters are frozen."""
        model = clay_mae_base()
        for param in model.teacher.parameters():
            assert not param.requires_grad
        assert not model.teacher.training # Should be in eval mode

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires network and potentially GPU")
    def test_clay_mae_download(self, weights: WeightsEnum) -> None:
        """Test actual weight download (requires network)."""
        # This uses the real weights enum, not the mocked one
        model = clay_mae_base(weights=weights, progress=False)
        assert isinstance(model, ClayMAE)
        # Maybe check a parameter value against a known value if available?

    def test_clay_mae_channel_dropout(self) -> None:
        """Test forward pass robustness with channel dropout active."""
        model = clay_mae_base()
        model.train() # Ensure dropout is active
        batch_size = 2
        # Use more channels to make dropout more likely to affect things
        channels = 10
        height = width = 224

        datacube = {
            "pixels": torch.randn(batch_size, channels, height, width),
            "waves": torch.linspace(400, 900, channels) # Match channels
        }

        # Test multiple forward passes; dropout mask should change
        # We mainly check that it doesn't crash or produce NaNs
        for i in range(5):
            with patch('torch.rand', return_value=torch.tensor([0.95])): # Force dropout condition
                 loss, _, _ = model(datacube)
                 assert isinstance(loss, torch.Tensor)
                 assert not torch.isnan(loss), f"Loss is NaN on iteration {i} with forced dropout"
            # Test without forcing dropout as well
            loss_normal, _, _ = model(datacube)
            assert isinstance(loss_normal, torch.Tensor)
            assert not torch.isnan(loss_normal), f"Loss is NaN on iteration {i} normally"

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_clay_mae_shuffle_option(self, shuffle: bool) -> None:
        """Test the patch shuffling option during forward pass."""
        model = clay_mae_base(shuffle=shuffle)
        model.train()
        batch_size = 2
        channels = 3
        height = width = 224

        datacube = {
            "pixels": torch.randn(batch_size, channels, height, width),
            "waves": torch.linspace(400, 700, channels)
        }

        loss, rec_loss, rep_loss = model(datacube)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
