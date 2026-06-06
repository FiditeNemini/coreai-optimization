# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""General test utilities."""

import importlib.util

import torch

COREAI_AVAILABLE = importlib.util.find_spec("coreai") is not None


class SNRBelowThresholdError(AssertionError):
    """Raised when SNR or PSNR is below the required threshold."""

    def __init__(
        self,
        snr: float,
        psnr: float,
        snr_thresh: float,
        psnr_thresh: float,
        prefix: str = "",
    ) -> None:
        if snr <= snr_thresh:
            msg = f"{prefix}SNR {snr:.2f} below threshold {snr_thresh} (PSNR: {psnr:.2f})"
        else:
            msg = f"{prefix}PSNR {psnr:.2f} below threshold {psnr_thresh} (SNR: {snr:.2f})"
        super().__init__(msg)


def compute_snr_psnr(
    data: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    """Compute Signal-to-Noise Ratio and Peak Signal-to-Noise Ratio.

    Compares a data tensor against a reference tensor, treating their difference
    as noise for SNR/PSNR calculation.

    Args:
        data: Data tensor to compare
        reference: Reference tensor

    Returns:
        Tuple of (SNR, PSNR) values

    """
    assert len(data) == len(reference), f"Tensor length mismatch: {len(data)} vs {len(reference)}"

    eps = 1e-5
    eps2 = 1e-10
    noise = data - reference
    noise_var = torch.sum(noise**2) / len(noise)
    signal_energy = torch.sum(reference**2) / len(reference)
    max_signal_energy = torch.amax(reference**2)
    snr = 10 * torch.log10((signal_energy + eps) / (noise_var + eps2))
    psnr = 10 * torch.log10((max_signal_energy + eps) / (noise_var + eps2))
    return snr.item(), psnr.item()


def verify_snr_psnr(
    data: torch.Tensor,
    reference: torch.Tensor,
    snr_thresh: float,
    psnr_thresh: float,
    label: str = "",
) -> None:
    """Verify SNR and PSNR meet thresholds.

    Args:
        data: Data tensor to compare (will be flattened)
        reference: Reference tensor (will be flattened)
        snr_thresh: Minimum acceptable SNR value
        psnr_thresh: Minimum acceptable PSNR value
        label: Optional label for error messages

    Raises:
        SNRBelowThresholdError: If SNR or PSNR is below the threshold
    """
    data_flat = data.float().flatten()
    reference_flat = reference.float().flatten()

    snr, psnr = compute_snr_psnr(data_flat, reference_flat)

    prefix = f"{label}: " if label else ""

    if snr <= snr_thresh or psnr <= psnr_thresh:
        raise SNRBelowThresholdError(snr, psnr, snr_thresh, psnr_thresh, prefix)
