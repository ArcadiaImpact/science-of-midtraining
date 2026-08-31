"""CPU-only static contracts for the Dispatch final-v1 pod setup."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = (
    REPO_ROOT
    / "experiments"
    / "prior_coins"
    / "dispatch_final_v1"
    / "pod"
    / "setup.sh"
)


def test_setup_shell_syntax_is_valid():
    subprocess.run(["bash", "-n", SETUP], check=True)


def test_flash_attn_wheel_is_pinned_verified_and_overrideable():
    setup = SETUP.read_text()

    assert "FLASH_ATTN_WHEEL_SOURCE=${FLASH_ATTN_WHEEL_SOURCE-" in setup
    assert re.search(
        r"python4-build-cache/resolve/[0-9a-f]{40}/", setup
    )
    assert (
        "FLASH_ATTN_WHEEL_SHA256="
        "56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611"
        in setup
    )
    assert 'actual_sha256=$(sha256sum -- "$wheel"' in setup
    assert "HARD prebuilt-wheel rejection: sha256 mismatch" in setup
    assert "verified prebuilt wheel failed its import/version probe" in setup


def test_flash_attn_rejection_falls_back_to_the_same_source_version():
    setup = SETUP.read_text()

    selection = re.search(
        r"if install_prebuilt_flash_attn; then\n"
        r".*?\nelse\n"
        r"  install_flash_attn_from_source\nfi",
        setup,
        flags=re.DOTALL,
    )
    assert selection is not None
    assert "FALLING BACK TO flash-attn==$FLASH_ATTN_VERSION SOURCE BUILD" in setup
    assert '--no-build-isolation "flash-attn==$FLASH_ATTN_VERSION"' in setup
    assert "flash-attn==2.8.3" in setup
