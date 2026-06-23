"""Tests for checksum algorithms used by structurally-faithful formats."""

from __future__ import annotations

import pytest

from detect.dp_honey.checksums import CHECKSUM_ALGORITHMS, compute_checksum


def test_github_crc32_base62_known_vectors():
    fn = CHECKSUM_ALGORITHMS["github-crc32-base62"]
    # Vectors confirmed against therootcompany/base62-token.js real-world
    # GitHub-token fixtures; these are body-only, without the "ghp_" prefix.
    vectors = {
        "zQWBuTSOoRi4A9spHcVY5ncnsDkxkJ": "0mLq17",
        "adE7dp8rHP6gUTuPwxLTZjZdtya3sV": "0UQzQM",
        "H3xbiBdlzffNx7Y56iNsPw3joObj7U": "2nO29h",
    }
    for body, expected in vectors.items():
        assert fn(body, length=6) == expected
        assert len(fn(body, length=6)) == 6


def test_compute_checksum_unknown_algorithm_raises():
    with pytest.raises(KeyError):
        compute_checksum("not-an-algorithm", "abc", length=6)
