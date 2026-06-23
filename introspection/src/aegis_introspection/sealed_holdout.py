from __future__ import annotations

from pathlib import Path

from aegis_introspection.artifacts import ActivationArtifact, load_activation_artifact_allowing_sealed_holdout
from aegis_introspection.sealed_holdout_policy import (
    SEALED_HOLDOUT_TAG as SEALED_HOLDOUT_TAG,
)
from aegis_introspection.sealed_holdout_policy import (
    UNSEAL_FLAG as UNSEAL_FLAG,
)
from aegis_introspection.sealed_holdout_policy import (
    SealedHoldoutPolicyError,
)
from aegis_introspection.sealed_holdout_policy import (
    add_unseal_flag as add_unseal_flag,
)
from aegis_introspection.sealed_holdout_policy import (
    assert_unsealed_jsonl_tags as assert_unsealed_jsonl_tags,
)
from aegis_introspection.sealed_holdout_policy import (
    assert_unsealed_path as assert_unsealed_path,
)
from aegis_introspection.sealed_holdout_policy import (
    assert_unsealed_paths as assert_unsealed_paths,
)
from aegis_introspection.sealed_holdout_policy import (
    assert_unsealed_tag_rows as assert_unsealed_tag_rows,
)
from aegis_introspection.sealed_holdout_policy import (
    path_is_sealed_holdout as path_is_sealed_holdout,
)
from aegis_introspection.sealed_holdout_policy import (
    tag_rows_are_sealed_holdout as tag_rows_are_sealed_holdout,
)
from aegis_introspection.sealed_holdout_policy import (
    tags_are_sealed_holdout as tags_are_sealed_holdout,
)

SealedHoldoutError = SealedHoldoutPolicyError


def assert_unsealed_activation_artifact_tags(
    artifact: ActivationArtifact,
    allow_sealed_holdout: bool,
    context: str,
) -> None:
    assert_unsealed_tag_rows(
        tag_rows=artifact["tags"],
        allow_sealed_holdout=allow_sealed_holdout,
        context=context,
    )


def assert_unsealed_activation_artifact_path(path: Path, allow_sealed_holdout: bool, context: str) -> None:
    assert_unsealed_path(path=path, allow_sealed_holdout=allow_sealed_holdout, context=context)
    if allow_sealed_holdout:
        return
    assert_unsealed_activation_artifact_tags(
        artifact=load_activation_artifact_allowing_sealed_holdout(path),
        allow_sealed_holdout=False,
        context=context,
    )


def load_activation_artifact_with_unseal_policy(
    path: Path,
    allow_sealed_holdout: bool,
    context: str,
) -> ActivationArtifact:
    assert_unsealed_path(path=path, allow_sealed_holdout=allow_sealed_holdout, context=context)
    artifact = load_activation_artifact_allowing_sealed_holdout(path)
    assert_unsealed_activation_artifact_tags(
        artifact=artifact,
        allow_sealed_holdout=allow_sealed_holdout,
        context=context,
    )
    return artifact
