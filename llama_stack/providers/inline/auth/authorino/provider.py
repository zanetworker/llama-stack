# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api, InlineProviderSpec
from llama_stack.providers.inline.auth.authorino.config import AuthorinoProviderConfig


def get_provider_spec() -> InlineProviderSpec:
    """
    Get the provider specification for the Authorino auth provider.

    Returns:
        The provider specification
    """
    return InlineProviderSpec(
        api=Api.auth,
        provider_type="inline::authorino",
        config_class="llama_stack.providers.inline.auth.authorino.config.AuthorinoProviderConfig",
        module="llama_stack.providers.inline.auth.authorino.auth",
        pip_packages=[
            "httpx>=0.24.0",
            "pyjwt>=2.6.0",
            "cryptography>=40.0.0",
        ],
    )
