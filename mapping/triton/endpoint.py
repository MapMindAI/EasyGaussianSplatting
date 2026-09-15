"""Where the stages in this package reach their Triton server.

Kept apart from the clients so a stage can resolve its endpoint, and fail on a
missing one, before importing anything that needs the models installed.
"""

import os


def add_endpoint_argument(parser, model_description):
    parser.add_argument(
        "--triton-url",
        default=None,
        help=f"Triton gRPC endpoint for the {model_description}; defaults to $TRITON_URL",
    )


def resolve_endpoint(triton_url):
    """`triton_url` or $TRITON_URL, refusing to start without one."""
    resolved = triton_url or os.environ.get("TRITON_URL")
    if not resolved:
        raise SystemExit("No Triton URL given; pass --triton-url or set TRITON_URL")
    return resolved
