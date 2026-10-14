"""Sign-in for the API: checks an OpenID Connect bearer token.

AUTH_MODE=off (default) keeps the open development behaviour. AUTH_MODE=oidc requires a valid token on every
call except the chat page itself, /healthz and /metrics. Works with any OIDC provider (Keycloak, Entra ID, Okta, ...):
it needs the issuer, the audience (the client id) and the address of the provider's signing keys.

A gateway such as oauth2-proxy does the browser login and forwards the ID token as `Authorization: Bearer ...`.
Programs can send an access token the same way. The signature, issuer, audience and expiry are all checked here, so
the API never trusts a header just because it arrived.
"""

import os
from typing import NamedTuple

import jwt
from fastapi import HTTPException, Request

MODE = os.environ.get("AUTH_MODE", "off")
ISSUER = os.environ.get("AUTH_ISSUER", "")
AUDIENCE = os.environ.get("AUTH_AUDIENCE", "")
JWKS_URL = os.environ.get("AUTH_JWKS_URL", "")
USER_CLAIM = os.environ.get("AUTH_USER_CLAIM", "preferred_username")
ROLES_CLAIM = os.environ.get(
    "AUTH_ROLES_CLAIM", "groups"
)  # dotted path, e.g. realm_access.roles
USER_ROLE = os.environ.get("AUTH_USER_ROLE", "")  # empty: any signed-in user may chat
ADMIN_ROLE = os.environ.get(
    "AUTH_ADMIN_ROLE", "rag-admins"
)  # may add and remove documents
# Where the page sends people to sign out (the gateway's sign-out, or the provider's logout address).
LOGOUT_URL = os.environ.get("AUTH_LOGOUT_URL", "/oauth2/sign_out")
# Only asymmetric algorithms: a shared-secret (HS*) token could be forged with the public key.
ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "ES256", "ES384"]

if MODE not in ("off", "oidc"):
    raise RuntimeError(f"AUTH_MODE must be 'off' or 'oidc', not {MODE!r}")
if MODE == "oidc" and not (ISSUER and AUDIENCE and JWKS_URL):
    raise RuntimeError(
        "AUTH_MODE=oidc needs AUTH_ISSUER, AUTH_AUDIENCE and AUTH_JWKS_URL"
    )

_jwks = (
    jwt.PyJWKClient(JWKS_URL, cache_keys=True, lifespan=3600)
    if MODE == "oidc"
    else None
)


class Identity(NamedTuple):
    user: str | None  # None when auth is off
    roles: frozenset[str]
    is_admin: bool


OPEN = Identity(None, frozenset(), True)


def _roles(claims: dict) -> frozenset[str]:
    node = claims
    for part in ROLES_CLAIM.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    values = node if isinstance(node, list) else []
    return frozenset(str(v).lstrip("/") for v in values)


def identity(request: Request) -> Identity:
    """FastAPI dependency: who is calling. Raises 401, 403 or 503."""
    if MODE == "off":
        return OPEN
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            401, "Sign in required.", headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        key = _jwks.get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            key,
            algorithms=ALGORITHMS,
            audience=AUDIENCE,
            issuer=ISSUER,
            leeway=30,
            options={"require": ["exp", "iss", "aud"]},
        )
    except jwt.PyJWKClientConnectionError:
        raise HTTPException(503, "The sign-in service is not reachable.") from None
    except jwt.PyJWTError:
        raise HTTPException(
            401, "Sign in again.", headers={"WWW-Authenticate": "Bearer"}
        ) from None
    user = claims.get(USER_CLAIM) or claims.get("sub")
    if not isinstance(user, str) or not user:
        raise HTTPException(401, "The token names no user.")
    roles = _roles(claims)
    is_admin = bool(ADMIN_ROLE) and ADMIN_ROLE in roles
    if USER_ROLE and USER_ROLE not in roles and not is_admin:
        raise HTTPException(403, "Your account is not allowed to use this service.")
    return Identity(user, roles, is_admin)


def require_admin(request: Request) -> Identity:
    ident = identity(request)
    if not ident.is_admin:
        raise HTTPException(403, "Only administrators can add or remove documents.")
    return ident
