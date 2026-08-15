"""Authentication dependency helpers for the governance review API."""

from fastapi import Header, HTTPException

from shared.security import get_review_keys, require_key

# Role ordering: reporter < reviewer < admin
ADMIN = "admin"
REVIEWER = "reviewer"


def require_role(required: str = REVIEWER):
    """FastAPI dependency requiring a valid review token + sufficient role.

    The token is checked against REVIEW_API_KEYS (constant time). If no keys are
    configured the token check is skipped for local/dev convenience.
    """

    def _dependency(
        authorization: str | None = Header(default=None, alias="Authorization"),
        x_reviewer_role: str | None = Header(default=None, alias="X-Reviewer-Role"),
    ) -> str:
        token = ""
        if authorization:
            token = authorization.removeprefix("Bearer ").strip()
        if not require_key(get_review_keys(), token):
            raise HTTPException(status_code=401, detail="missing or invalid token")
        role = (x_reviewer_role or "").strip().lower()
        allowed = {ADMIN} if required == ADMIN else {REVIEWER, ADMIN}
        if role not in allowed:
            raise HTTPException(status_code=403, detail=f"role '{role or 'none'}' insufficient")
        return role

    return _dependency