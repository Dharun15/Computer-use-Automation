"""
Fake Bank -- a deliberately "legacy-looking" local target application.

Purpose: give the computer-use automation system (discovery agent + replay
engine) a real, controllable UI to operate against. This is NOT the product
under test; it exists only as a proxy target standing in for a real
core-banking/back-office screen that has no API.

Design choices (intentional, to make locator-robustness a real problem):
  - Server-rendered HTML, table-based layout, no CSS framework.
  - No `data-testid` / `data-qa` attributes anywhere (legacy apps never have them).
  - Inputs are NOT wrapped in <label for=...> -- accessible name must be inferred
    from nearby text / value attributes, the way many legacy apps behave.
  - Synchronous full-page navigations only, no SPA/client-side routing.

Controlled failure modes (see app/data.py):
  - member not found            -> business outcome, not a crash
  - invalid member id format    -> validation error
  - slow_load member            -> artificial server-side delay (recoverable/retry)
  - confirm_dialog member       -> native JS confirm() interstitial on load
  - permission_denied member    -> 403 Access Denied page
  - app_error member            -> 500 Application Error page
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.data import MEMBERS, find_member, is_valid_member_id_format

app = FastAPI(title="Fake Bank - Legacy Member Servicing")
templates = Jinja2Templates(directory="app/templates")

SLOW_LOAD_SECONDS = 3.0


@app.get("/", response_class=HTMLResponse)
def search_page(request: Request):
    return templates.TemplateResponse(
        request, "search.html", {"error": None}
    )


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, member_id: str = Form(...)):
    member_id = member_id.strip()

    if not is_valid_member_id_format(member_id):
        return templates.TemplateResponse(
        request, "validation_error.html", {"member_id": member_id},
            status_code=200
    )

    member = find_member(member_id)
    if member is None:
        return templates.TemplateResponse(
        request, "not_found.html", {"member_id": member_id},
            status_code=200
    )

    return templates.TemplateResponse(
        request, "search_results.html", {"member": member}
    )


@app.get("/members/{member_id}", response_class=HTMLResponse)
def member_details(request: Request, member_id: str):
    member = find_member(member_id)

    if member is None:
        return templates.TemplateResponse(
        request, "not_found.html", {"member_id": member_id},
            status_code=200
    )

    if member.app_error:
        # Simulate an unhandled legacy application error.
        return templates.TemplateResponse(
        request, "app_error.html", {"member_id": member_id},
            status_code=500
    )

    if member.permission_denied:
        return templates.TemplateResponse(
        request, "permission_denied.html", {"member_id": member_id},
            status_code=403
    )

    if member.slow_load:
        time.sleep(SLOW_LOAD_SECONDS)

    return templates.TemplateResponse(
        request, "member_details.html", {"member": member}
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok", "members": len(MEMBERS)}
