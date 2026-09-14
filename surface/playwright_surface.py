"""
PlaywrightSurface -- the ONLY file in this project that imports Playwright.

Everything above `surface.base.ComputerSurface` (agent, tools, artifact,
replay) talks exclusively in terms of Target/Observation/ActionResult. This
file's job is to turn those surface-independent shapes into real browser
actions, and turn real browser state back into an Observation.

Locator resolution implements the tiered strategy documented in
`surface/observation.py`:
    1. role + accessible name  -> Playwright get_by_role()
    2. label                   -> <label for=...> (Playwright get_by_label)
                                   OR, if that finds nothing, a legacy
                                   "nearest preceding table-cell text" XPath
                                   heuristic (many legacy/back-office screens
                                   lay out forms as label-cell / value-cell
                                   table rows with no <label> at all)
    3. text                    -> exact visible text match
    4. css                     -> stable-attribute fallback
Ambiguous resolution (>1 match with no `nth` given) is a controlled,
reported failure -- never a silent guess.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

from playwright.sync_api import Locator, Page, sync_playwright

from surface.observation import (
    ActionResult,
    ActionStatus,
    Control,
    DialogInfo,
    Observation,
    Target,
)

# JS run in-page to build a semantic control list without shipping raw HTML
# back to Python. Computes a best-effort accessible name per element using
# the same priority order a browser's accessibility tree would use, with an
# added legacy fallback (nearest preceding <td> text in the same table row)
# for elements that have no standard accessible name at all.
_EXTRACT_CONTROLS_JS = r"""
() => {
  function rowLabel(el) {
    let cell = el.closest('td');
    if (!cell) return "";
    let sib = cell.previousElementSibling;
    while (sib && sib.tagName !== 'TD' && sib.tagName !== 'TH') {
      sib = sib.previousElementSibling;
    }
    return sib ? sib.textContent.trim() : "";
  }
  function accessibleName(el) {
    const ariaLabel = el.getAttribute('aria-label');
    if (ariaLabel) return ariaLabel.trim();
    if (el.id) {
      const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (lbl) return lbl.textContent.trim();
    }
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'input' && (type === 'submit' || type === 'button')) {
      return (el.value || '').trim();
    }
    if (tag === 'button' || tag === 'a') {
      return el.textContent.trim();
    }
    if (el.placeholder) return el.placeholder.trim();
    const rl = rowLabel(el);
    if (rl) return rl;
    return '';
  }
  function roleOf(el) {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || 'text').toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'button') return 'button';
    if (tag === 'input') {
      if (type === 'submit' || type === 'button') return 'button';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      return 'textbox';
    }
    return tag;
  }
  const els = Array.from(document.querySelectorAll(
    'input, button, select, textarea, a[href]'
  ));
  return els.map(el => {
    let value = null;
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      if (type !== 'submit' && type !== 'button') {
        value = el.value || null;
      }
    }
    return {
      role: roleOf(el),
      name: accessibleName(el),
      value: value,
      enabled: !el.disabled,
    };
  });
}
"""

_VISIBLE_TEXT_JS = r"""
() => {
  function isVisible(el) {
    const style = window.getComputedStyle(el);
    return style.display !== 'none' && style.visibility !== 'hidden';
  }
  const skip = new Set(['SCRIPT', 'STYLE', 'INPUT', 'BUTTON', 'SELECT', 'OPTION']);
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let out = [];
  let node;
  while ((node = walker.nextNode())) {
    const parent = node.parentElement;
    if (!parent || skip.has(parent.tagName)) continue;
    if (!isVisible(parent)) continue;
    const t = node.textContent.trim();
    if (t) out.push(t);
  }
  return out.join(' | ');
}
"""


class PlaywrightSurface:
    """Concrete ComputerSurface backed by a real Chromium browser session."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        headless: bool = True,
        default_timeout_ms: int = 10_000,
        screenshot_dir: str = "evidence/_screenshots",
    ):
        self.base_url = base_url
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._context = self._browser.new_context()
        self._page: Page = self._context.new_page()
        self._page.set_default_timeout(default_timeout_ms)

        self._last_status: Optional[int] = None
        self._pending_dialog: Optional[DialogInfo] = None

        self._page.on("response", self._on_response)
        self._page.on("dialog", self._on_dialog)

    # ---------------------------------------------------------------- events

    def _on_response(self, response) -> None:
        try:
            if (
                response.request.resource_type == "document"
                and response.frame == self._page.main_frame
            ):
                self._last_status = response.status
        except Exception:
            pass

    def _on_dialog(self, dialog) -> None:
        # Safe default: auto-dismiss and record it. A human-controlled
        # session (Phase 9) can override this policy to leave dialogs for
        # manual handling; for autonomous discovery/replay, silently
        # *accepting* an unknown dialog would be unsafe, so we dismiss.
        self._pending_dialog = DialogInfo(
            dialog_type=dialog.type,
            message=dialog.message,
            auto_dismissed=True,
        )
        dialog.dismiss()

    # ------------------------------------------------------------- resolve

    @staticmethod
    def _xpath_literal(s: str) -> str:
        if '"' not in s:
            return f'"{s}"'
        if "'" not in s:
            return f"'{s}'"
        parts = s.split('"')
        return "concat(" + ', \'"\', '.join(f'"{p}"' for p in parts) + ")"

    def _resolve_by_row_label(self, label: str) -> Optional[Locator]:
        """Legacy heuristic: find every <td>/<th> whose text equals `label`,
        then look at each following-sibling <td>. If any of those value
        cells contains an *enabled* interactive control, resolve to the set
        of such controls (for click/type). Otherwise fall back to the set
        of value cells themselves (for extract of read-only text).

        Deliberately returns the full matching set rather than `.first`:
        the caller's ambiguity check (>1 match, no `nth`) is what decides
        whether this is a clean resolution or a reported ambiguity -- this
        heuristic must not silently pick a winner itself.

        Disabled controls (e.g. Phase 1's intentional decoy field) are
        excluded here because they are not actionable targets for click/type
        -- this mirrors Playwright's own actionability requirements, it is
        not "guessing" between two otherwise-equal candidates.
        """
        # Tolerate the common legacy convention of a trailing colon on label
        # cells (e.g. "Member ID:") without requiring the caller to guess
        # whether a given screen uses one. This is a documented normalization,
        # not a fuzzy/best-effort match -- exact text (with or without a
        # single trailing colon) is still required.
        lit = self._xpath_literal(label)
        lit_colon = self._xpath_literal(label + ":")
        cond = (
            f"normalize-space(string(.))={lit} or "
            f"normalize-space(string(.))={lit_colon}"
        )
        cells = self._page.locator(
            f"xpath=//td[{cond}]/following-sibling::td[1] | "
            f"//th[{cond}]/following-sibling::td[1]"
        )
        if cells.count() == 0:
            return None
        controls = cells.locator(
            "input:not([disabled]), select:not([disabled]), "
            "textarea:not([disabled]), button:not([disabled]), a"
        )
        if controls.count() > 0:
            return controls
        return cells

    def _resolve(self, target: Target) -> tuple[Optional[Locator], Optional[int], int]:
        """Returns (locator, tier_used, match_count). locator is None if
        nothing matched at any tier. If match_count > 1 and target.nth is
        None, the caller must treat this as AMBIGUOUS rather than guessing."""
        page = self._page

        if target.role and target.name:
            loc = page.get_by_role(target.role, name=target.name, exact=target.exact)
            n = loc.count()
            if n > 0:
                return self._apply_nth(loc, target), 1, n

        if target.label:
            loc = page.get_by_label(target.label, exact=target.exact)
            n = loc.count()
            if n == 0:
                loc = self._resolve_by_row_label(target.label)
                n = loc.count() if loc is not None else 0
            if n > 0:
                return self._apply_nth(loc, target), 2, n

        if target.text:
            loc = page.get_by_text(target.text, exact=target.exact)
            n = loc.count()
            if n > 0:
                return self._apply_nth(loc, target), 3, n

        if target.css:
            loc = page.locator(target.css)
            n = loc.count()
            if n > 0:
                return self._apply_nth(loc, target), 4, n

        return None, None, 0

    @staticmethod
    def _apply_nth(loc: Locator, target: Target) -> Locator:
        if target.nth is not None:
            return loc.nth(target.nth)
        return loc

    def _resolve_or_result(
        self, target: Target
    ) -> tuple[Optional[Locator], Optional[ActionResult]]:
        """Shared resolution + error-reporting used by click/type/extract."""
        loc, tier, n = self._resolve(target)
        if loc is None:
            return None, ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"No element matched target ({target.describe()}).",
            )
        if n > 1 and target.nth is None:
            return None, ActionResult(
                status=ActionStatus.AMBIGUOUS,
                message=(
                    f"Target ({target.describe()}) matched {n} elements. "
                    f"Refusing to guess -- add `nth` to disambiguate."
                ),
                target_tier_used=tier,
            )
        return loc, None

    # -------------------------------------------------------------- actions

    def observe(self) -> Observation:
        page = self._page
        raw_controls = page.evaluate(_EXTRACT_CONTROLS_JS)
        controls = [
            Control(
                role=c["role"],
                name=c["name"] or "(no accessible name)",
                value=c["value"],
                enabled=c["enabled"],
            )
            for c in raw_controls
        ]
        text_summary = page.evaluate(_VISIBLE_TEXT_JS)
        text_summary = re.sub(r"\s*\|\s*", " | ", text_summary).strip()
        if len(text_summary) > 1200:
            text_summary = text_summary[:1200] + " …[truncated]"

        dialog = self._pending_dialog
        self._pending_dialog = None  # report once

        return Observation(
            url=page.url,
            title=page.title(),
            status_code=self._last_status,
            controls=controls,
            text_summary=text_summary,
            dialog=dialog,
        )

    def navigate(self, url: str) -> ActionResult:
        full_url = urljoin(self.base_url + "/", url) if self.base_url else url
        try:
            resp = self._page.goto(full_url, wait_until="load")
            status = resp.status if resp else self._last_status
            return ActionResult(
                status=ActionStatus.OK,
                message=f"Navigated to {full_url} ({status})",
            )
        except Exception as e:  # noqa: BLE001
            return ActionResult(
                status=ActionStatus.TIMEOUT if "Timeout" in type(e).__name__ else ActionStatus.ERROR,
                message=f"navigate({full_url}) failed: {e}",
            )

    def click(self, target: Target) -> ActionResult:
        loc, err = self._resolve_or_result(target)
        if err:
            return err
        try:
            loc.click()
            try:
                self._page.wait_for_load_state("load", timeout=5_000)
            except Exception:
                pass  # click may not have triggered navigation; that's fine
            return ActionResult(
                status=ActionStatus.OK,
                message=f"Clicked target ({target.describe()}).",
                target_tier_used=target.strategy_tier(),
            )
        except Exception as e:  # noqa: BLE001
            status = ActionStatus.TIMEOUT if "Timeout" in type(e).__name__ else ActionStatus.ERROR
            return ActionResult(status=status, message=f"click failed: {e}")

    def type(self, target: Target, value: str) -> ActionResult:
        loc, err = self._resolve_or_result(target)
        if err:
            return err
        try:
            loc.fill(value)
            return ActionResult(
                status=ActionStatus.OK,
                message=f"Typed into target ({target.describe()}).",
                target_tier_used=target.strategy_tier(),
            )
        except Exception as e:  # noqa: BLE001
            status = ActionStatus.TIMEOUT if "Timeout" in type(e).__name__ else ActionStatus.ERROR
            return ActionResult(status=status, message=f"type failed: {e}")

    def extract(self, target: Target) -> ActionResult:
        loc, err = self._resolve_or_result(target)
        if err:
            return err
        try:
            if not loc.is_visible():
                # The element exists in the DOM but is not visible to a
                # human operator (e.g. content hidden behind a dismissed
                # confirmation dialog). Treat this the same as "not present"
                # rather than silently returning an empty/misleading value.
                return ActionResult(
                    status=ActionStatus.NOT_FOUND,
                    message=(
                        f"Target ({target.describe()}) matched an element "
                        f"that is not visible (hidden on the page)."
                    ),
                    target_tier_used=target.strategy_tier(),
                )
            tag = loc.evaluate("el => el.tagName.toLowerCase()")
            if tag in ("input", "textarea"):
                value = loc.input_value()
            else:
                value = loc.inner_text().strip()
            return ActionResult(
                status=ActionStatus.OK,
                message=f"Extracted from target ({target.describe()}).",
                data=value,
                target_tier_used=target.strategy_tier(),
            )
        except Exception as e:  # noqa: BLE001
            return ActionResult(status=ActionStatus.ERROR, message=f"extract failed: {e}")

    def screenshot(self, path: Optional[str] = None) -> str:
        if path is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = str(self.screenshot_dir / f"screenshot_{ts}_{int(time.time() * 1000) % 1000}.png")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._page.screenshot(path=path)
        return path

    def close(self) -> None:
        try:
            self._context.close()
        finally:
            try:
                self._browser.close()
            finally:
                self._pw.stop()
