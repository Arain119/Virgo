"""Shared helper utilities for the news crawler.

Includes time helpers, stable digests, request retry/backoff, header building,
and TLS relaxation for sites with legacy HTTPS configuration.
"""

from __future__ import annotations

import hashlib
import json
import random
import ssl
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter

from . import config

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

try:
    from urllib3.util import ssl_ as urllib3_ssl
except Exception:  # pragma: no cover - older urllib3
    urllib3_ssl = None


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_tz(dt: datetime, assume_local: ZoneInfo = SHANGHAI_TZ) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=assume_local).astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def make_digest(*parts: str) -> str:
    payload = "||".join(part or "" for part in parts)
    # Non-cryptographic digest used as a stable content fingerprint / primary key.
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()  # noqa: S324


def sanitize(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return " ".join(text.split())


def build_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "User-Agent": config.USER_AGENT,
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "close",
    }
    if extra:
        headers.update(extra)
    return headers


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    retries: int = config.MAX_RETRIES,
    allow_relax_tls: bool = True,
    **kwargs: Any,
) -> requests.Response:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = session.request(method, url, timeout=config.REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.SSLError as exc:  # pragma: no cover - TLS issues
            last_error = exc
            if allow_relax_tls and relax_tls_security(session):
                continue
            sleep_with_jitter((attempt + 1) * config.REQUEST_GAP_SECONDS)
        except requests.RequestException as exc:  # pragma: no cover - network branch
            last_error = exc
            sleep_with_jitter((attempt + 1) * config.REQUEST_GAP_SECONDS)
    if last_error is None:
        raise RuntimeError("Request failed but no exception details were captured.")
    raise last_error


def sleep_with_jitter(base: float) -> None:
    # Non-cryptographic jitter used only for pacing requests.
    jitter = random.uniform(0, 0.35)  # noqa: S311
    time.sleep(base + jitter)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def relax_tls_security(session: requests.Session) -> bool:
    if getattr(session, "_virgo_relaxed_tls", False):
        return False
    context = None
    if urllib3_ssl is not None:
        try:
            context = urllib3_ssl.create_urllib3_context(ciphers="DEFAULT:@SECLEVEL=1")
        except Exception:
            context = None
    if context is None:
        try:
            context = ssl.create_default_context()
            context.set_ciphers("DEFAULT:@SECLEVEL=1")
        except Exception:
            context = None
    if context is None:
        session.verify = False
    else:
        try:
            adapter = HTTPAdapter(ssl_context=context)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        except TypeError:
            session.verify = False
    session._virgo_relaxed_tls = True
    return True
