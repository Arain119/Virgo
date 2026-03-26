"""SSE announcements source adapter.

Fetches announcements from the Shanghai Stock Exchange site and extracts a
textual summary, including optional PDF parsing for richer content.
"""

from __future__ import annotations

import io
import logging
import math
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from PIL import Image

from ..models import NewsItem, Target
from ..utils import SHANGHAI_TZ, request_with_retry

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:  # pragma: no cover - optional dependency
    pdf_extract_text = None

try:
    import fitz  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    fitz = None

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    RapidOCR = None
from .base import NewsSource

logger = logging.getLogger(__name__)


class SSEAnnouncementSource(NewsSource):
    """Shanghai Stock Exchange official announcements (queryCompanyBulletinNew)."""

    BASE_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do"
    REFERER = "http://www.sse.com.cn/disclosure/listedinfo/announcement/"
    MIN_TEXT_CHARS = 200

    def __init__(self, page_size: int = 50) -> None:
        super().__init__("sse", "announcement")
        self.page_size = max(1, page_size)
        self.session.headers.update(
            {
                "Referer": self.REFERER,
                "Accept": "application/json",
            }
        )
        self._ocr_engine: Optional[RapidOCR] = None  # type: ignore[type-arg]

    def fetch(self, target: Target, start_time: datetime, end_time: datetime) -> List[NewsItem]:
        # SSE interface only covers Shanghai-listed securities
        if target.exchange != "SH":
            return []

        params = {
            "isPagination": "true",
            "pageHelp.pageSize": self.page_size,
            "pageHelp.cacheSize": 1,
            "TITLE": "",
            "BULLETIN_TYPE": "",
            "stockType": "",
        }

        records: List[NewsItem] = []
        page = 1
        reached_earliest = False

        while True:
            params.update(
                {
                    "pageHelp.pageNo": page,
                    "pageHelp.beginPage": page,
                    "pageHelp.endPage": page,
                    "START_DATE": start_time.astimezone(SHANGHAI_TZ).strftime("%Y-%m-%d"),
                    "END_DATE": end_time.astimezone(SHANGHAI_TZ).strftime("%Y-%m-%d"),
                    "SECURITY_CODE": target.code,
                }
            )
            try:
                response = request_with_retry(self.session, "GET", self.BASE_URL, params=params)
            except Exception as exc:  # pragma: no cover - network branch
                logger.warning(
                    "SSE fetch failed for %s page %s: %s", target.symbol_alpha, page, exc
                )
                break

            data = response.json()
            page_help = data.get("pageHelp") or {}
            rows = page_help.get("data") or []
            if not rows:
                break

            for group in rows:
                for item in group:
                    date_str = item.get("SSEDATE")
                    if not date_str:
                        continue
                    try:
                        local_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                            tzinfo=SHANGHAI_TZ
                        )
                    except ValueError:
                        continue
                    published = local_dt.astimezone(timezone.utc)
                    if published > end_time:
                        continue
                    if published < start_time:
                        reached_earliest = True
                        continue
                    headline = (item.get("TITLE") or "").strip()
                    if not headline:
                        continue
                    url = item.get("URL") or ""
                    if url and url.startswith("/"):
                        url = f"https://www.sse.com.cn{url}"

                    summary = self._extract_pdf_summary(url) or f"详见公告：{url}"

                    records.append(
                        NewsItem(
                            source=self.name,
                            news_type=self.news_type,
                            target=target,
                            headline=headline,
                            summary=summary,
                            url=url,
                            published_at=published,
                            metadata={
                                "bulletin_type": item.get("BULLETIN_TYPE_DESC"),
                                "bulletin_year": item.get("BULLETIN_YEAR"),
                                "org_bulletin_id": item.get("ORG_BULLETIN_ID"),
                            },
                        )
                    )

            total = page_help.get("total") or 0
            max_pages = math.ceil(total / self.page_size) if total else None
            if reached_earliest:
                break
            if max_pages is not None and page >= max_pages:
                break
            page += 1

        return records

    def _extract_pdf_summary(self, url: str | None) -> str | None:
        if not url:
            return None
        pdf_bytes = self._download_pdf(url)
        if not pdf_bytes:
            return None
        cleaned = ""
        if pdf_extract_text is not None:
            try:
                buffer = io.BytesIO(pdf_bytes)
                text = pdf_extract_text(buffer)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - pdf parsing issues
                text = None
            if text:
                cleaned = " ".join(text.split())
        ocr_text: str | None = None
        if cleaned:
            if len(cleaned) >= self.MIN_TEXT_CHARS:
                return cleaned[:4000]
            ocr_text = self._ocr_pdf(pdf_bytes)
            if ocr_text and len(ocr_text) > len(cleaned):
                return ocr_text[:4000]
            return cleaned[:4000]
        ocr_text = ocr_text or self._ocr_pdf(pdf_bytes)
        if ocr_text:
            return ocr_text[:4000]
        return None

    def _download_pdf(self, url: str) -> bytes | None:
        headers = {
            "Accept": "application/pdf",
            "Referer": self.REFERER,
        }
        try:
            resp = request_with_retry(self.session, "GET", url, headers=headers)
        except Exception:
            return None
        if resp.headers.get("Content-Type", "").lower().startswith("application/pdf"):
            return resp.content
        cookie_value = self._solve_js_challenge(resp.text)
        if not cookie_value:
            return None
        self.session.cookies.set("acw_sc__v2", cookie_value, domain=".sse.com.cn")
        try:
            second = request_with_retry(self.session, "GET", url, headers=headers)
        except Exception:
            return None
        if second.headers.get("Content-Type", "").lower().startswith("application/pdf"):
            return second.content
        return None

    @staticmethod
    def _solve_js_challenge(html: str) -> str | None:
        """
        SSE returns an obfuscated script requiring clients to set acw_sc__v2.
        Re-implement the obfuscated logic to derive the cookie.
        """
        import re

        match = re.search(r"var arg1='([0-9A-F]+)'", html)
        if not match:
            return None
        arg1 = match.group(1)
        pos_list = [
            0x0F,
            0x23,
            0x1D,
            0x18,
            0x21,
            0x10,
            0x01,
            0x26,
            0x0A,
            0x09,
            0x13,
            0x1F,
            0x28,
            0x1B,
            0x16,
            0x17,
            0x19,
            0x0D,
            0x06,
            0x0B,
            0x27,
            0x12,
            0x14,
            0x08,
            0x0E,
            0x15,
            0x20,
            0x1A,
            0x02,
            0x1E,
            0x07,
            0x04,
            0x11,
            0x05,
            0x03,
            0x1C,
            0x22,
            0x25,
            0x0C,
            0x24,
        ]
        out = [""] * len(pos_list)
        for i, ch in enumerate(arg1):
            for j, pos in enumerate(pos_list):
                if pos == i + 1:
                    out[j] = ch
                    break
        arg2 = "".join(out)
        mask = "3000176000856006061501533003690027800375"
        limit = min(len(arg2), len(mask))
        res = []
        for i in range(0, limit, 2):
            s_char = int(arg2[i : i + 2], 16)
            m_char = int(mask[i : i + 2], 16)
            xor_val = s_char ^ m_char
            res.append(f"{xor_val:02x}")
        return "".join(res)

    def _ocr_pdf(self, pdf_bytes: bytes) -> str | None:
        if fitz is None or RapidOCR is None:
            return None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[arg-type]
        except Exception:
            return None
        try:
            engine = self._ocr_engine or RapidOCR()
            self._ocr_engine = engine
        except Exception:
            doc.close()
            return None
        texts: list[str] = []
        max_pages = min(doc.page_count, 5)
        for page_index in range(max_pages):
            try:
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                np_img = np.array(img)
                result, _ = engine(np_img)
            except Exception:
                result = None
            if result:
                texts.extend(line[1] for line in result if len(line) > 1 and line[1])
        doc.close()
        combined = " ".join(t.strip() for t in texts if t.strip())
        return combined if combined else None
