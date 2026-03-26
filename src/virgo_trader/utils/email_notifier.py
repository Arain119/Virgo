"""Email notification utilities for long-running jobs.

SMTP configuration is provided via environment variables; sending is best-effort
and used for training/backtest completion notifications.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.header import Header
from email.mime.text import MIMEText

# Email settings are intentionally sourced from environment variables.
# Never hardcode credentials into source code (especially important for code deposits).
SMTP_SERVER = os.environ.get("VIRGO_SMTP_SERVER", "").strip()
SMTP_PORT = int(os.environ.get("VIRGO_SMTP_PORT", "25"))
SENDER_EMAIL = os.environ.get("VIRGO_SENDER_EMAIL", "").strip()
SENDER_PASSWORD = os.environ.get("VIRGO_SENDER_PASSWORD", "")


def send_notification(recipient_email: str | None, subject: str, message: str) -> bool:
    """Send a plain-text email notification.

    Environment variables:
    - `VIRGO_SMTP_SERVER`
    - `VIRGO_SMTP_PORT` (default=25)
    - `VIRGO_SENDER_EMAIL`
    - `VIRGO_SENDER_PASSWORD`

    Returns:
        True if email was sent successfully; otherwise False.
    """

    if not recipient_email:
        logging.warning("Recipient email not configured; skipping email notification.")
        return False

    if not SMTP_SERVER or not SENDER_EMAIL or not SENDER_PASSWORD:
        logging.warning(
            "SMTP not configured (missing VIRGO_SMTP_SERVER / VIRGO_SENDER_EMAIL / VIRGO_SENDER_PASSWORD); "
            "skipping email notification."
        )
        return False

    msg = MIMEText(message, "plain", "utf-8")
    msg["From"] = f"Virgo Trader <{SENDER_EMAIL}>"
    msg["To"] = recipient_email
    msg["Subject"] = Header(subject, "utf-8")

    try:
        logging.info("Connecting to SMTP server %s:%s...", SMTP_SERVER, SMTP_PORT)
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        # server.set_debuglevel(1)  # Enable for detailed debug output.

        logging.info("Logging in to SMTP server...")
        server.login(SENDER_EMAIL, SENDER_PASSWORD)

        logging.info("Sending email to %s...", recipient_email)
        server.sendmail(SENDER_EMAIL, [recipient_email], msg.as_string())
        server.quit()
        logging.info("Email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError as exc:
        logging.error("SMTP authentication failed: %s", exc)
        return False
    except Exception as exc:
        logging.error("Failed to send email: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    test_recipient = os.environ.get("VIRGO_TEST_RECIPIENT", "").strip()
    if not test_recipient:
        logging.error("VIRGO_TEST_RECIPIENT not set; refusing to send a test email.")
        raise SystemExit(2)
    logging.info("Sending a test email to %s...", test_recipient)

    success = send_notification(
        recipient_email=test_recipient,
        subject="Virgo Trader - Test Email",
        message="This is a test email from Virgo Trader.",
    )
    logging.info("Test email sent: %s", success)
