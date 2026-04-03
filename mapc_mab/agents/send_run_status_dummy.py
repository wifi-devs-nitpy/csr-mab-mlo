import sys
import smtplib
from email.message import EmailMessage
from pathlib import Path

# ---------- CONFIG ----------
SENDER_EMAIL = "youremail@gmail.com"
APP_PASSWORD = "tqlnqdresvlmjbnn"
RECEIVER_EMAIL = "recieveremail@nitpy.ac.in"


def send_email(log_path: str, subject: str, body: str):
    file_path = Path(log_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{log_path} not found")

    msg = EmailMessage()
    msg["Subject"] = str(subject)
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(str(body))

    content = file_path.read_text(encoding="utf-8", errors="replace")
    msg.add_attachment(
        content,
        subtype="plain",
        filename=file_path.name,
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)

    print(f"Email sent successfully with attachment: {file_path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python send_logs.py <log_file>")
        sys.exit(1)

    send_email(sys.argv[1])