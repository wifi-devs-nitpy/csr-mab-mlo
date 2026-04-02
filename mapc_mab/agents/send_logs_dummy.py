import sys
import smtplib
from email.message import EmailMessage
from pathlib import Path

# ---------- CONFIG ----------
SENDER_EMAIL = "youremail"
APP_PASSWORD = "app password"
RECEIVER_EMAIL = "reciveremail"


def send_email(zip_path: str):
    file_path = Path(zip_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")

    msg = EmailMessage()
    msg["Subject"] = "Script Execution Completed"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content("Your script has finished running.\nLogs are attached.")

    with open(file_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="zip",
            filename=file_path.name,
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)

    print(f"Email sent successfully with attachment: {file_path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python send_logs.py <zip_file>")
        sys.exit(1)

    send_email(sys.argv[1])