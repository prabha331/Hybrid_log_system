import re

REGEX_PATTERNS = {
    "System Notification": re.compile(
        r"File data_.*\.csv uploaded successfully by user User\S+"
        r"|Backup completed successfully"
        r"|Backup started at .+"
        r"|System reboot initiated by user User\S+",
        re.IGNORECASE,
    ),
    "User Action": re.compile(
        r"Account with ID \S+ created by User\S+"
        r"|User User\S+ logged out",
        re.IGNORECASE,
    ),
    "HTTP Status": re.compile(
        r"(status:\s*\d{3}|RCODE\s+\d{3}|HTTP status code\s*[-–]\s*\d{3}|Return code:\s*\d{3})"
        r"|nova\.(osapi_compute|metadata\.wsgi\.server)",
        re.IGNORECASE,
    ),
}


def classify_with_regex(log_message: str) -> str | None:
    for label, pattern in REGEX_PATTERNS.items():
        if pattern.search(log_message):
            return label
    return None


if __name__ == "__main__":
    samples = {
        "System Notification": [
            "File data_2024-01-01.csv uploaded successfully by user User42",
            "Backup completed successfully",
            "Backup started at 2024-01-01 03:00:00",
        ],
        "User Action": [
            "Account with ID 9981 created by UserAdmin",
            "User User99 logged out",
            "Account with ID abc-123 created by UserJohn",
        ],
        "HTTP Status": [
            "nova.osapi_compute GET /servers 200 0.512",
            "Return code: 404 for request /api/data",
            "HTTP status code - 500 internal server error",
        ],
    }

    for expected_label, logs in samples.items():
        for log in logs:
            result = classify_with_regex(log)
            status = "PASS" if result == expected_label else "FAIL"
            print(f"[{status}] Expected={expected_label!r}, Got={result!r} | {log}")
