FROM python:3.14-slim AS prod

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY mail_pipeline/ /build/mail_pipeline/
COPY pyproject.toml requirements.txt /build/
RUN pip install --no-cache-dir -r /build/requirements.txt /build

RUN mkdir -p /state /config /secrets

EXPOSE 8080
CMD ["python", "-m", "mail_pipeline"]

FROM prod AS dev
COPY requirements-dev.txt /tmp/reqs-dev.txt
RUN pip install --no-cache-dir -r /tmp/reqs-dev.txt
COPY tests/ /app/tests/
WORKDIR /app
ENTRYPOINT []
CMD ["pytest", "tests/"]
