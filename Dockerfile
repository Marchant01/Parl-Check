FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/
WORKDIR /backend
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY . .
RUN uv sync --frozen --no-editable

FROM python:3.13-slim AS backend
WORKDIR /backend
COPY --from=builder /backend/.venv /backend/.venv
ENV PATH="/backend/.venv/bin:$PATH"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.13-slim AS test
WORKDIR /backend
COPY --from=builder /backend/.venv /backend/.venv
COPY --from=builder /backend /backend
ENV PATH="/backend/.venv/bin:$PATH"
CMD ["python", "-m", "pytest", "tests/", "-q"]