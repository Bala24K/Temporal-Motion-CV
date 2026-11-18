"""OpenTelemetry tracing (own TracerProvider so tests can attach an in-memory exporter) + optional Sentry.
Export: set OTEL_EXPORTER_OTLP_ENDPOINT (needs opentelemetry-exporter-otlp-proto-http). Spans are per request,
per recognition and per analysed clip - NOT per frame (30 spans/s/user would drown the collector); per-frame
cost is covered by the latency metrics."""
from __future__ import annotations

import logging
import os

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

log = logging.getLogger("fitness.telemetry")
provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "kinetic-api")}))
tracer = provider.get_tracer("kinetic")
_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    ep = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if ep:
        try:                                                # pragma: no cover - needs the otlp exporter package
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=ep.rstrip("/") + "/v1/traces")))
        except Exception as e:                              # pragma: no cover
            log.warning("OTLP exporter unavailable: %s", e)
    dsn = os.getenv("SENTRY_DSN")
    if dsn:
        try:                                                # pragma: no cover - needs sentry_sdk
            import sentry_sdk
            sentry_sdk.init(dsn=dsn, traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.05")), send_default_pii=False)
        except Exception as e:                              # pragma: no cover
            log.warning("Sentry unavailable: %s", e)
