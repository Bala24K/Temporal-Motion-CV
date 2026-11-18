# Stack decisions

Rule: **adopt a technology when it removes a measured problem; otherwise leave a seam.** Status is what was actually
run in the authoring sandbox, not what is merely written.

## In the product (verified)
| Tech | Where | Evidence |
| --- | --- | --- |
| FastAPI, WebSockets, SQLAlchemy, Postgres (compose) / SQLite | `backend/` | 190+ pytest tests; Postgres path **not** run here |
| Next.js 15 (static export), React 19, TypeScript, Tailwind, TanStack Query, Zustand, Framer Motion, React Hook Form + Zod, Recharts | `web/` | `tsc` clean, `next build` OK, full Playwright run (`scripts/e2e_web.py`) incl. real WebSocket, 0 console errors |
| shadcn-style components | `web/components/ui/` | hand-written with cva + tailwind-merge (no Radix) |
| Three.js / React Three Fiber | `web/components/skeleton-3d.tsx` | renders in headless Chromium (SwiftShader); shows *your* calibrated proportions vs average |
| PyTorch | `ml/torch_models.py`, `ml/train_encoder.py` | **ran here**: shapes, SupCon, overfit, ONNX export parity (7e-8), train script smoke |
| ONNX + ONNX Runtime | `motion_engine/learned.py` | export verified numerically against torch and numpy |
| Contrastive learning (SupCon), prototypes (few-shot), Mahalanobis OOD, conformal prediction | `ml/torch_models.py`, `motion_engine/uncertainty.py` | numpy parts fully tested; coverage and OOD measured (see README) |
| MediaPipe (browser Tasks Vision) | `web/lib/pose.ts` | **written, not exercised** (no camera/CDN in sandbox) |
| RTMPose / YOLO-Pose adapters | `motion_engine/pose.py` | conversion + benchmark harness tested; the library-backed adapters are **untested** (need the libs + footage) |
| Label Studio bridge, DVC pipeline, run tracking (MLflow-compatible) | `ml/data/labelstudio.py`, `dvc.yaml`, `ml/tracking.py` | converters round-trip tested against the documented format; Label Studio/DVC/MLflow servers **not run** |
| OIDC (RS256/ES256 JWKS) | `backend/oidc.py` | tested with a local IdP keypair incl. alg-confusion, wrong aud/iss, expired, unknown kid flood. **No browser SSO redirect flow** (PKCE) is built |
| OpenTelemetry, Prometheus, Grafana, alerts | `backend/telemetry.py`, `infrastructure/` | spans tested with in-memory exporter; a test proves every alert/dashboard metric is really emitted; collector/Grafana containers **not run** |
| Docker (multi-stage), GitHub Actions | `Dockerfile`, `.github/` | YAML parsed in tests; **image not built here** |
| OWASP basics | `backend/main.py`, `security.py` | CSP, headers, login throttling, PBKDF2, tokens never logged |

## Deliberately not adopted, with the trigger that should change that
| Tech | Why not now | Adopt when |
| --- | --- | --- |
| **WebRTC** | Inference runs in the browser and only landmarks go over a WebSocket; WebRTC would stream *video* we promise never to upload. | You need server-side inference on user video (then also revisit the privacy promise) or peer coaching/live sessions. |
| Redis / Celery / RabbitMQ / NATS | `backend/jobs.py` covers the single job type (video ingest) in-process, same interface. | >1 API replica, or jobs must survive restarts. The login limiter and WebSocket sessions are in-process too. |
| CTR-GCN, VideoMAE, PyG, continual learning | Architecture is not the bottleneck; labelled data is. The encoder + SupCon is the baseline to beat. | A real, subject-disjoint dataset exists and the baseline plateaus. |
| FiftyOne, CVAT | Dataset report + Label Studio cover first annotation round. | Failure-case browsing at thousands of clips. |
| MLflow/W&B server | `ml/tracking.py` logs locally and mirrors to MLflow when `MLFLOW_TRACKING_URI` is set. | Multiple people training. |
| Depth Anything, SMPL(-X), 3D lifting | Needs GPU + data to validate. Yaw compensation (`skeleton.normalize`) is the cheap, verified step. | Sagittal-plane errors from front-view camera dominate real-world failures. |
| RTMPose/YOLO as default | No benchmark on your footage yet. | `python -m ml.benchmark_pose` says so. |
| Terraform/AWS (ECS, RDS, S3, ElastiCache, CloudFront), Kubernetes, Helm, ArgoCD | Unvalidated IaC that provisions paid resources is a liability; compose covers a single host. | A target account/region and a load profile exist. |
| TensorRT, Triton, OpenVINO, Rust/C++ | Server frame processing measured 0.54 ms p50 / 0.76 ms p95 over 744 frames (localhost, synthetic input); client pose latency is the real unknown and is now reported by browsers (`client_pose_ms`). | A GPU deployment with measured latency problems. |
| Kafka/Flink/Spark/Iceberg/Delta | Postgres is not stressed. | Event volume that Postgres cannot take. |
| React Native / Expo | The PWA manifest ships; native adds camera APIs and store review. | Web camera quality/background execution proves insufficient on phones. |
| Passkeys/WebAuthn, Keycloak, Vault | Local auth is prototype-grade; OIDC login is supported for APIs. | Before real users: put an IdP in front and add the PKCE redirect flow in `web/`. |
