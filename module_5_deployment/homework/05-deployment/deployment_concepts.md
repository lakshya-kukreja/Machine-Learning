# ML Model Deployment --- Concepts, Workflow & Troubleshooting

> **Project:** Customer Churn Prediction API\
> **Learning path:** DataTalksClub Machine Learning Zoomcamp --- Module
> 5: Deployment\
> **Purpose:** Document the deployment workflow, the technologies used,
> the issues encountered, and how to run the project after cloning the
> repository.

------------------------------------------------------------------------

## Table of Contents

-   [1. Project overview](#1-project-overview)
-   [2. Architecture and request flow](#2-architecture-and-request-flow)
-   [3. End-to-end workflow](#3-end-to-end-workflow)
-   [4. Technologies and why we chose
    them](#4-technologies-and-why-we-chose-them)
-   [5. API endpoints](#5-api-endpoints)
-   [6. Logging, error handling, and basic
    monitoring](#6-logging-error-handling-and-basic-monitoring)
-   [7. CI/CD workflow](#7-cicd-workflow)
-   [8. Problems encountered and
    solutions](#8-problems-encountered-and-solutions)
-   [9. Run the project after cloning](#9-run-the-project-after-cloning)
-   [10. Testing checklist](#10-testing-checklist)
-   [11. Important limitations and next
    steps](#11-important-limitations-and-next-steps)
-   [12. Key concepts to remember](#12-key-concepts-to-remember)
-   [References](#references)

------------------------------------------------------------------------

## 1. Project overview

This project takes a trained machine-learning pipeline and exposes it
through an HTTP API so another application can request a prediction.

The model predicts the probability that a customer will churn. The API
accepts customer attributes as JSON and returns:

-   `churn_probability`: predicted probability of churn.
-   `churn`: Boolean decision based on a probability threshold of `0.5`.

The trained pipeline is stored in `pipeline.bin`. The API loads this
artifact and uses it for incoming prediction requests.

### Why deploy a model?

A notebook is useful for experimentation, but it is not a convenient
interface for another application. Deployment wraps the trained model in
a service that can be called over HTTP, tested automatically, packaged
in a container, and hosted online.

**Important:** The API makes predictions using the saved pipeline. It
does not retrain the model every time a request arrives.

------------------------------------------------------------------------

## 2. Architecture and request flow

### High-level architecture

``` mermaid
flowchart TD
    A[Client / Swagger UI] -->|HTTP JSON request| B[FastAPI application]
    B --> C[Pydantic input validation]
    C --> D[Loaded ML pipeline]
    D --> E[Prediction probability]
    E --> F[JSON response]
    B --> G[Application logs and basic metrics]
    H[GitHub repository] --> I[GitHub Actions CI]
    I -->|checks pass| J[Render deployment]
    J --> B
```

### What happens during a prediction?

1.  A client sends customer data to `POST /predict`.
2.  FastAPI receives the HTTP request.
3.  Pydantic validates the request against the input schema.
4.  The application passes the validated data to the saved pipeline.
5.  The pipeline returns a churn probability.
6.  The application applies the `0.5` threshold to produce a Boolean
    prediction.
7.  FastAPI returns the result as JSON.
8.  Logging records useful operational information; basic counters track
    successes, errors, and latency.

If the request data fails schema validation, FastAPI normally returns
**HTTP 422** before the prediction function executes. If an unexpected
error occurs during prediction, the application logs the exception and
returns a generic **HTTP 500** response.

### Deployment lifecycle

``` mermaid
flowchart LR
    A[Edit code] --> B[Run tests locally]
    B --> C[Commit and push]
    C --> D[GitHub Actions]
    D -->|tests pass| E[Render auto-deploy]
    D -->|tests fail| F[Fix the issue]
    F --> B
    E --> G[Check deployed API]
```

This is the core CI/CD loop: change, test, push, verify, and deploy only
after the configured checks pass.

------------------------------------------------------------------------

## 3. End-to-end workflow

### Step 1 --- Start with a trained model

The training workflow produces a serialized scikit-learn pipeline,
`pipeline.bin`. The pipeline contains the preprocessing and model steps
required to make predictions from customer features.

Saving the entire pipeline is important: the serving application must
apply the same preprocessing used during training. Reimplementing
preprocessing separately in the API could cause training-serving
inconsistencies.

### Step 2 --- Load the artifact

The application locates `pipeline.bin` relative to the Python file using
`pathlib.Path`. This is more reliable than depending on whichever
directory happens to be the current working directory.

The FastAPI application's lifespan startup function loads the pipeline
once when the application starts. Requests then reuse the loaded model
instead of opening and unpickling the file on every request.

> **Security note:** Only load pickle files from sources you trust.
> Unpickling an untrusted artifact can execute arbitrary code.

### Step 3 --- Create a prediction API

FastAPI defines HTTP endpoints. Pydantic models describe and validate
the input and output data structures.

The main prediction route is `POST /predict`. It accepts a JSON body,
converts the validated input to a dictionary, calls `predict_proba`, and
returns the probability and threshold-based decision.

### Step 4 --- Validate behavior with tests

Pytest and FastAPI's test client are used to verify expected behavior.
The test suite covers:

-   `GET /health` returns a successful health response.
-   A valid prediction returns a probability between `0` and `1` and a
    Boolean churn decision.
-   Invalid tenure (for example, `-1`) is rejected with HTTP 422.
-   `GET /predict` is rejected with HTTP 405 because the endpoint
    expects POST.

Automated tests help catch regressions before deployment.

### Step 5 --- Package with Docker

Docker packages the application and its runtime dependencies into an
image. A container created from that image runs the API in a consistent
environment.

The Dockerfile specifies the Python base image, dependency installation,
application files, exposed port, and server command. The image is built
locally and can then be run as a container.

### Step 6 --- Deploy to Render

Render hosts the API so it can be accessed without running the
application on a personal computer. The deployed service exposes the API
over the internet.

The project was configured for automatic deployment after CI checks
pass. After pushing a change, verify both the GitHub Actions run and the
Render deployment status.

**Live service used during this project:**\
https://churn-prediction-ml-model-bkdm.onrender.com

Hosted services on free tiers may sleep when inactive or take time to
wake up. A delayed first response does not necessarily mean the model is
broken.

### Step 7 --- Add operational basics

After the core deployment worked, the project was extended with:

-   **Logging:** records startup and prediction events, and captures
    exception details.
-   **Error handling:** avoids returning internal exception details to
    API clients.
-   **Basic monitoring:** tracks successful predictions, prediction
    errors, and cumulative request latency in memory.

These are practical additions beyond the minimum of simply serving a
model.

------------------------------------------------------------------------

## 4. Technologies and why we chose them

  -------------------------------------------------------------------------
  Technology                Role in this project    Why this choice
  ------------------------- ----------------------- -----------------------
  **Python**                Application and ML code The model and its
                                                    supporting ML ecosystem
                                                    are already
                                                    Python-based.

  **scikit-learn**          Saved ML pipeline and   Provides
                            prediction              preprocessing/model
                                                    pipelines and
                                                    `predict_proba` for
                                                    probability estimates.

  **Pickle**                Serialize and reload    Convenient for saving a
                            the pipeline            Python model pipeline
                                                    for this learning
                                                    project. Only load
                                                    trusted artifacts.

  **FastAPI**               HTTP API                Provides typed request
                                                    handling, validation
                                                    integration, and
                                                    automatic OpenAPI
                                                    documentation.

  **Pydantic**              Request/response        Defines expected fields
                            validation              and constraints and
                                                    returns useful
                                                    validation errors.

  **Uvicorn**               ASGI server             Runs the FastAPI
                                                    application.

  **pytest**                Automated tests         Makes endpoint behavior
                                                    repeatable and helps
                                                    catch regressions.

  **httpx / FastAPI test    Exercise API routes in  Allows tests to call
  client**                  tests                   the application without
                                                    manually starting a
                                                    public server.

  **uv**                    Python dependencies and Manages project
                            environment             dependencies and can
                                                    synchronize them from
                                                    the lockfile
                                                    reproducibly.

  **Docker**                Container image         Packages the runtime
                                                    and application into a
                                                    portable unit.

  **Git and GitHub**        Version control and     Track changes and
                            source hosting          trigger automation from
                                                    repository events.

  **GitHub Actions**        Continuous integration  Automatically installs
                            (CI)                    dependencies and runs
                                                    tests on pushes and
                                                    pull requests.

  **Render**                Hosting and deployment  Provides a
                                                    straightforward way to
                                                    deploy the API and
                                                    connect deployment to
                                                    repository changes.

  **Python `logging`**      Application logs        Supports severity
                                                    levels and exception
                                                    tracebacks instead of
                                                    relying only on
                                                    `print`.

  **Python                  Request timing          Measures elapsed time
  `time.perf_counter()`**                           for latency tracking.
  -------------------------------------------------------------------------

### Why FastAPI instead of alternatives?

-   **FastAPI vs Flask:** Both can serve APIs. FastAPI integrates
    type-driven validation and OpenAPI documentation directly, which is
    useful for an ML prediction service.
-   **FastAPI vs Django:** Django is a larger web framework with many
    built-in features. For a focused prediction API, FastAPI is
    lightweight and direct.
-   **Uvicorn:** FastAPI is an application framework; Uvicorn is the
    server that runs the ASGI application. They have different roles.

### Why Docker instead of only running Python locally?

A local environment can depend on installed packages, Python versions,
and machine configuration. Docker packages the runtime environment and
application together, making the deployment process more repeatable.

Docker does not replace tests or CI. A container can still contain a
bug; tests help detect that bug.

### Why GitHub Actions and Render?

GitHub Actions performs checks on code changes. Render runs the deployed
service. Keeping these roles separate creates a simple pipeline:

-   **CI:** Is this change passing the automated checks?
-   **CD/deployment:** Can the verified version be released to the
    hosting service?

------------------------------------------------------------------------

## 5. API endpoints

  ------------------------------------------------------------------------
  Method            Endpoint          Purpose           Expected behavior
  ----------------- ----------------- ----------------- ------------------
  `GET`             `/health`         Basic health      Returns `200 OK`
                                      check             with a status
                                                        response.

  `POST`            `/predict`        Predict customer  Returns churn
                                      churn             probability and
                                                        Boolean decision
                                                        for valid input.

  `GET`             `/docs`           Interactive API   Opens Swagger UI.
                                      documentation     

  `GET`             `/redoc`          Alternative API   Opens ReDoc.
                                      documentation     

  `GET`             `/openapi.json`   OpenAPI schema    Returns the
                                                        machine-readable
                                                        API definition.

  `GET`             `/metrics`        Basic in-memory   Returns prediction
                                      metrics           count, error
                                                        count, and
                                                        cumulative
                                                        latency.
  ------------------------------------------------------------------------

### Example prediction request

The exact fields must match the `Customer` schema in `app.py`. A
simplified example of the request structure is:

``` json
{
  "tenure": 15,
  "monthlycharges": 79.85,
  "totalcharges": 950.2
}
```

This is **not necessarily a complete valid request**: include every
required field from the project's actual Pydantic schema.

### Example response

``` json
{
  "churn_probability": 0.73,
  "churn": true
}
```

These values are illustrative, not a guaranteed output for a particular
customer.

### Understanding the HTTP status codes

  -----------------------------------------------------------------------
  Status                              Meaning in this API
  ----------------------------------- -----------------------------------
  `200 OK`                            Request completed successfully.

  `405 Method Not Allowed`            The route exists, but the HTTP
                                      method is unsupported.

  `422 Unprocessable Entity`          Request body does not satisfy the
                                      input schema.

  `500 Internal Server Error`         An unexpected prediction failure
                                      occurred.
  -----------------------------------------------------------------------

The exact response details depend on the code and validation rules.

------------------------------------------------------------------------

## 6. Logging, error handling, and basic monitoring

### Logging

Logging records events that help explain what the application is doing.
The project configures a logger with a timestamp, severity level, logger
name, and message.

Example format:

``` text
2026-10-28 06:45:02 | INFO | app | Model loaded successfully
```

Useful levels:

-   `DEBUG`: detailed diagnostic information.
-   `INFO`: normal application events.
-   `WARNING`: unexpected conditions that do not necessarily stop the
    application.
-   `ERROR`: an operation failed.
-   `CRITICAL`: a serious failure that may prevent continued operation.

Use `logger.exception(...)` inside an exception handler when a traceback
is useful. Avoid logging sensitive customer information unnecessarily.

### Error handling

The prediction route catches unexpected exceptions, logs the detailed
exception internally, and returns a generic message to the client.

This separates two concerns:

1.  **Developer/operator:** needs enough details in logs to debug the
    issue.
2.  **API client:** needs a stable response without internal paths,
    tracebacks, or sensitive implementation details.

Input validation errors should remain validation errors (for example,
HTTP 422). Do not convert every possible failure into the same status
code without considering its meaning.

### Basic monitoring

The in-memory `/metrics` endpoint tracks:

-   `prediction_count`: successful predictions.
-   `error_count`: prediction failures caught by the endpoint.
-   `total_latency_seconds`: cumulative elapsed time measured for
    prediction requests.

The latency is measured with `time.perf_counter()`, which is designed
for measuring elapsed time.

**Limitations of this basic approach:**

-   Metrics reset when the application process restarts.
-   Each worker/process maintains its own counters.
-   Counters are not durable storage.
-   The endpoint is not a full monitoring/alerting system.
-   The latency value is cumulative; average latency would require
    dividing by the number of measured requests.
-   A public metrics endpoint may expose operational information, so
    consider access controls before using it in production.

Prometheus and Grafana were discussed as possible future enhancements
but were **not part of the completed implementation documented here**.

------------------------------------------------------------------------

## 7. CI/CD workflow

### Continuous Integration (CI)

The GitHub Actions workflow runs automatically for configured repository
events, such as pushes and pull requests to the target branch.

The workflow used in this project:

1.  Checks out the repository.
2.  Sets up Python and `uv`.
3.  Synchronizes dependencies using the lockfile.
4.  Runs the pytest suite.
5.  Reports whether the checks passed or failed.

Conceptually:

``` text
Git push / Pull request
          |
          v
    GitHub Actions
          |
          v
 Install locked dependencies
          |
          v
      Run pytest
       /      \
   Failure   Success
      |         |
 Fix code    CI passes
                |
                v
       Render auto-deploy
                |
                v
        Verify live API
```

### Continuous Deployment (CD)

The Render service was configured to deploy after CI checks pass. This
reduces the risk of deploying a change that fails the configured test
suite.

CI passing does **not** guarantee that every production behavior is
correct. It means the configured checks passed. Always verify the
deployed service after a release.

### How to verify a deployment

1.  Open the repository's **Actions** tab.
2.  Confirm the workflow for the latest commit passed.
3.  Open the Render service dashboard and confirm deployment succeeded.
4.  Check `/health`.
5.  Open `/docs` and send a valid `POST /predict` request.
6.  Check `/metrics` if the in-memory monitoring endpoint is enabled.

------------------------------------------------------------------------

## 8. Problems encountered and solutions

The following summarizes the main issues encountered during development
and what they taught us.

### 8.1 Docker was unavailable in WSL

**Symptom:** Docker commands could not connect to the Docker engine.

**Cause:** Docker CLI availability inside WSL does not itself guarantee
that the Docker daemon is running or that WSL integration is enabled.

**Solution:** Start Docker Desktop and enable the appropriate WSL
integration, then verify the engine using:

``` bash
docker version
docker info
```

If the client works but the server information is unavailable,
troubleshoot the Docker engine/integration rather than changing
application code.

### 8.2 A local API URL returned connection refused

**Symptom:** Opening `localhost` or `/docs` failed.

**Cause:** A browser can connect only while a server is running and
listening on the expected port.

**Solution:** Start the server from the directory containing `app.py`:

``` bash
uv run uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

Keep the server terminal open while testing. Stop it with `Ctrl+C`.

### 8.3 Logging configuration caused a syntax error

**Symptom:** Python rejected the logging format.

**Cause:** The format string must be a quoted Python string. Text such
as `%(asctime)s` is not valid as a bare expression in the `format`
argument.

**Correct form:**

``` python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
```

### 8.4 Model file paths were fragile

**Potential failure:** The app cannot find `pipeline.bin` when started
from a different working directory.

**Solution:** Resolve the model path relative to the Python file:

``` python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pipeline.bin"
```

This is generally more robust than relying on a relative path such as
`"pipeline.bin"` alone.

### 8.5 Python environment and dependency synchronization

**Issue:** `uv` warned that an existing `VIRTUAL_ENV` was being ignored
because the active environment did not match the project environment.

**Lesson:** Run commands from the intended project directory and keep
the project's dependency files (`pyproject.toml` and `uv.lock`) aligned.
Avoid accidentally mixing multiple virtual environments.

Useful checks:

``` bash
pwd
which python
uv --version
```

Use the project's documented environment setup rather than creating
another environment unnecessarily.

### 8.6 Test failure versus application failure

A failed test does not always mean the deployed API is broken. It may
reveal a mismatch between the expected response and actual behavior, a
test setup problem, or an application bug.

**Approach:**

1.  Read the failing assertion and traceback.
2.  Reproduce the issue locally.
3.  Check the endpoint contract and status code.
4.  Fix the code or correct an incorrect test expectation.
5.  Rerun the full suite before pushing.

### 8.7 Deployment and CI are different stages

A successful local run does not guarantee that CI or deployment will
succeed. Differences in dependencies, paths, environment variables,
Docker configuration, and hosting settings can cause failures.

**Solution:** Inspect the failing stage's logs. Do not rebuild the
Docker image or change unrelated code unless the failure requires it.

------------------------------------------------------------------------

## 9. Run the project after cloning

> **Important:** Run commands from the directory that contains the
> relevant `pyproject.toml`, `uv.lock`, `app.py`, and Dockerfile. In
> this repository, the deployment work has been under
> `module_5_deployment`; some homework files may be nested under
> `homework/05-deployment`. Check the actual file layout after cloning.

### 9.1 Clone the repository

``` bash
git clone https://github.com/lakshya-kukreja/Machine-Learning.git
cd Machine-Learning
```

Inspect the deployment directory:

``` bash
find module_5_deployment -maxdepth 3 -type f
```

Change into the directory containing `app.py` and the relevant project
files. For example, if they are in the homework directory:

``` bash
cd module_5_deployment/homework/05-deployment
```

If the files are directly inside `module_5_deployment`, use that
directory instead.

### 9.2 Install prerequisites

You need:

-   Git
-   A compatible Python version for the project
-   `uv`
-   Docker Desktop/Engine only if you want to build and run the
    container

Install `uv` using the official instructions:
https://docs.astral.sh/uv/getting-started/installation/

Check installations:

``` bash
git --version
python --version
uv --version
docker version
```

Docker is optional for running the app directly with Python, but
required for the container workflow.

### 9.3 Install project dependencies

From the directory containing `pyproject.toml` and `uv.lock`:

``` bash
uv sync --locked
```

`--locked` asks `uv` to use the lockfile without updating it. If this
command reports that the lockfile is out of date, investigate the
dependency-file mismatch rather than removing the flag automatically.

### 9.4 Run tests

``` bash
uv run pytest -v
```

If the project uses a particular test file, for example:

``` bash
uv run pytest -v test_app.py
```

A passing suite should show all configured tests passing. The exact
number depends on the current repository state.

### 9.5 Start the API locally

``` bash
uv run uvicorn app:app --reload
```

Open these URLs in a browser:

-   Swagger UI: http://127.0.0.1:8000/docs
-   ReDoc: http://127.0.0.1:8000/redoc
-   Health check: http://127.0.0.1:8000/health
-   Basic metrics: http://127.0.0.1:8000/metrics

The `/metrics` endpoint is available only if the monitoring addition is
present in the checked-out version.

To stop the server, press `Ctrl+C` in the terminal.

### 9.6 Call the health endpoint from a terminal

With the server running:

``` bash
curl http://127.0.0.1:8000/health
```

Expected response shape:

``` json
{"status":"ok"}
```

### 9.7 Test predictions

The easiest way is to open Swagger UI at `http://127.0.0.1:8000/docs`,
expand `POST /predict`, select **Try it out**, enter a complete JSON
object matching the `Customer` schema, and click **Execute**.

Use the full input schema shown in Swagger. Do not rely on the
abbreviated example earlier in this document as a complete request.

### 9.8 Build and run Docker

From the directory containing the Dockerfile:

``` bash
docker build -t churn-prediction .
```

Run the container:

``` bash
docker run --rm -p 8000:8000 churn-prediction
```

Then visit `http://127.0.0.1:8000/docs`.

The mapping `8000:8000` means host port 8000 maps to container port
8000. If port 8000 is already occupied by a local Uvicorn process, stop
that process or use another host port, for example:

``` bash
docker run --rm -p 8001:8000 churn-prediction
```

In that case, open `http://127.0.0.1:8001/docs`.

To rebuild the image, rerun `docker build` after changing the
Dockerfile, application code, or dependencies that are copied into the
image. You do not need to rebuild just to run an image that already
reflects the current code.

### 9.9 Test the deployed API

Live service:

https://churn-prediction-ml-model-bkdm.onrender.com

Try:

``` bash
curl https://churn-prediction-ml-model-bkdm.onrender.com/health
```

Open the deployed docs:

https://churn-prediction-ml-model-bkdm.onrender.com/docs

A hosted service can change, sleep, or be unavailable. If the links stop
working, check the Render dashboard and current deployment logs.

------------------------------------------------------------------------

## 10. Testing checklist

Before pushing a change:

-   [ ] Dependencies synchronize successfully with `uv sync --locked`.
-   [ ] All automated tests pass.
-   [ ] `/health` returns HTTP 200.
-   [ ] Valid prediction input returns a probability from `0` to `1`.
-   [ ] The `churn` output is Boolean.
-   [ ] Invalid values such as negative tenure return a validation
    error.
-   [ ] Unsupported HTTP methods return the expected status code.
-   [ ] Logs show model startup and prediction events.
-   [ ] The `/metrics` endpoint returns metrics, if implemented.
-   [ ] Docker builds successfully if the container files changed.

After pushing:

-   [ ] GitHub Actions passes for the new commit.
-   [ ] Render deployment succeeds.
-   [ ] The deployed health endpoint works.
-   [ ] A valid prediction works through the deployed Swagger UI.

------------------------------------------------------------------------

## 11. Important limitations and next steps

This project is a learning deployment, not a fully hardened production
platform.

Possible future improvements:

1.  Add proper persistent metrics and dashboards with Prometheus and
    Grafana.
2.  Protect operational endpoints such as `/metrics` if they expose
    sensitive information.
3.  Add request IDs and structured logs.
4.  Add API authentication and rate limiting if the service requires
    them.
5.  Add integration tests against a staging deployment.
6.  Add monitoring alerts for elevated errors or latency.
7.  Consider a model artifact format and loading strategy appropriate
    for the deployment environment.
8.  Monitor model quality and data drift when real feedback/labels
    become available.
9.  Document required environment variables and secrets without
    committing secrets to Git.
10. Add health/readiness checks that distinguish "process is running"
    from "model is loaded and ready."

These are future enhancements, not claims about features already
implemented.

------------------------------------------------------------------------

## 12. Key concepts to remember

  -----------------------------------------------------------------------
  Concept                             Mental model
  ----------------------------------- -----------------------------------
  Model artifact                      The saved model pipeline the API
                                      loads to make predictions.

  API                                 A contract through which other
                                      programs communicate with the
                                      model.

  HTTP request/response               The client sends data; the server
                                      returns a result and status code.

  Pydantic validation                 Checks incoming data against the
                                      schema before prediction.

  Uvicorn                             The server process that runs the
                                      FastAPI application.

  Docker image                        A packaged template containing the
                                      app and its runtime environment.

  Docker container                    A running instance of an image.

  CI                                  Automatically test code changes.

  CD / auto-deploy                    Release a version after configured
                                      checks pass.

  Logging                             Record events and diagnostic
                                      details.

  Error handling                      Return controlled responses when
                                      something goes wrong.

  Monitoring                          Track metrics such as request
                                      counts, failures, and latency over
                                      time.

  Health check                        A simple endpoint used to check
                                      whether the service responds.

  Reproducibility                     Use version-controlled code and
                                      locked dependencies to make setup
                                      repeatable.
  -----------------------------------------------------------------------

### One-line summary

**Train and save the pipeline → serve it with FastAPI → validate and
test it → package it with Docker → run CI with GitHub Actions → deploy
on Render → verify the service and inspect logs/metrics.**

------------------------------------------------------------------------

## References

-   [This project's deployment
    directory](https://github.com/lakshya-kukreja/Machine-Learning/tree/025415732a41994636841726e438f992734ac6d1/module_5_deployment)
-   [DataTalksClub Machine Learning Zoomcamp --- Deployment
    workshop](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/9bb19da5777191f0dc0a038f93ef1b429db5a2bb/05-deployment/workshop)
-   [FastAPI documentation](https://fastapi.tiangolo.com/)
-   [Uvicorn documentation](https://www.uvicorn.org/)
-   [uv documentation](https://docs.astral.sh/uv/)
-   [Docker documentation](https://docs.docker.com/)
-   [GitHub Actions documentation](https://docs.github.com/en/actions)
-   [Render documentation](https://render.com/docs)
-   [pytest documentation](https://docs.pytest.org/)
