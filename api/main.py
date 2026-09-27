"""FastAPI service for churn prediction.

Loads the saved Phase 9 artifact once at startup (no training here) and
delegates validation and prediction to churn.predict.

Run locally:
    uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import CustomerFeatures, HealthResponse, ModelInfoResponse, PredictionResponse
from churn.predict import ChurnModel, InvalidCustomerError, load_model, predict_customer

logger = logging.getLogger(__name__)


def create_app(model_loader: Callable[[], ChurnModel] = load_model) -> FastAPI:
    """Build the app; `model_loader` is injectable so tests can supply a model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
            logger.info("Loaded churn model %s", app.state.model.version)
        except Exception:
            # Stay up so /health can report the problem; /predict returns 503.
            logger.exception("Could not load the churn model")
            app.state.model = None
        yield

    app = FastAPI(
        title="Customer Churn Prediction API",
        version="1.0.0",
        description="Scores one customer's churn probability using the saved preprocessing + model pipeline.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Return location/message/type only. Echoing the submitted input (FastAPI's default)
        # fails for non-JSON values such as NaN and reflects client data back unnecessarily.
        detail = [{"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]} for err in exc.errors()]
        return JSONResponse({"detail": detail}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    def get_model(request: Request) -> ChurnModel:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not loaded")
        return model

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request):
        model = request.app.state.model
        body = HealthResponse(
            status="ok" if model else "unavailable",
            model_loaded=model is not None,
            model_version=model.version if model else None,
        )
        code = status.HTTP_200_OK if model else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(body.model_dump(), status_code=code)

    @app.get("/model", response_model=ModelInfoResponse, tags=["model"])
    def model_info(request: Request):
        meta = get_model(request).metadata
        return ModelInfoResponse(
            model_version=meta["model_version"],
            model=meta["model"],
            created_at_utc=meta["created_at_utc"],
            threshold=meta["threshold"],
            threshold_rule=meta.get("threshold_rule"),
            risk_levels=meta["risk_levels"],
            features=meta["input_schema"]["features"],
            training_rows=meta["training_data"]["rows"],
            cv_metrics_at_threshold=meta.get("cv_metrics_at_threshold"),
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["model"])
    def predict(customer: CustomerFeatures, request: Request):
        model = get_model(request)
        try:
            result = predict_customer(model, customer.model_dump())
        except InvalidCustomerError as exc:
            # Passed field-level checks but failed schema rules (e.g. inconsistent services).
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, exc.errors) from exc
        return PredictionResponse(**result.to_dict())

    return app


app = create_app()
