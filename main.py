# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Any, Dict
import os
import json
import joblib
import pickle
import pandas as pd

# ==== Opcional: SageMaker como fallback si no hay modelo local ====
_USE_SM = False
try:
    import boto3
    _USE_SM = True
except Exception:
    _USE_SM = False

# Para componer preprocesador + modelo si vienen separados
try:
    from sklearn.pipeline import make_pipeline
except Exception:
    make_pipeline = None  # si no está scikit-learn, igual seguimos

app = FastAPI(
    title="ML Models API",
    description="Serve ML models (Iris, Titanic, Penguins) with metadata and predictions",
    version="1.2.0",
)

# ====== CORS ======
ALLOWED_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Directorios ======
HERE = Path(__file__).resolve().parent
BASE_MODELS_DIR = (HERE / "models") if (HERE / "models").exists() else HERE

# en tu repo "penguins" vive en carpeta "pinguins"
FOLDER_BY_MODEL = {"penguins": "pinguins"}

AVAILABLE_MODELS = {
    "iris": {
        "model_path": BASE_MODELS_DIR / "iris" / "model.pkl",
        "metadata_path": BASE_MODELS_DIR / "iris" / "iris_metadata.json",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "sm_endpoint": os.getenv("SM_ENDPOINT_IRIS", None),
    },
    # Titaníc: modelo típico pide estas columnas
    "titanic": {
        "model_path": BASE_MODELS_DIR / "titanic" / "model.pkl",
        "metadata_path": BASE_MODELS_DIR / "titanic" / "titanic_metadata.json",
        "features": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
        "sm_endpoint": os.getenv("SM_ENDPOINT_TITANIC", None),
    },
    # Penguins: usa culmen_* (sin species en la entrada)
    "penguins": {
        "model_path": BASE_MODELS_DIR / "pinguins" / "model.pkl",
        "metadata_path": BASE_MODELS_DIR / "pinguins" / "penguins_metadata.json",
        "features": ["island", "sex", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"],
        "sm_endpoint": os.getenv("SM_ENDPOINT_PENGUINS", None),
    },
}

LOADED_MODELS: Dict[str, Any] = {}

# ====== Schemas ======
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class TitanicRequest(BaseModel):
    age: float
    pclass: int                  # 1, 2, 3
    sex: Optional[str] = None    # "male" | "female"
    who: Optional[str] = None    # "man" | "woman" | "child" (opcional, se mapea a sex si llega)
    sibsp: int
    parch: int
    fare: float
    embarked: str                # "C" | "Q" | "S"

class PenguinsRequest(BaseModel):
    island: str
    sex: str
    # acepta ambos nombres; al menos uno de cada par debe venir
    culmen_length_mm: Optional[float] = None
    culmen_depth_mm: Optional[float] = None
    bill_length_mm: Optional[float] = None
    bill_depth_mm: Optional[float] = None
    flipper_length_mm: float
    body_mass_g: float

# ====== Utils ======
def _load_metadata(model_name: str) -> dict:
    info = AVAILABLE_MODELS[model_name]
    path = info["metadata_path"]
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"model_name": model_name, "features": info["features"], "class_names": None, "source": "fallback"}

def _env_model_path(model_name: str) -> Optional[Path]:
    env_key = f"{model_name.upper()}_MODEL_PATH"
    p = os.getenv(env_key)
    if p:
        pp = Path(p)
        if pp.exists():
            return pp
    return None

def _resolve_model_path(model_name: str) -> Optional[Path]:
    # 1) override por env
    envp = _env_model_path(model_name)
    if envp is not None:
        return envp

    info = AVAILABLE_MODELS[model_name]
    default_path = info["model_path"]
    folder_name = FOLDER_BY_MODEL.get(model_name, model_name)

    dirs_to_search = []
    if (BASE_MODELS_DIR / folder_name).exists():
        dirs_to_search.append(BASE_MODELS_DIR / folder_name)
    if (HERE / folder_name).exists():
        dirs_to_search.append(HERE / folder_name)
    if default_path.parent.exists():
        dirs_to_search.append(default_path.parent)

    typical_names = [
        f"{model_name}_pipeline.joblib",
        f"{model_name}_model.joblib",
        f"{model_name}.joblib",
        f"{model_name}.pkl",
        f"{model_name}_model.pkl",
        f"{model_name}_pipeline.pkl",
        "model.joblib",
        "model.pkl",
        "pipeline.joblib",
        "pipeline.pkl",
    ]

    candidates = [default_path]
    for d in dirs_to_search:
        for n in typical_names:
            candidates.append(d / n)
        for pattern in ["*.joblib", "*.pkl", "*.pickle", "*.sav"]:
            candidates.extend(d.glob(pattern))

    seen, uniq = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    for c in uniq:
        if c.exists():
            return c
    return None

def _coerce_predictor(obj: Any) -> Any:
    """
    Acepta:
      - Estimator/Pipeline con .predict (lo devuelve tal cual)
      - dict con claves típicas: 'pipeline', 'model', 'estimator', 'clf', 'classifier',
        'regressor', 'best_estimator_', 'sk_model', 'predictor', 'wrapped_model'
        y opcionalmente 'preprocessor'
      - Si hay 'preprocessor' + 'model' y existe sklearn, construye make_pipeline(preproc, model)
    """
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        if "pipeline" in obj and hasattr(obj["pipeline"], "predict"):
            return obj["pipeline"]
        for key in ("model", "estimator", "clf", "classifier", "regressor",
                    "best_estimator_", "sk_model", "predictor", "wrapped_model"):
            if key in obj and hasattr(obj[key], "predict"):
                est = obj[key]
                if "preprocessor" in obj and hasattr(obj["preprocessor"], "transform") and make_pipeline is not None:
                    try:
                        return make_pipeline(obj["preprocessor"], est)
                    except Exception:
                        return est
                return est
    return obj  # fallará más adelante con error claro

def _ensure_local_model(model_name: str):
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    resolved = _resolve_model_path(model_name)
    if resolved is None:
        return None

    loaded = None
    try:
        loaded = joblib.load(str(resolved))
    except Exception:
        try:
            with open(resolved, "rb") as f:
                loaded = pickle.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo '{model_name}' desde {resolved}: {e}")

    predictor = _coerce_predictor(loaded)
    if not hasattr(predictor, "predict"):
        keys = list(loaded.keys()) if isinstance(loaded, dict) else None
        raise HTTPException(
            status_code=500,
            detail=(
                f"El archivo cargado para '{model_name}' no contiene un estimador con .predict. "
                f"Tipo cargado: {type(loaded).__name__}" + (f", claves: {keys}" if keys else "")
            ),
        )

    LOADED_MODELS[model_name] = predictor
    return predictor

def _backend_for(name: str) -> Optional[str]:
    if _resolve_model_path(name) is not None:
        return "local"
    if AVAILABLE_MODELS[name]["sm_endpoint"] and os.getenv("AWS_REGION"):
        return "sagemaker"
    return None

def _get_backend(model_name: str) -> str:
    backend = _backend_for(model_name)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"No hay backend disponible para '{model_name}'. "
                   f"Sube un archivo .joblib/.pkl en la carpeta del modelo o define SM_ENDPOINT_* y AWS_REGION."
        )
    return backend

def _predict_local(model_name: str, X: pd.DataFrame):
    model = _ensure_local_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Modelo local '{model_name}' no encontrado.")
    try:
        pred = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
            except Exception:
                proba = None
        pred = pred.tolist() if hasattr(pred, "tolist") else list(pred)
        if proba is not None:
            proba = proba.tolist()
        return pred, proba
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fallo en predicción '{model_name}': {e}")

def _invoke_sagemaker(model_name: str, payload: dict):
    info = AVAILABLE_MODELS[model_name]
    endpoint = info["sm_endpoint"]
    region = os.getenv("AWS_REGION", "us-east-1")
    if not endpoint:
        raise HTTPException(status_code=500, detail=f"SM endpoint no configurado para '{model_name}'")
    try:
        sm = boto3.client("sagemaker-runtime", region_name=region)
        body = json.dumps(payload).encode("utf-8")
        resp = sm.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=body,
        )
        result = resp["Body"].read().decode("utf-8")
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo invocando SageMaker para '{model_name}': {e}")

# ====== Helpers específicos ======
def _penguins_df(req: PenguinsRequest, features: list[str]) -> pd.DataFrame:
    # aceptar sinónimos culmen_* o bill_*
    length = req.culmen_length_mm if req.culmen_length_mm is not None else req.bill_length_mm
    depth  = req.culmen_depth_mm  if req.culmen_depth_mm  is not None else req.bill_depth_mm
    if length is None or depth is None:
        raise HTTPException(
            status_code=422,
            detail="Faltan medidas: provee culmen_length_mm/culmen_depth_mm o sus equivalentes bill_length_mm/bill_depth_mm."
        )
    row = {
        "island": req.island,
        "sex": req.sex,
        "culmen_length_mm": float(length),
        "culmen_depth_mm": float(depth),
        "flipper_length_mm": float(req.flipper_length_mm),
        "body_mass_g": float(req.body_mass_g),
    }
    return pd.DataFrame([row], columns=features)

# ====== Endpoints ======
@app.get("/")
def root():
    return {"message": "Bienvenidos a ML Models API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return {
        "available_models": list(AVAILABLE_MODELS.keys()),
        "backends": {name: _backend_for(name) for name in AVAILABLE_MODELS.keys()},
    }

@app.get("/metadata/{model_name}")
def get_metadata(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    return _load_metadata(model_name)

@app.get("/models/{model_name}")
def get_model_description(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    meta = _load_metadata(model_name)
    resolved = _resolve_model_path(model_name)
    backend = _backend_for(model_name)
    info = AVAILABLE_MODELS[model_name]
    return {
        "model_name": model_name,
        "description": meta.get("description"),
        "version": meta.get("version"),
        "metrics": meta.get("metrics"),
        "features": info["features"],
        "class_names": meta.get("class_names"),
        "backend": backend,
        "available": backend is not None,
        "paths": {
            "resolved_model_path": str(resolved) if resolved else None,
            "default_model_path": str(info["model_path"]),
            "metadata_path": str(info["metadata_path"]),
        },
    }

# ---------- IRIS ----------
@app.post("/predict/iris")
def predict_iris(req: IrisRequest):
    model_name = "iris"
    backend = _get_backend(model_name)
    features = AVAILABLE_MODELS[model_name]["features"]
    X = pd.DataFrame([{
        "sepal_length": req.sepal_length,
        "sepal_width": req.sepal_width,
        "petal_length": req.petal_length,
        "petal_width": req.petal_width,
    }], columns=features)

    if backend == "local":
        pred, proba = _predict_local(model_name, X)
        # Map opcional a nombres si el pipeline NO lo hace:
        try:
            from sklearn.datasets import load_iris
            iris = load_iris()
            pred_label = iris.target_names[int(pred[0])]
        except Exception:
            pred_label = pred[0]
        return {"model_name": model_name, "backend": backend, "prediction": pred_label, "probabilities": proba}
    else:
        payload = {"instances": X.to_dict(orient="records")}
        result = _invoke_sagemaker(model_name, payload)
        return {"model_name": model_name, "backend": backend,
                "prediction": result.get("prediction"), "probabilities": result.get("probabilities")}

# ---------- TITANIC ----------
@app.post("/predict/titanic")
def predict_titanic(req: TitanicRequest):
    model_name = "titanic"
    backend = _get_backend(model_name)

    # Derivar 'sex' si no viene, a partir de 'who'
    sex_val = req.sex
    if sex_val is None and req.who is not None:
        if req.who.lower() == "man":
            sex_val = "male"
        elif req.who.lower() == "woman":
            sex_val = "female"
        else:
            sex_val = "male"  # fallback simple para 'child' u otros

    if sex_val is None:
        raise HTTPException(status_code=422, detail="Falta 'sex' o un 'who' mapeable ('man'/'woman').")

    features = AVAILABLE_MODELS[model_name]["features"]
    X = pd.DataFrame([{
        "pclass": req.pclass,
        "sex": sex_val,
        "age": req.age,
        "sibsp": req.sibsp,
        "parch": req.parch,
        "fare": req.fare,
        "embarked": req.embarked,
    }], columns=features)

    if backend == "local":
        pred, proba = _predict_local(model_name, X)
        try:
            pred_value = int(pred[0])
        except Exception:
            pred_value = pred[0]
        return {"model_name": model_name, "backend": backend, "prediction": pred_value, "probabilities": proba}
    else:
        payload = {"instances": X.to_dict(orient="records")}
        result = _invoke_sagemaker(model_name, payload)
        return {"model_name": model_name, "backend": backend,
                "prediction": result.get("prediction"), "probabilities": result.get("probabilities")}

# ---------- PENGUINS ----------
@app.post("/predict/penguins")
def predict_penguins(req: PenguinsRequest):
    model_name = "penguins"
    backend = _get_backend(model_name)
    features = AVAILABLE_MODELS[model_name]["features"]
    X = _penguins_df(req, features)

    if backend == "local":
        pred, proba = _predict_local(model_name, X)
        return {"model_name": model_name, "backend": backend, "prediction": pred[0], "probabilities": proba}
    else:
        payload = {"instances": X.to_dict(orient="records")}
        result = _invoke_sagemaker(model_name, payload)
        return {"model_name": model_name, "backend": backend,
                "prediction": result.get("prediction"), "probabilities": result.get("probabilities")}
