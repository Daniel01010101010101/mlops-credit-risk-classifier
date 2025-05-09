
# MLOps - Credit Risk Classifier

Este proyecto implementa un pipeline de Machine Learning completamente automatizado mediante GitHub Actions, utilizando MLflow para el seguimiento de experimentos.

## Estructura

- `src/train.py`: Entrenamiento del modelo
- `config.yaml`: Parámetros del modelo y rutas
- `Makefile`: Comandos automáticos
- `.github/workflows/ml.yml`: Workflow CI/CD
- `requirements.txt`: Dependencias
- `data/loan.csv`: Dataset
- `tests/test_dummy.py`: Prueba básica

## Uso

```bash
make install
make train
make test
```

## MLflow

Los experimentos son registrados localmente en `mlruns/`.

## <!-- Forzando ejecución del workflow -->
git add README.md
git commit -m "Forzar ejecución del pipeline"
git push



