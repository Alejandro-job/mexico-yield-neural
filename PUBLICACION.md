# Comprobar en Windows y publicar el proyecto

## Qué indica la carpeta que abriste

Al descomprimir el ZIP se copian el código y un modelo previamente entrenado. El modelo guardado está en `hf_model/model.safetensors`; la arquitectura está en `hf_model/config.json`. La estructura de la red se programa en `src/network.py`.

Para confirmar que funciona en tu PC, ejecuta la verificación. Para entrenarlo tú, ejecuta el entrenamiento y conserva el registro de esa ejecución.

## 1. Comprobarlo desde VS Code

Abre la carpeta `mexico-neural-hf` y selecciona **Terminal → Nuevo terminal**. La ruta de la terminal debe terminar en `mexico-neural-hf`.

Estos comandos usan el Python del entorno directamente y funcionan sin activar scripts de PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/verify_model.py
Start-Process .\outputs\verification_report.html
```

Se necesita Python 3.11 o posterior. La última orden abre el informe en tu navegador. En VS Code también puedes abrir el archivo HTML y elegir abrirlo en un navegador.

Al finalizar correctamente, la terminal termina con `VERIFICACION COMPLETADA` y muestra:

- Integridad de los pesos y dimensiones de las capas.
- Los 1,066 parámetros entrenables.
- Reproducción de 18 pronósticos y recálculo de las métricas guardadas.
- Número de pruebas aprobadas y sistema donde se ejecutaron.
- Existencia de un registro de entrenamiento correspondiente a los pesos actuales.

La primera vez puede decir `Registro de entrenamiento asociado: NO`: el ZIP incluye pesos que ya estaban entrenados. Ese mensaje no invalida la inferencia.

El informe explica por separado la validez de los archivos y la precisión histórica. Pasar las pruebas no garantiza precisión futura.

## 2. Entrenarlo en tu PC

```powershell
.\.venv\Scripts\python.exe scripts/train_neural.py
.\.venv\Scripts\python.exe scripts/verify_model.py
Start-Process .\outputs\verification_report.html
```

Al finalizar el entrenamiento debe aparecer `TRAINING COMPLETED`. Se vuelven a escribir los pesos, la configuración y los resultados, y se crea `outputs/local_training_run.json`. Este registro contiene el identificador de ejecución, inicio y fin en UTC, sistema operativo, versión de Python y los hashes del CSV y de los pesos. La verificación comprueba que corresponde al modelo actual.

Con los mismos datos y semilla, los pesos pueden coincidir con los incluidos. El identificador y los tiempos del registro distinguen la nueva ejecución. Es un registro generado por el programa, no una certificación externa.

## 3. Qué archivo sirve para cada cosa

| Archivo | Qué demuestra |
| --- | --- |
| `src/network.py` | Cálculo de la red, derivadas y actualizaciones Adam |
| `hf_model/model.safetensors` | Pesos y valores de normalización que se usan para predecir |
| `hf_model/config.json` | Arquitectura, variables, hiperparámetros, mes de referencia y hashes |
| `outputs/local_training_run.json` | Registro de una ejecución de entrenamiento con estos pesos |
| `outputs/local_verification.json` | Resultados de la verificación ejecutada en tu entorno |
| `outputs/verification_report.html` | Lectura visual de las comprobaciones, métricas y pronósticos |
| `outputs/neural_backtest.csv` | Cada predicción histórica frente a su valor observado |
| `outputs/comparison.csv` | RMSE y MAE comparados con otros modelos |

## 4. GitHub: código, documentación y evaluación

Cuenta conectada identificada: `Alejandro-job`. Nombre propuesto: `mexico-yield-neural`.

Descripción lista para el repositorio:

> Pronóstico multisalida de tasas de México con una MLP en NumPy, validación temporal, comparación con persistencia y Ridge, pesos safetensors y exportación a Hugging Face.

Temas sugeridos: `python`, `numpy`, `neural-network`, `time-series`, `forecasting`, `finance`, `huggingface`, `backtesting`.

Instala [Git](https://git-scm.com/download/win) y [GitHub CLI](https://cli.github.com/) si todavía no están disponibles. Inicia sesión en tu propia terminal:

```powershell
gh auth login
gh auth status
git init -b main
git add .gitignore .gitattributes .env.example README.md PUBLICACION.md requirements.txt src scripts tests docs hf_model data outputs/comparison.csv outputs/neural_backtest.csv outputs/training_curve.json outputs/epoch_selection_curve.json
git commit -m "Add neural yield forecaster with temporal validation and diagrams"
gh repo create mexico-yield-neural --private --source=. --remote=origin --description "MLP en NumPy para pronosticar tasas de Mexico, con validacion temporal y pesos para Hugging Face" --push
```

Estos comandos crean un repositorio privado nuevo en la cuenta activa de GitHub CLI. Confirma esa cuenta en la salida de `gh auth status`; la sesión de tu terminal es independiente de la conexión de ChatGPT. Si Git solicita identidad para el commit, configura `git config user.name "TU_NOMBRE"` y `git config user.email "TU_CORREO_DE_COMMITS"` dentro de esta carpeta, con tus datos reales, y repite el commit.

Si ya existe el repositorio, utiliza su URL como remoto y ejecuta `git push -u origin main`. El comando de creación anterior está pensado para un repositorio nuevo. El código, los datos incluidos del ejercicio, los pesos y los resultados comparativos forman parte de la selección de archivos de GitHub. Los registros locales y el entorno virtual quedan excluidos.

Para mostrarlo públicamente en tu portafolio puedes cambiar la visibilidad desde los ajustes del repositorio después de revisar los datos y la documentación.

## 5. Hugging Face: pesos y ficha del modelo

Nombre propuesto: `TU_USUARIO_HF/mexico-yield-neural`. GitHub y Hugging Face pueden tener usuarios diferentes.

La descripción y los dos diagramas están preparados en `hf_model/README.md` y `hf_model/assets/`. El script envía exclusivamente los once archivos de modelo y documentación permitidos.

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe scripts/upload_hf.py --dry-run
.\.venv\Scripts\python.exe scripts/upload_hf.py --name mexico-yield-neural
```

El inicio de sesión necesita una cuenta de Hugging Face con acceso de escritura. Se crea un repositorio privado bajo la cuenta autenticada; el script imprime su URL y el commit. También vuelve a descargar los pesos de ese commit y comprueba el hash antes de informar de una subida correcta. Conserva `outputs/hf_upload_receipt.json` para identificar la versión.

Si ya existe un repositorio público con ese nombre, utiliza otro nombre con `--name`. El script está configurado para crear o actualizar repositorios privados.

### Comprobar el modelo subido

Reemplaza el usuario y el hash por los valores reales del recibo:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --repo TU_USUARIO_HF/mexico-yield-neural --revision HASH_COMPLETO_DEL_COMMIT
```

El repositorio del Hub conserva y versiona los archivos. La predicción se ejecuta con el código local incluido en este proyecto.

## 6. Información necesaria para completar una publicación

| Destino | Datos o acceso necesarios |
| --- | --- |
| GitHub | Cuenta activa, nombre o URL del repositorio y visibilidad deseada; acceso para crear o escribir |
| Hugging Face | Cuenta activa, nombre del repositorio y autenticación con permiso de escritura |
| Ambos | Archivos verificados y descripciones incluidas en este proyecto |

Las credenciales se introducen con `gh auth login` y `hf auth login` en tu terminal. Las claves de APIs de datos son independientes y este modelo no necesita esas claves para entrenarse con el CSV incluido.

El archivo preparado no significa que se haya publicado. La publicación se completa al ejecutar los comandos y obtener las URLs y commits de ambas plataformas.

Referencias oficiales: [GitHub CLI: crear repositorio](https://cli.github.com/manual/gh_repo_create), [Hugging Face: autenticación](https://huggingface.co/docs/huggingface_hub/quick-start#authentication), [Hugging Face: subir archivos](https://huggingface.co/docs/huggingface_hub/guides/upload).
