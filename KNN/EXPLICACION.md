# Algoritmo KNN (k-Nearest Neighbors)

El **KNN** (*k-Nearest Neighbors*, «k vecinos más cercanos») es un algoritmo de
**supervised learning** que sirve para **classification** y **regression**. Su
idea central:

> Para predecir la etiqueta de un punto nuevo, busca los `k` ejemplos de
> **train** más parecidos y se queda con lo que opine la mayoría.

Es **non-parametric** (no impone una forma concreta a la **decision boundary**) y
de **lazy learning**: el entrenamiento solo guarda los datos y todo el cálculo
ocurre en la predicción. KNN no ajusta pesos ni parámetros; su «modelo» es el
propio **train set**.

## Idea general

1. Guardar `(X_train, y_train)`.
2. Para cada punto nuevo `x`:
   - calcular la distancia de `x` a todos los puntos de **train**;
   - tomar los `k` más cercanos;
   - **classification**: *majority vote* de sus etiquetas;
   - **regression**: promedio de sus valores.
3. Devolver la predicción.

Con `k = 1`, cada punto se asigna a la clase de su vecino más cercano y las
regiones de decisión son un **diagrama de Voronoi**. Al crecer `k`, la
**decision boundary** se vuelve más suave.

## Cómo se mide «cercanía»

La distancia más usada es la **euclidiana**:

```
d(a, b) = sqrt( Σᵢ (aᵢ - bᵢ)² )
```

Otras métricas habituales:

| Distancia | Fórmula | Cuándo usarla |
|---|---|---|
| Euclidean | `sqrt(Σ (aᵢ - bᵢ)²)` | Variables continuas en la misma escala |
| Manhattan | `Σ \|aᵢ - bᵢ\|` | Datos con ruido/outliers; valores en rejilla |
| Minkowski | `(Σ \|aᵢ - bᵢ\|^p)^(1/p)` | Generaliza las anteriores (`p = 1` Manhattan, `p = 2` Euclidean) |
| Cosine | `1 - (a·b) / (‖a‖ ‖b‖)` | Texto o vectores donde importa la dirección, no la longitud |

### La escala importa: standardization

KNN compara magnitudes: si una variable se mide en miles (ingresos) y otra en
unidades (edad), la primera domina por completo la distancia. Por eso hay que
**standardize** cada columna a media 0 y desviación 1, o normalizarla a un rango
fijo, antes de aplicar KNN.

> El escalado se ajusta **solo con el train** y se aplica al test. Ajustarlo con
> todo el dataset produce **data leakage** y métricas artificialmente optimistas.

En el notebook de esta carpeta se ve el efecto: al añadir una variable
irrelevante con escala grande, la **accuracy** cae de **0.94 a 0.49**; al
estandarizar, vuelve a **0.94**.

## Pseudocódigo

```
Entrada: X_train, y_train, k, metrica
1. Guardar (X_train, y_train)                     # "entrenamiento"
2. Para cada punto x en X_test:
   a. D = distancias(x, X_train)                  # n_train valores
   b. I = índices de los k menores valores de D
   c. votos = contar o ponderar las clases y_train[I]
   d. predicción(x) = clase con más votos
3. Devolver las predicciones
```

### Majority vote y pesos

- **Uniforme**: cada vecino vale un voto.
- **Por distancia**: cada vecino pesa `1 / (d + ε)`, de modo que los más
  cercanos influyen más. Puede mejorar cuando `k` es grande, aunque no siempre.

Si hay **empate**, se necesita una regla determinista. En el código gana la
clase con la etiqueta menor (`np.argmax` sobre etiquetas ordenadas). Usar `k`
impar en problemas binarios reduce los empates, pero no los elimina: dos vecinos
pueden estar exactamente a la misma distancia.

## Ejemplo numérico

Puntos de **train** y consulta `q = (2, 3)`:

| Punto | Coordenadas | Clase | Distancia a `q` |
|---|---|---|---|
| A | (1, 1) | 0 | 2.236 |
| B | (2, 2) | 0 | **1.000** |
| C | (4, 5) | 1 | 2.828 |
| D | (5, 5) | 1 | 3.606 |
| E | (1, 4) | 1 | **1.414** |

Con `k = 3` los vecinos son **B (clase 0)**, **E (clase 1)** y **A (clase 0)**:
2 votos contra 1 → `q` se clasifica como **clase 0**.

Con `k = 2` habría un empate 1-1 (B y E). Con pesos por distancia y `k = 3`:

```
clase 0: 1/1.000 + 1/2.236 = 1.447
clase 1: 1/1.414           = 0.707   -> gana la clase 0
```

## Cómo elegir `k`

`k` es el **hyperparameter** principal y controla el **bias-variance tradeoff**:

- `k` **pequeño** (`k = 1`): el modelo se ajusta a cada punto, incluidos los
  *outliers* → **overfitting**, frontera irregular.
- `k` **grande**: se promedian demasiados vecinos, incluso de otras clases →
  **underfitting**, frontera demasiado suave. Con `k = n` siempre gana la clase
  mayoritaria.

Criterios prácticos:

1. Elegir `k` con **cross-validation** sobre el train (o un **holdout** interno);
   el test se reserva para la evaluación final.
2. Probar valores pequeños e impares (3, 5, 7, …).
3. Heurística inicial: `k ≈ sqrt(n)`.
4. Estandarizar y, si hay muchas dimensiones, reducirla antes.

## Decision boundary

En el ejemplo de `knn.py` (tres clases gaussianas 2D) se observa el efecto de
`k`: con `k = 1` las regiones se pegan a cada punto; con `k = 15` o `k = 31` son
más suaves. La imagen se guarda como `knn_frontera.png`, y el notebook compara
`k = 1` con el `k` elegido.

## Complejidad

KNN no entrena, así que el coste se mide en la predicción:

| Fase | Coste |
|---|---|
| Entrenamiento | `O(1)` (solo guardar); memoria `O(n · d)` |
| Predicción de 1 punto | `O(n · d)` |
| Predicción de `m` puntos | `O(m · n · d)` |

Para datasets grandes se usan índices espaciales como **KD-tree** o **Ball
tree**, que aceleran la búsqueda en **baja dimensión** (del orden de `O(log n)`
en promedio). En dimensión alta pierden eficacia.

## Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Muy intuitivo y fácil de explicar | Predicción lenta: compara con todo el train |
| No asume una forma concreta para la frontera | Necesita guardar todo el dataset |
| Captura fronteras no lineales | Sensible a la escala de las variables |
| Sirve para classification y regression | Sensible a variables irrelevantes y al ruido |
| Un solo hyperparameter importante (`k`) | Sufre la **curse of dimensionality** |

## Curse of dimensionality

En muchas dimensiones, todos los puntos tienden a estar **casi a la misma
distancia** entre sí: «vecino más cercano» pierde significado y KNN empeora. Por
eso, con datos de alta dimensión conviene:

- seleccionar variables (eliminar las irrelevantes),
- reducir dimensionalidad (PCA, etc.),
- o usar otra familia de modelos.

## KNN para regression

En lugar de votar, se promedia el valor de los `k` vecinos:

```
ŷ(x) = (1/k) Σ_{i ∈ vecinos(x)} yᵢ        (o media ponderada por 1/distancia)
```

Es útil para datos espaciales o series, pero hereda los mismos problemas:
escala, dimensionalidad y coste de predicción.

---

# El código explicado

## Estructura de `knn.py`

| Bloque | Qué hace |
|---|---|
| `distancias()` | Matriz de distancias entre dos conjuntos de puntos |
| `class KNN` | Clasificador: `fit`, `kneighbors`, `predict`, `predict_proba`, `score` |
| `class Estandarizador` | Standardization ajustada solo con train |
| `dividir_train_test()` | Split reproducible en train/test |
| `matriz_confusion()` | Confusion matrix |
| `_graficar_frontera()` | Gráfico de la decision boundary |
| `__main__` | Ejemplo ejecutable con 3 clases |

El archivo solo depende de **NumPy**; `matplotlib` se importa dentro de la
función de gráfico para que el resto funcione sin él.

## `distancias(X, X_train, metrica)`

```python
def distancias(X, X_train, metrica="euclidiana"):
    X = np.asarray(X, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    if X.shape[1] != X_train.shape[1]:
        raise ValueError("X y X_train deben tener el mismo numero de columnas")

    diferencia = X[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    if metrica == "euclidiana":
        return np.sqrt(np.sum(diferencia**2, axis=2))
    if metrica == "manhattan":
        return np.sum(np.abs(diferencia), axis=2)
    raise ValueError(f"metrica debe ser una de {METRICAS}, se recibio {metrica!r}")
```

- **Entrada**: `X` de forma `(m, d)` y `X_train` de forma `(n, d)`.
- **Salida**: matriz `(m, n)` con la distancia de cada punto de `X` a cada punto
  de `X_train`.
- `X[:, newaxis, :]` tiene forma `(m, 1, d)` y `X_train[newaxis, :, :]` tiene
  forma `(1, n, d)`. Al restarlas, **broadcasting** produce `(m, n, d)`: todas
  las diferencias por pares.
- `axis=2` suma las `d` coordenadas de cada par; `sqrt` completa la euclidiana.
- Manhattan no eleva al cuadrado: suma `|a - b|`.
- **Memoria**: `O(m · n · d)`. Con datasets grandes conviene procesar por
  bloques (*chunking*) o usar KD-tree.
- Se valida que ambos conjuntos tengan el mismo número de columnas.

## `class KNN`

### Constructor y `fit`

```python
def __init__(self, k=3, metrica="euclidiana", pesos="uniforme"):
    ...
    self.k = int(k)
    self.metrica = metrica
    self.pesos = pesos

def fit(self, X, y):
    ...
    self.X_ = X
    self.y_ = y
    self.clases_ = np.unique(y)
    return self
```

- El constructor valida que `k` sea entero `>= 1` y que `metrica`/`pesos` sean
  valores admitidos.
- `fit` **no calcula nada**: convierte a arrays, valida dimensiones, que `X` e
  `y` tengan el mismo número de filas y que `1 <= k <= n_train`.
- Guarda `X_`, `y_` y las clases ordenadas en `clases_` (`np.unique`).
- Devuelve `self` para poder encadenar `KNN(k=5).fit(X, y)`.
- Termina con `_` en los atributos aprendidos, convención de scikit-learn.

### `kneighbors`

```python
def kneighbors(self, X, n_neighbors=None):
    n_neighbors = self.k if n_neighbors is None else int(n_neighbors)
    D = distancias(X, self.X_, self.metrica)
    indices = np.argsort(D, axis=1, kind="stable")[:, :n_neighbors]
    return np.take_along_axis(D, indices, axis=1), indices
```

- Calcula la matriz `(m, n_train)` de distancias.
- `np.argsort(D, axis=1)` ordena **cada fila** y devuelve índices, no valores.
  `kind="stable"` conserva el orden original en caso de empate, lo que hace la
  predicción reproducible.
- `[:, :n_neighbors]` se queda con los `k` vecinos más cercanos.
- `np.take_along_axis(D, indices, axis=1)` recupera las distancias en el orden
  de los índices; devuelve dos matrices `(m, k)`.
- Permite pedir un número de vecinos distinto de `k` sin reentrenar.

### `_votos`

```python
def _votos(self, dist, indices):
    etiquetas_vecinas = self.y_[indices]          # (m, k)
    if self.pesos == "uniforme":
        pesos = np.ones_like(dist)
    else:
        pesos = 1.0 / (dist + np.finfo(float).eps)

    votos = np.zeros((dist.shape[0], self.clases_.size))
    for j, clase in enumerate(self.clases_):
        votos[:, j] = np.sum(pesos * (etiquetas_vecinas == clase), axis=1)
    return votos
```

- `self.y_[indices]` trae la etiqueta de cada vecino: matriz `(m, k)`.
- Con pesos uniformes todos valen 1; con `pesos="distancia"` cada voto vale
  `1/(d + ε)`. El `ε` evita dividir por cero si el vecino coincide con el punto.
- Para cada clase se compara `etiquetas_vecinas == clase`, se multiplica por el
  peso y se suma por fila. El resultado es `(m, n_clases)`.
- No normaliza todavía: eso lo hace `predict_proba`.

### `predict`, `predict_proba` y `score`

```python
def predict_proba(self, X):
    dist, indices = self.kneighbors(X)
    votos = self._votos(dist, indices)
    return votos / votos.sum(axis=1, keepdims=True)

def predict(self, X):
    dist, indices = self.kneighbors(X)
    votos = self._votos(dist, indices)
    return self.clases_[np.argmax(votos, axis=1)]

def score(self, X, y):
    return float(np.mean(self.predict(X) == y))
```

- `predict_proba` normaliza cada fila: cada valor es la fracción de votos (o de
  peso) de esa clase y las filas suman 1.
- `predict` toma `argmax` por fila y lo traduce a la etiqueta con `clases_`.
  `argmax` devuelve el primer máximo, así que un empate lo gana la clase con
  etiqueta menor.
- `score` calcula la **accuracy**: media de comparaciones correctas.

## `class Estandarizador`

```python
def fit(self, X):
    self.media_ = X.mean(axis=0)
    self.desv_ = X.std(axis=0)
    self.desv_[self.desv_ == 0] = 1.0
    return self

def transform(self, X):
    return (X - self.media_) / self.desv_
```

- `axis=0` calcula media y desviación **por columna** (una por variable).
- Si una columna es constante, su desviación es 0 y dividir daría error; se
  reemplaza por 1: la columna queda en 0.
- `fit` guarda los parámetros; `transform` los aplica. `fit_transform` encadena
  ambos.
- Se ajusta solo con `X_train` y se aplica a `X_test` para evitar
  **data leakage**.

## `dividir_train_test`

```python
rng = np.random.default_rng(random_state)
indices = rng.permutation(X.shape[0])
corte = int(round(X.shape[0] * (1.0 - test_size)))
train, test = indices[:corte], indices[corte:]
return X[train], X[test], y[train], y[test]
```

- `rng.permutation(n)` mezcla los índices de forma reproducible si se fija
  `random_state`.
- `corte` deja la fracción `1 - test_size` para train.
- Devuelve cuatro arrays: `X_train, X_test, y_train, y_test`.
- No estratifica por clase; en datasets desbalanceados conviene hacerlo.

## `matriz_confusion`

```python
clases = np.unique(np.concatenate([y_true, y_pred]))
matriz = np.zeros((clases.size, clases.size), dtype=int)
for i, real in enumerate(clases):
    for j, predicha in enumerate(clases):
        matriz[i, j] = np.sum((y_true == real) & (y_pred == predicha))
```

- Reúne todas las clases que aparecen en reales y predicciones.
- La fila `i` es la clase real y la columna `j` la predicha.
- La diagonal son los aciertos; fuera de la diagonal, los errores.
- Devuelve `(matriz, clases)` para poder etiquetar filas y columnas.

## `_graficar_frontera`

```python
xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolucion),
                     np.linspace(y_min, y_max, resolucion))
rejilla = np.c_[xx.ravel(), yy.ravel()]
clases_predichas = modelo.predict(rejilla).reshape(xx.shape)

regiones = np.zeros_like(clases_predichas, dtype=int)
for j, clase in enumerate(modelo.clases_):
    regiones[clases_predichas == clase] = j

ax.contourf(xx, yy, regiones, alpha=0.2)
```

- Construye una rejilla de `resolucion × resolucion` puntos que cubre los datos
  con margen y predice **toda la rejilla** de una vez.
- `ravel` aplana y `np.c_` forma pares `(x1, x2)`; luego se recupera la forma 2D
  con `reshape` para dibujar.
- `contourf` colorea las regiones de cada clase; `scatter` dibuja train y test
  (test con triángulos).
- Usa el backend `Agg` para guardar un PNG sin abrir ventana.
- El coste crece con `resolucion² · n_train`; por eso el valor por defecto es
  moderado.

## Bloque `__main__`

El ejemplo del final del archivo:

1. Genera **3 clases** gaussianas 2D (150 puntos cada una).
2. Hace el **train/test split** y estandariza con el train.
3. Prueba `k = 1, 3, 5, 7, 11, 15, 21, 31` y elige el de mayor accuracy.
4. Entrena el modelo final con `pesos="distancia"`.
5. Imprime la **confusion matrix** y los vecinos de un punto.
6. Genera `knn_frontera.png`.

Salida esperada (resumida):

```
k= 1 -> 0.985 ... k= 7 -> 0.993 ...
Mejor k: 7 (accuracy test: 0.993)
Matriz de confusion (clases [0, 1, 2]):
[[47  1  0]
 [ 0 38  0]
 [ 0  0 49]]
```

## Flujo completo de una predicción

```
modelo = KNN(k=5).fit(X_train, y_train)     # solo guarda
modelo.predict(X_test)
  -> kneighbors(X_test)
     -> distancias(X_test, X_train)         # (m, n_train)
     -> argsort(...)[:, :5]                 # (m, 5) índices
     -> take_along_axis(...)                # (m, 5) distancias
  -> _votos(distancias, índices)            # (m, n_clases)
  -> argmax(votos)                          # (m,) índices de clase
  -> clases_[...]                           # etiquetas finales
```

## Código pedagógico del notebook

El notebook añade tres funciones para explicar el algoritmo por partes:

| Función | Para qué sirve |
|---|---|
| `distancia_euclidiana(a, b)` | Distancia entre dos vectores |
| `distancias_a_todos(x, X)` | Distancia de un punto a todas las filas de `X` |
| `predecir_uno(x, X, y, k)` | KNN mínimo para un solo punto: distancia, `argsort`, voto |
| `graficar_frontera(ax, modelo)` | Rejilla + `predict` + `contourf` para la boundary |

`predecir_uno` es la versión didáctica de `KNN.predict`; sirve para ver el
mecanismo sin la clase completa. El notebook explica cada celda de código justo
debajo de la celda.

---

## API y uso rápido

```python
KNN(k=3, metrica="euclidiana", pesos="uniforme")
```

- `fit(X, y)` → guarda `X_train` / `y_train` y devuelve el modelo.
- `predict(X)` → etiqueta predicha para cada fila.
- `predict_proba(X)` → fracción de votos por clase (matriz `n × n_clases`).
- `kneighbors(X, n_neighbors=None)` → `(distancias, índices)` de los vecinos.
- `score(X, y)` → **accuracy**.

Utilidades:

```python
Estandarizador().fit(X_train).transform(X_test)
dividir_train_test(X, y, test_size=0.2, random_state=42)
matriz_confusion(y_true, y_pred)  # -> (matriz, clases)
```

Ejemplo completo:

```python
import numpy as np
from knn import KNN, Estandarizador, dividir_train_test

rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal([0, 0], 1.0, size=(100, 2)),
    rng.normal([3, 3], 1.0, size=(100, 2)),
])
y = np.repeat([0, 1], 100)

X_train, X_test, y_train, y_test = dividir_train_test(X, y, test_size=0.3, random_state=1)

est = Estandarizador().fit(X_train)          # ajustar SOLO con train
X_train, X_test = est.transform(X_train), est.transform(X_test)

modelo = KNN(k=5, metrica="euclidiana", pesos="distancia").fit(X_train, y_train)
print("Accuracy:", modelo.score(X_test, y_test))
print("Predicciones:", modelo.predict(X_test[:10]))
print("Probabilidades:\n", modelo.predict_proba(X_test[:5]))
```

Detalles de implementación:

- Las distancias se calculan de forma vectorizada con *broadcasting*; la memoria
  usada es `O(m · n · d)`.
- `argsort(kind="stable")` hace que los empates de distancia se resuelvan de
  forma reproducible.
- `k` debe cumplir `1 ≤ k ≤ n_train`; los errores se validan con `ValueError`.
- No depende de `scikit-learn`: solo `numpy` y, para el gráfico, `matplotlib`.

## Archivos y comandos

| Archivo | Contenido |
|---|---|
| `knn.py` | Implementación desde cero + ejemplo ejecutable |
| `knn_unit_tests.py` | Pruebas unitarias con `unittest` (21 tests) |
| `KNN.ipynb` | Notebook ejecutado: teoría, código explicado, gráficos y resultados |
| `EXPLICACION.md` | Este documento |
| `knn_frontera.png` | Gráfico generado por `python knn.py` |

```bash
cd KNN
python knn.py                        # ejemplo + gráfico
python -m unittest knn_unit_tests -v # 21 tests
```

## Resultados del ejemplo

- `knn.py`: 3 clases, 450 puntos, `k = 7` → **accuracy 0.993** en test; la
  confusion matrix es casi diagonal.
- `KNN.ipynb`: 2 clases, `k = 31` → **accuracy 0.958**; estandarizar mejora de
  **0.49** a **0.94** cuando hay una variable de escala grande.
- `pesos="uniforme"` vs `"distancia"` y `euclidiana` vs `manhattan` se comparan
  en el notebook: las diferencias dependen del dataset, no hay una variante
  universalmente mejor.

## Conceptos relacionados

- **Vecindad y distancia**: base de KNN; también en clustering y recomendación.
- **Standardization / normalization**: requisito de los modelos basados en
  distancias.
- **Bias-variance tradeoff**: explica por qué `k` pequeño hace overfitting y `k`
  grande underfitting.
- **Cross-validation**: forma correcta de elegir `k`.
- **KD-tree / Ball tree**: índices espaciales para acelerar la búsqueda.
- **Curse of dimensionality**: motivo del mal rendimiento en alta dimensión.
- **k-means**: unsupervised, agrupa sin etiquetas; no confundir con KNN.
