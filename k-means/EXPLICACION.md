# Algoritmo k-means

El **k-means** es un algoritmo de **aprendizaje no supervisado** que agrupa `N` puntos en `K` grupos (clusters), donde `K` lo elige el usuario. Su objetivo es minimizar la **inercia**:

```
inercia = Σᵢ Punto más cercano (min_j ||xᵢ - cⱼ||²)
```

Es decir, minimizar la suma de los cuadrados de las distancias de cada punto a su centroide.

## Idea general

1. Cada cluster se representa por un **centroide** `cⱼ`, que es la media de todos los puntos asignados a ese grupo.
2. Cada punto `xᵢ` se asigna al centroide **más cercano** (distancia euclídea).
3. Todo el problema se reduce a: encontrar los `K` centroides que hagan la inercia mínima.

El problema es **NP-duro** en general, de modo que k-means busca una solución aproximada de forma iterativa y eficiente.

## Pseudo-código

```
Entrada: X (puntos), K (clusters), max_iter
1. Elegir K centroides iniciales (k-means++)
2. Repetir hasta converger o alcanzar max_iter:
   a. Asignar cada punto al centroide más cercano  ->  etiquetas
   b. Recalcular cada centroide como la media de sus puntos
3. Devolver etiquetas, centroides e inercia
```

Se dice que el algoritmo ha **convergido** cuando las asignaciones dejan de cambiar entre dos iteraciones consecutivas.

## Paso a paso

### 1. Inicialización (k-means++)

Elegir los primeros centroides al azar suele terminar en malas soluciones. El método **k-means++** elige:

- El primer centroide al azar.
- Cada siguiente centroide con probabilidad proporcional al cuadrado de la distancia al centroide más cercano.

Esto reparte los centroides iniciales por todo el espacio y mejora la convergencia.

### 2. Asignación

Para cada punto, se calcula la distancia al cuadrado a todos los centroides y se toma el mínimo:

```python
distancias = sum((x[i] - c)² para cada c)
etiqueta[i] = argmin(distancias)
```

### 3. Actualización

Cada centroide se reemplaza por la media aritmética de los puntos que le fueron asignados:

```python
c[j] = media(x[i] para todos los i con etiqueta[i] == j)
```

> Nota: si un cluster queda vacío, el código reasigna ese centroide al punto más lejano de cualquier centroide para no desperdiciar el cluster.

### 4. Parada

El bucle se detiene cuando las etiquetas no cambian o cuando se llega a `max_iter`.

## Complejidad

- Por **iteración**: `O(N · K · d)`, con `N` puntos, `K` clusters y `d` dimensiones.
- El número de iteraciones no es fijo; en la práctica suele ser pequeño, sobre todo con una buena inicialización (k-means++).

## Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Rápido y simple | Hay que elegir `K` de antemano |
| Escala bien a grandes datasets | Sensible a la inicialización (mitigado con k-means++) |
| Fácil de interpretar | Asume clusters esféricos de tamaño similar |
| Converge en un óptimo local | No tolera bien valores atípicos (*outliers*) |

## Uso del código

```python
import numpy as np
from k_means import kmeans

rng = np.random.default_rng(42)
g1 = rng.normal(loc=[0, 0], scale=0.5, size=(100, 2))
g2 = rng.normal(loc=[5, 5], scale=0.5, size=(100, 2))
X = np.vstack([g1, g2])

etiquetas, centroides, inercia = kmeans(X, k=2, random_state=7)

print("Centroides:", centroides)
print("Inercia:", inercia)
print("Puntos por cluster:", np.bincount(etiquetas))
```

Y para lanzar las pruebas:

```bash
python -m unittest k_means_unit_tests -v
```

## Firma de la función

```python
kmeans(X, k, max_iter=300, random_state=None) -> (etiquetas, centroides, inercia)
```

- `X` : `(n, d)` matriz de puntos.
- `k` : número de clusters (`1 <= k <= n`).
- `max_iter` : iteraciones máximas.
- `random_state` : semilla para reproducibilidad.

Devuelve:

- `etiquetas`: índice del cluster de cada punto.
- `centroides`: posición final de cada cluster.
- `inercia`: suma de distancias al cuadrado a cada centroide (cuanto menor, mejor).

## Conceptos relacionados

- **Inercia / SSE**: la función objetivo que k-means minimiza.
- **Elbow method**: técnica para elegir `K` representando la inercia frente a `K` y buscando el "codo".
- **K-medoids**: variante que usa puntos reales del dataset como centros.
- **Gaussian Mixture Models (GMM)**: generalización probabilística que permite clusters con forma elíptica.