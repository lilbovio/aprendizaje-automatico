"""Algoritmo k-means implementado desde cero (solo usamos NumPy para numeros y vectores).

k-means agrupa N puntos en K grupos (clusters). Cada cluster se representa
por su centroide (la media de todos los puntos que pertenecen a el).

Pseudo-codigo:
    1. Elige K centroides iniciales.
    2. Repite hasta que no cambien las asignaciones:
        a. Asigna cada punto al centroide mas cercano (etiqueta).
        b. Recalcula cada centroide como la media de los puntos asignados.

Criterio de parada: cuando ninguna etiqueta cambia entre iteraciones,
o cuando se alcanza el numero maximo de iteraciones.
"""
import numpy as np
from numpy.typing import NDArray


def _inicializar_centroides(X: NDArray, k: int, random_state: int | None = None) -> NDArray:
    """Devuelve k centroides elegidos com el metodo k-means++.

    k-means++ elige el primer centroide al azar y luego los siguientes con
    probabilidad proporcional a la distancia al centroide mas cercano. Esto
    mejora la probabilidad de una buena convergencia inicial.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    centroides = np.empty((k, X.shape[1]))
    indice = rng.integers(0, n)
    centroides[0] = X[indice]

    for i in range(1, k):
        distancias = np.min(
            np.sum((X[:, np.newaxis, :] - centroides[:i][np.newaxis, :, :]) ** 2, axis=2),
            axis=1,
        )
        # Para que arcos con distancia 0 no tengan probabilidad nula.
        distancias = np.maximum(distancias, np.finfo(float).eps)
        probabilidades = distancias / distancias.sum()
        indice = rng.choice(n, p=probabilidades)
        centroides[i] = X[indice]

    return centroides


def _asignar(X: NDArray, centroides: NDArray) -> NDArray:
    """Asigna cada punto al indice del centroide mas cercano (distancia euclidea)."""
    distancias = np.sum((X[:, np.newaxis, :] - centroides[np.newaxis, :, :]) ** 2, axis=2)
    return np.argmin(distancias, axis=1)


def _recalcular_centroides(X: NDArray, etiquetas: NDArray, k: int) -> NDArray:
    """Recalcula cada centroide como la media de los puntos asignados."""
    nuevos = np.zeros((k, X.shape[1]))
    for i in range(k):
        mascara = etiquetas == i
        if mascara.any():
            nuevos[i] = X[mascara].mean(axis=0)
        else:
            # Si un cluster queda vacio, lo reponemos con el punto mas lejano
            # a cualquier centroide para no perder un cluster.
            distancias = np.min(
                np.sum((X[:, np.newaxis, :] - nuevos[np.newaxis, :, :]) ** 2, axis=2), axis=1
            )
            nuevos[i] = X[np.argmax(distancias)]
    return nuevos


def kmeans(
    X: NDArray,
    k: int,
    max_iter: int = 300,
    random_state: int | None = None,
) -> tuple[NDArray, NDArray, float]:
    """Agrupa los puntos de X (n x d) en k clusters.

    Parametros
    ----------
    X : (n, d) matriz de n puntos en d dimensiones.
    k : numero de clusters.
    max_iter : iteraciones maximas del algoritmo.
    random_state : semilla para reproducibilidad.

    Devuelve
    --------
    (etiquetas, centroides, inercia)
        - etiquetas: array (n,) con el indice del cluster de cada punto.
        - centroides: array (k, d) con la posicion final de cada centroide.
        - inercia: suma de las distancias al cuadrado de cada punto a su
          centroide (funcion objetivo que k-means minimiza).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional (n, d)")
    if not (1 <= k <= X.shape[0]):
        raise ValueError(f"k debe estar entre 1 y {X.shape[0]}, se recibio {k}")

    centroides = _inicializar_centroides(X, k, random_state)
    etiquetas = np.zeros(X.shape[0], dtype=int)

    for _ in range(max_iter):
        nuevas = _asignar(X, centroides)
        if np.array_equal(nuevas, etiquetas):
            etiquetas = nuevas
            break
        etiquetas = nuevas
        centroides = _recalcular_centroides(X, etiquetas, k)

    inercia = float(
        np.sum(np.min(np.sum((X[:, np.newaxis, :] - centroides[np.newaxis, :, :]) ** 2, axis=2), axis=1))
    )
    return etiquetas, centroides, inercia


if __name__ == "__main__":
    print("Ejemplo de uso de kmeans:")
    rng = np.random.default_rng(42)
    # Tres grupos bien separados de puntos 2D.
    g1 = rng.normal(loc=[0, 0], scale=0.5, size=(100, 2))
    g2 = rng.normal(loc=[5, 5], scale=0.5, size=(100, 2))
    g3 = rng.normal(loc=[0, 8], scale=0.5, size=(100, 2))
    X = np.vstack([g1, g2, g3])

    etiquetas, centroides, inercia = kmeans(X, k=3, random_state=7)
    print(f"Centroides:\n{centroides}")
    print(f"Inercia: {inercia:.2f}")
    print(f"Distribucion de puntos por cluster: {np.bincount(etiquetas)}")