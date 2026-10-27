"""Algoritmo KNN (k vecinos mas cercanos) implementado desde cero (solo NumPy).

KNN es un algoritmo de aprendizaje supervisado que no construye un modelo
explicito: guarda el conjunto de entrenamiento y, para predecir un punto
nuevo, busca sus k vecinos mas cercanos y usa sus etiquetas.

Pseudo-codigo (clasificacion):
    1. Guardar X_train e y_train.
    2. Para cada punto nuevo x:
        a. Calcular la distancia de x a todos los puntos de entrenamiento.
        b. Tomar los k indices con menor distancia.
        c. Votar: la clase mas frecuente entre esos k vecinos (o la de mayor
           peso si se usa peso por distancia).
    3. Devolver la clase ganadora.

El ejemplo de uso completo esta al final del archivo (bloque __main__).
"""
import numpy as np
from numpy.typing import NDArray

METRICAS = ("euclidiana", "manhattan")
PESOS = ("uniforme", "distancia")


def _validar_metrica(metrica: str, pesos: str) -> None:
    if metrica not in METRICAS:
        raise ValueError(f"metrica debe ser una de {METRICAS}, se recibio {metrica!r}")
    if pesos not in PESOS:
        raise ValueError(f"pesos debe ser uno de {PESOS}, se recibio {pesos!r}")


def distancias(
    X: NDArray,
    X_train: NDArray,
    metrica: str = "euclidiana",
) -> NDArray:
    """Matriz (m, n_train) con la distancia de cada punto a cada vecino.

    m es el numero de puntos de X y n_train el numero de puntos de X_train.

    - euclidiana: sqrt(sum((a - b)^2))
    - manhattan:  sum(|a - b|)
    """
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


class KNN:
    """Clasificador k-Nearest Neighbors.

    Parametros
    ----------
    k : numero de vecinos a consultar (> 0).
    metrica : "euclidiana" o "manhattan".
    pesos : "uniforme" (todos los vecinos votan igual) o "distancia"
        (cada vecino pesa 1 / distancia; los mas cercanos mandan mas).
    """

    def __init__(self, k: int = 3, metrica: str = "euclidiana", pesos: str = "uniforme"):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k debe ser un entero >= 1, se recibio {k!r}")
        _validar_metrica(metrica, pesos)
        self.k = int(k)
        self.metrica = metrica
        self.pesos = pesos

    def fit(self, X: NDArray, y: NDArray) -> "KNN":
        """Guarda el conjunto de entrenamiento (KNN no entrena nada mas)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional (n, d)")
        if y.ndim != 1:
            raise ValueError("y debe ser un vector unidimensional (n,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo numero de filas")
        if self.k > X.shape[0]:
            raise ValueError(
                f"k ({self.k}) no puede ser mayor que el numero de muestras ({X.shape[0]})"
            )

        self.X_ = X
        self.y_ = y
        self.clases_ = np.unique(y)
        return self

    def kneighbors(self, X: NDArray, n_neighbors: int | None = None) -> tuple[NDArray, NDArray]:
        """Devuelve (distancias, indices) de los vecinos mas cercanos de cada punto.

        Las distancias e indices tienen forma (n_puntos, n_neighbors) y estan
        ordenados de menor a mayor distancia.
        """
        if not hasattr(self, "X_"):
            raise RuntimeError("El modelo no esta entrenado: llama primero a fit()")
        n_neighbors = self.k if n_neighbors is None else int(n_neighbors)
        if not (1 <= n_neighbors <= self.X_.shape[0]):
            raise ValueError(
                f"n_neighbors debe estar entre 1 y {self.X_.shape[0]}, "
                f"se recibio {n_neighbors}"
            )

        D = distancias(X, self.X_, self.metrica)
        # argsort estable: en caso de empate gana el vecino que aparece antes.
        indices = np.argsort(D, axis=1, kind="stable")[:, :n_neighbors]
        return np.take_along_axis(D, indices, axis=1), indices

    def _votos(self, dist: NDArray, indices: NDArray) -> NDArray:
        """Suma de pesos por clase para cada punto: matriz (n, n_clases)."""
        etiquetas_vecinas = self.y_[indices]
        if self.pesos == "uniforme":
            pesos = np.ones_like(dist)
        else:
            # eps evita division por cero cuando el vecino es identico al punto.
            pesos = 1.0 / (dist + np.finfo(float).eps)

        votos = np.zeros((dist.shape[0], self.clases_.size))
        for j, clase in enumerate(self.clases_):
            votos[:, j] = np.sum(pesos * (etiquetas_vecinas == clase), axis=1)
        return votos

    def predict_proba(self, X: NDArray) -> NDArray:
        """Probabilidad estimada de cada clase (fraccion de votos)."""
        dist, indices = self.kneighbors(X)
        votos = self._votos(dist, indices)
        return votos / votos.sum(axis=1, keepdims=True)

    def predict(self, X: NDArray) -> NDArray:
        """Clase predicha para cada fila de X."""
        dist, indices = self.kneighbors(X)
        votos = self._votos(dist, indices)
        # argmax devuelve el primer maximo: los empates los gana la clase
        # con la etiqueta mas pequena (orden de np.unique).
        return self.clases_[np.argmax(votos, axis=1)]

    def score(self, X: NDArray, y: NDArray) -> float:
        """Exactitud (accuracy) sobre el conjunto (X, y)."""
        y = np.asarray(y)
        if y.shape[0] != np.asarray(X).shape[0]:
            raise ValueError("X e y deben tener el mismo numero de filas")
        return float(np.mean(self.predict(X) == y))


class Estandarizador:
    """Normaliza cada columna a media 0 y desviacion tipica 1.

    KNN usa distancias, asi que una variable con escala grande (p. ej. ingresos
    en miles) domina sobre otra con escala pequena (p. ej. edad). Ajustar el
    escalado solo con el train evita fuga de informacion (data leakage).
    """

    def fit(self, X: NDArray) -> "Estandarizador":
        X = np.asarray(X, dtype=float)
        self.media_ = X.mean(axis=0)
        self.desv_ = X.std(axis=0)
        self.desv_[self.desv_ == 0] = 1.0  # columnas constantes
        return self

    def transform(self, X: NDArray) -> NDArray:
        if not hasattr(self, "media_"):
            raise RuntimeError("El estandarizador no esta ajustado: llama a fit() primero")
        return (np.asarray(X, dtype=float) - self.media_) / self.desv_

    def fit_transform(self, X: NDArray) -> NDArray:
        return self.fit(X).transform(X)


def dividir_train_test(
    X: NDArray,
    y: NDArray,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Divide (X, y) en train/test mezclando los indices al azar."""
    X = np.asarray(X)
    y = np.asarray(y)
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size debe estar en (0, 1), se recibio {test_size}")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(X.shape[0])
    corte = int(round(X.shape[0] * (1.0 - test_size)))
    train, test = indices[:corte], indices[corte:]
    return X[train], X[test], y[train], y[test]


def matriz_confusion(y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
    """Matriz de confusion: filas = clase real, columnas = clase predicha."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    clases = np.unique(np.concatenate([y_true, y_pred]))
    matriz = np.zeros((clases.size, clases.size), dtype=int)
    for i, real in enumerate(clases):
        for j, predicha in enumerate(clases):
            matriz[i, j] = np.sum((y_true == real) & (y_pred == predicha))
    return matriz, clases


def _graficar_frontera(
    modelo: KNN,
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    archivo: str = "knn_frontera.png",
    resolucion: int = 150,
) -> None:
    """Dibuja la frontera de decision y los puntos de train/test."""
    import matplotlib

    matplotlib.use("Agg")  # Backend sin ventana: guarda directo a archivo.
    import matplotlib.pyplot as plt

    # Rejilla que cubre los datos con un pequeno margen.
    X = np.vstack([X_train, X_test])
    margen = 0.5
    x_min, x_max = X[:, 0].min() - margen, X[:, 0].max() + margen
    y_min, y_max = X[:, 1].min() - margen, X[:, 1].max() + margen
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolucion),
        np.linspace(y_min, y_max, resolucion),
    )
    rejilla = np.c_[xx.ravel(), yy.ravel()]
    clases_predichas = modelo.predict(rejilla).reshape(xx.shape)
    # Convertimos etiquetas a enteros consecutivos para pintar las regiones.
    codigos, _ = np.unique(modelo.clases_, return_inverse=True)
    regiones = np.zeros_like(clases_predichas, dtype=int)
    for j, clase in enumerate(modelo.clases_):
        regiones[clases_predichas == clase] = j

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, regiones, alpha=0.2, levels=np.arange(codigos.size + 1) - 0.5)
    colores = plt.cm.viridis(np.linspace(0, 1, modelo.clases_.size))

    for clase, color in zip(modelo.clases_, colores):
        mascara_train = y_train == clase
        mascara_test = y_test == clase
        ax.scatter(
            X_train[mascara_train, 0],
            X_train[mascara_train, 1],
            c=[color],
            s=30,
            alpha=0.6,
            label=f"Train clase {clase}",
        )
        ax.scatter(
            X_test[mascara_test, 0],
            X_test[mascara_test, 1],
            c=[color],
            s=90,
            marker="^",
            edgecolors="black",
            linewidths=0.8,
        )
        ax.scatter([], [], c=[color], s=90, marker="^", edgecolors="black", label=f"Test clase {clase}")

    ax.set_title(
        f"Frontera de decision KNN (k={modelo.k}, metrica={modelo.metrica}, "
        f"pesos={modelo.pesos}) - accuracy test={modelo.score(X_test, y_test):.3f}"
    )
    ax.set_xlabel("X1 (estandarizada)")
    ax.set_ylabel("X2 (estandarizada)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)
    print(f"Grafico guardado en: {archivo}")


if __name__ == "__main__":
    print("Ejemplo de uso de KNN (3 clases sinteticas):")

    # 1. Datos: 3 grupos gaussianos 2D que se solapan un poco.
    rng = np.random.default_rng(42)
    centros = np.array([[0.0, 0.0], [3.5, 4.0], [-3.0, 3.5]])
    X_partes, y_partes = [], []
    for etiqueta, centro in enumerate(centros):
        X_partes.append(rng.normal(loc=centro, scale=1.0, size=(150, 2)))
        y_partes.append(np.full(150, etiqueta))
    X = np.vstack(X_partes)
    y = np.concatenate(y_partes)

    # 2. Division train/test y estandarizacion (ajustada solo con train).
    X_train, X_test, y_train, y_test = dividir_train_test(
        X, y, test_size=0.3, random_state=7
    )
    est = Estandarizador().fit(X_train)
    X_train, X_test = est.transform(X_train), est.transform(X_test)

    # 3. Eleccion de k con el conjunto de test.
    print("\nAccuracy en test segun k:")
    mejor_k, mejor_accuracy = 1, -1.0
    for k in (1, 3, 5, 7, 11, 15, 21, 31):
        modelo = KNN(k=k).fit(X_train, y_train)
        accuracy = modelo.score(X_test, y_test)
        print(f"  k={k:2d} -> {accuracy:.3f}")
        if accuracy > mejor_accuracy:
            mejor_k, mejor_accuracy = k, accuracy

    # 4. Modelo final con el mejor k.
    modelo = KNN(k=mejor_k, pesos="distancia").fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"\nMejor k: {mejor_k} (accuracy test: {modelo.score(X_test, y_test):.3f})")

    matriz, clases = matriz_confusion(y_test, y_pred)
    print(f"\nMatriz de confusion (clases {clases.tolist()}):")
    print(matriz)

    # 5. Vecinos de un punto concreto (utilidad didactica).
    punto = X_test[0:1]
    dist, indices = modelo.kneighbors(punto)
    print(f"\nVecinos mas cercanos del primer punto de test (k={mejor_k}):")
    for d, i in zip(dist[0], indices[0]):
        print(f"  distancia={d:.3f}  clase={y_train[i]}  posicion={X_train[i]}")

    # 6. Grafico de la frontera de decision.
    try:
        _graficar_frontera(modelo, X_train, y_train, X_test, y_test)
    except ImportError:
        print("No se encontro matplotlib. Instalalo con: pip install matplotlib")
