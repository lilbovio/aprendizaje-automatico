"""Pruebas unitarias de knn.py.

Ejecutar desde la carpeta KNN:

    python -m unittest knn_unit_tests -v
"""
import unittest

import numpy as np

from knn import (
    Estandarizador,
    KNN,
    distancias,
    dividir_train_test,
    matriz_confusion,
)


class TestDistancias(unittest.TestCase):
    def test_euclidiana(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = distancias(X, np.array([[0.0, 0.0]]))
        np.testing.assert_allclose(D[:, 0], [0.0, 5.0])

    def test_manhattan(self):
        X = np.array([[1.0, 1.0], [-2.0, 3.0]])
        D = distancias(X, np.array([[0.0, 0.0]]), metrica="manhattan")
        np.testing.assert_allclose(D[:, 0], [2.0, 5.0])

    def test_metrica_invalida(self):
        with self.assertRaises(ValueError):
            distancias(np.zeros((1, 2)), np.zeros((1, 2)), metrica="coseno")

    def test_dimensiones_incompatibles(self):
        with self.assertRaises(ValueError):
            distancias(np.zeros((1, 2)), np.zeros((1, 3)))


class TestKNN(unittest.TestCase):
    def setUp(self):
        # Dos grupos bien separados en 2D.
        self.X_train = np.array([[0.0, 0.0], [0.5, 0.5], [10.0, 10.0], [10.5, 10.5]])
        self.y_train = np.array([0, 0, 1, 1])
        self.X_test = np.array([[0.1, 0.1], [9.9, 9.9]])
        self.y_test = np.array([0, 1])

    def test_acierta_grupos_separados(self):
        modelo = KNN(k=1).fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(modelo.predict(self.X_test), self.y_test)
        self.assertEqual(modelo.score(self.X_test, self.y_test), 1.0)

    def test_ejemplo_documento(self):
        X_toy = np.array([[1, 1], [2, 2], [4, 5], [5, 5], [1, 4]], dtype=float)
        y_toy = np.array([0, 0, 1, 1, 1])
        modelo = KNN(k=3).fit(X_toy, y_toy)
        self.assertEqual(modelo.predict(np.array([[2.0, 3.0]]))[0], 0)

    def test_kneighbors_ordenados(self):
        modelo = KNN(k=3).fit(self.X_train, self.y_train)
        dist, idx = modelo.kneighbors(self.X_test)
        self.assertEqual(dist.shape, (2, 3))
        self.assertEqual(idx.shape, (2, 3))
        self.assertTrue(np.all(np.diff(dist, axis=1) >= 0))
        # Los vecinos de un punto son los puntos de su propio grupo.
        np.testing.assert_array_equal(self.y_train[idx[0]], [0, 0, 1])

    def test_kneighbors_n_neighbors_distinto_de_k(self):
        modelo = KNN(k=3).fit(self.X_train, self.y_train)
        dist, idx = modelo.kneighbors(self.X_test[:1], n_neighbors=1)
        self.assertEqual(dist.shape, (1, 1))
        self.assertEqual(int(idx[0, 0]), 0)

    def test_predict_proba(self):
        modelo = KNN(k=4, pesos="distancia").fit(self.X_train, self.y_train)
        proba = modelo.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (2, 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_pesos_distancia(self):
        X = np.array([[0.0], [1.0], [10.0]])
        y = np.array([0, 1, 1])
        # Con k=3 uniforme gana la clase 1 (2 votos). El vecino de la clase 0
        # esta mucho mas cerca, asi que con pesos por distancia debe ganar 0.
        self.assertEqual(KNN(k=3).fit(X, y).predict(np.array([[0.0]]))[0], 1)
        self.assertEqual(
            KNN(k=3, pesos="distancia").fit(X, y).predict(np.array([[0.0]]))[0], 0
        )

    def test_k_mayor_que_muestras(self):
        with self.assertRaises(ValueError):
            KNN(k=10).fit(self.X_train, self.y_train)

    def test_k_invalido(self):
        with self.assertRaises(ValueError):
            KNN(k=0)

    def test_metrica_invalida(self):
        with self.assertRaises(ValueError):
            KNN(k=1, metrica="coseno")

    def test_predecir_sin_fit(self):
        with self.assertRaises(RuntimeError):
            KNN(k=1).predict(self.X_test)

    def test_longitudes_incompatibles(self):
        with self.assertRaises(ValueError):
            KNN(k=1).fit(self.X_train, np.array([0, 1]))


class TestUtilidades(unittest.TestCase):
    def test_estandarizador(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        Z = Estandarizador().fit_transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0)
        np.testing.assert_allclose(Z.std(axis=0), 1.0)

    def test_estandarizador_columna_constante(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        Z = Estandarizador().fit_transform(X)
        np.testing.assert_allclose(Z[:, 1], 0.0)

    def test_estandarizador_usa_parametros_del_train(self):
        train = np.array([[1.0], [3.0]])
        test = np.array([[5.0]])
        est = Estandarizador().fit(train)
        self.assertAlmostEqual(float(est.transform(test)[0, 0]), 3.0)

    def test_dividir_train_test(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_train, X_test, y_train, y_test = dividir_train_test(
            X, y, test_size=0.2, random_state=1
        )
        self.assertEqual(X_train.shape[0], 40)
        self.assertEqual(X_test.shape[0], 10)
        self.assertEqual(y_train.shape[0], 40)
        self.assertEqual(y_test.shape[0], 10)
        # Reproducible con la misma semilla.
        X_train2, *_ = dividir_train_test(X, y, test_size=0.2, random_state=1)
        np.testing.assert_array_equal(X_train, X_train2)

    def test_test_size_invalido(self):
        with self.assertRaises(ValueError):
            dividir_train_test(np.zeros((10, 1)), np.zeros(10), test_size=1.5)

    def test_matriz_confusion(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        matriz, clases = matriz_confusion(y_true, y_pred)
        np.testing.assert_array_equal(clases, [0, 1])
        np.testing.assert_array_equal(matriz, [[1, 1], [1, 2]])
        self.assertEqual(matriz.sum(), y_true.size)


if __name__ == "__main__":
    unittest.main()
