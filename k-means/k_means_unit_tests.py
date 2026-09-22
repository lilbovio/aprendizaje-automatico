"""Pruebas unitarias para el algoritmo k-means.

Ejecutar con:
    python -m unittest k_means_unit_tests -v
o:
    python k_means_unit_tests.py
"""
import unittest

import numpy as np

from k_means import kmeans


class TestKMeans(unittest.TestCase):
    def setUp(self):
        # Datos conocidos: 4 puntos en un cuadrado.
        self.X_cuadrado = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [10.0, 0.0],
                [10.0, 1.0],
            ]
        )

    def test_numero_de_clusters_y_formas(self):
        """Devuelve k centroides y una etiqueta por punto."""
        X = self.X_cuadrado
        k = 2
        etiquetas, centroides, inercia = kmeans(X, k, random_state=1)

        self.assertEqual(etiquetas.shape, (X.shape[0],))
        self.assertEqual(centroides.shape, (k, X.shape[1]))
        self.assertIsInstance(inercia, float)
        self.assertGreaterEqual(inercia, 0.0)
        self.assertEqual(set(etiquetas), {0, 1})

    def test_clusters_lineales_separables(self):
        """Puntos de una linea (1D) se separan correctamente en K=2."""
        X = np.array([[0.0], [0.1], [10.0], [10.1]])
        etiquetas, centroides, _ = kmeans(X, 2, random_state=3)

        self.assertEqual(set(etiquetas), {0, 1})
        # Los dos grupos deben quedar separados entre si.
        self.assertTrue(np.all(np.abs(centroides[0] - centroides[1]) > 1.0))

    def test_centroide_es_la_media_de_su_grupo(self):
        """Con grupos perfectamente separados, los centroides son la media exacta."""
        g1 = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        g2 = np.array([[10.0, 0.0], [10.0, 2.0], [10.0, 1.0]])
        X = np.vstack([g1, g2])

        etiquetas, centroides, _ = kmeans(X, 2, random_state=5)

        # Centroide 1: media de g1 = (1, 0); centroide 2: media de g2 = (10, 1).
        esperado_1 = np.array([1.0, 0.0])
        esperado_2 = np.array([10.0, 1.0])
        self.assertTrue(
            np.allclose(centroides, [esperado_1, esperado_2], atol=1e-6)
            or np.allclose(centroides, [esperado_2, esperado_1], atol=1e-6)
        )
        self.assertEqual(len(set(etiquetas)), 2)

    def test_reproducibilidad(self):
        """La misma semilla produce exactamente el mismo resultado."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))

        e1, c1, i1 = kmeans(X, 4, random_state=42)
        e2, c2, i2 = kmeans(X, 4, random_state=42)

        np.testing.assert_array_equal(e1, e2)
        np.testing.assert_array_equal(c1, c2)
        self.assertEqual(i1, i2)

    def test_semilla_distinta_puede_dar_mismo_resultado(self):
        """Con datos bien separados el resultado no depende de la semilla."""
        rng = np.random.default_rng(1)
        X = np.vstack(
            [
                rng.normal(loc=[0, 0], scale=0.1, size=(50, 2)),
                rng.normal(loc=[5, 5], scale=0.1, size=(50, 2)),
            ]
        )

        e1, _, _ = kmeans(X, 2, random_state=1)
        e2, _, _ = kmeans(X, 2, random_state=999)
        # Solo importa que las particiones sean equivalentes (orden de etiquetas ignorado).
        self.assertEqual(
            sorted(e1.tolist()),
            sorted(e2.tolist()),
        )

    def test_la_inercia_no_empeora(self):
        """La funcion objetivo debe ser <= a la de una asignacion trivial."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(100, 2))

        _, _, inercia = kmeans(X, 3, random_state=11)
        # Una asignacion arbitraria (todo al punto medio) tiene inercia mayor.
        centro = X.mean(axis=0)
        inercia_trivial = float(np.sum(np.sum((X - centro) ** 2, axis=1)))
        self.assertLessEqual(inercia, inercia_trivial)

    def test_k_igual_a_numero_de_puntos(self):
        """Con k = n, cada punto es su propio centroide y la inercia es 0."""
        X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        etiquetas, centroides, inercia = kmeans(X, 3, random_state=0)

        np.testing.assert_array_equal(np.sort(centroides, axis=0), np.sort(X, axis=0))
        np.testing.assert_array_equal(np.sort(etiquetas), np.array([0, 1, 2]))
        self.assertAlmostEqual(inercia, 0.0, places=6)

    def test_k_invalido_lanza_error(self):
        """k fuera de rango debe lanzar ValueError."""
        X = np.array([[0.0], [1.0]])
        with self.assertRaises(ValueError):
            kmeans(X, k=0)
        with self.assertRaises(ValueError):
            kmeans(X, k=5)

    def test_entrada_unidimensional_lanza_error(self):
        """X debe ser bidimensional (n, d)."""
        X = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            kmeans(X, k=2)

    def test_mas_iteraciones_no_empeoran_inercia(self):
        """Mas iteraciones nunca deberian dejar la inercia peor."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 4))

        _, _, i_max1 = kmeans(X, 3, max_iter=1, random_state=0)
        _, _, i_max10 = kmeans(X, 3, max_iter=10, random_state=0)
        _, _, i_max300 = kmeans(X, 3, max_iter=300, random_state=0)

        self.assertLessEqual(i_max10, i_max1 + 1e-9)
        self.assertLessEqual(i_max300, i_max10 + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)