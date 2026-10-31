
from pathlib import Path
from typing import Any, cast

import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class RecomendadorLibros:
    """Genera recomendaciones de libros usando similitud TF-IDF."""

    def __init__(
        self,
        ruta_libros: str | Path,
        ruta_matriz: str | Path,
    ) -> None:
        self.libros: pd.DataFrame = pd.read_csv(ruta_libros)

        # SciPy no proporciona stubs completos para todas las operaciones
        # de matrices dispersas. Se conserva la matriz cargada como Any
        # para permitir el acceso por índice sin bloquear mypy.
        self.matriz_tfidf: Any = cast(
            Any,
            sp.load_npz(ruta_matriz),
        )

        if len(self.libros) != self.matriz_tfidf.shape[0]:
            raise ValueError(
                "La cantidad de libros no coincide con las filas de la matriz."
            )

    def recomendar(
        self,
        indice_libro: int,
        cantidad: int = 3,
    ) -> pd.DataFrame:
        """Devuelve libros similares al libro seleccionado."""

        if not 0 <= indice_libro < len(self.libros):
            raise ValueError("Índice de libro inválido.")

        if cantidad < 1:
            raise ValueError(
                "La cantidad de recomendaciones debe ser positiva."
            )

        similitudes = cosine_similarity(
            self.matriz_tfidf[indice_libro],
            self.matriz_tfidf,
        ).flatten()

        indices_similares = similitudes.argsort()[::-1]
        indices_similares = [
            int(indice)
            for indice in indices_similares
            if indice != indice_libro
        ][:cantidad]

        recomendaciones = self.libros.iloc[indices_similares].copy()
        recomendaciones["similitud"] = similitudes[indices_similares]

        return recomendaciones
