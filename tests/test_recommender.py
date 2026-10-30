import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from app.recommender import RecomendadorLibros


@pytest.fixture
def recomendador_temporal(tmp_path):
    """Crea un catálogo y una matriz TF-IDF pequeños para las pruebas."""
    libros = pd.DataFrame(
        {
            "title": ["Libro A", "Libro B", "Libro C"],
            "authors": ["Autor A", "Autor B", "Autor C"],
        }
    )

    matriz = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]
        )
    )

    ruta_libros = tmp_path / "libros.csv"
    ruta_matriz = tmp_path / "matriz.npz"

    libros.to_csv(ruta_libros, index=False)
    sp.save_npz(ruta_matriz, matriz)

    return RecomendadorLibros(ruta_libros, ruta_matriz)


def test_recomendar_excluye_libro_seleccionado(recomendador_temporal):
    resultado = recomendador_temporal.recomendar(0, cantidad=2)

    assert "Libro A" not in resultado["title"].values
    assert len(resultado) == 2


def test_recomendar_ordena_por_similitud(recomendador_temporal):
    resultado = recomendador_temporal.recomendar(0, cantidad=2)

    assert resultado.iloc[0]["title"] == "Libro B"
    assert resultado.iloc[0]["similitud"] > resultado.iloc[1]["similitud"]


def test_indice_invalido_lanza_error(recomendador_temporal):
    with pytest.raises(ValueError, match="Índice de libro inválido"):
        recomendador_temporal.recomendar(10)


def test_cantidad_invalida_lanza_error(recomendador_temporal):
    with pytest.raises(ValueError, match="cantidad"):
        recomendador_temporal.recomendar(0, cantidad=0)