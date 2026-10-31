
from pathlib import Path

import pandas as pd
import streamlit as st

from app.recommender import RecomendadorLibros


# --------------------------------------------------
# Configuración general
# --------------------------------------------------

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_LIBROS = RAIZ_PROYECTO / "data" / "books.csv"
RUTA_MATRIZ = RAIZ_PROYECTO / "models" / "tfidf_matrix.npz"
RUTA_BANNER = RAIZ_PROYECTO / "assets" / "banner.png"

st.set_page_config(
    page_title="BookMatch | Tu próxima lectura",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# Estilos visuales
# --------------------------------------------------

def aplicar_estilos():
    """Aplica los estilos visuales de BookMatch."""

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8f5ef;
            color: #302c28;
        }

        [data-testid="stSidebar"] {
            background-color: #eee7dc;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] span {
            color: #3e3932 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] {
            background-color: #fffdf9;
            border-radius: 8px;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #302c28 !important;
        }

        /* Encabezado */
        .hero {
            background-color: #eee7dc;
            padding: 1.5rem 2rem;
            border-radius: 18px;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            font-size: 2.5rem;
            color: #3e4a37;
            margin: 0;
        }

        .hero p {
            color: #5d574f;
            font-size: 1.05rem;
            margin-top: 0.5rem;
        }

        /* Tarjetas de libros */
        .book-card {
            background-color: #fffdf9;
            border: 1px solid #e7dfd3;
            border-radius: 16px;
            padding: 1rem;
            height: 100%;
        }

        .book-card h4 {
            color: #3e4a37;
            min-height: 3rem;
        }

        .muted {
            color: #756e64;
            font-size: 0.9rem;
        }

        div[data-testid="stImage"] img {
            border-radius: 10px;
        }

        hr {
            border-color: #e7dfd3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Carga del recomendador
# --------------------------------------------------

@st.cache_resource
def cargar_recomendador():
    """Carga el catálogo y la matriz TF-IDF."""

    return RecomendadorLibros(
        RUTA_LIBROS,
        RUTA_MATRIZ,
    )


# --------------------------------------------------
# Presentación de libros
# --------------------------------------------------

def mostrar_libro(libro, mostrar_similitud=False):
    """Muestra la portada y los datos de un libro."""

    portada = libro.get("cover_image_url", "")

    if pd.notna(portada) and str(portada).strip():
        st.image(
            portada,
            use_container_width=True,
        )
    else:
        st.caption("Portada no disponible")

    st.markdown(f"#### {libro['title']}")

    st.markdown(
        f"<p class='muted'>Por {libro['authors']}</p>",
        unsafe_allow_html=True,
    )

    generos = libro.get("genres", "No disponible")

    if pd.notna(generos):
        st.caption(f"Géneros: {generos}")

    calificacion = libro.get("rating")

    if pd.notna(calificacion):
        st.caption(f"⭐ Calificación: {calificacion}")

    if mostrar_similitud and "similitud" in libro:
        st.caption(
            f"Similitud: {libro['similitud']:.2f}"
        )


# --------------------------------------------------
# Aplicación principal
# --------------------------------------------------

def main():
    aplicar_estilos()

    recomendador = cargar_recomendador()
    libros = recomendador.libros

    # Banner decorativo
    if RUTA_BANNER.exists():
        st.image(
            str(RUTA_BANNER),
            use_container_width=True,
        )

    # Encabezado usando componentes nativos
    st.markdown("# 📚 BookMatch")

    st.subheader("Tu próxima lectura está aquí.")

    st.write(
        "Descubre libros que conectan con tus gustos. "
        "BookMatch analiza el contenido de cada libro y encuentra "
        "lecturas con características similares."
    )

    st.divider()

    # ----------------------------------------------
    # Panel lateral
    # ----------------------------------------------

    with st.sidebar:
        st.markdown("## 📖 Tu libro favorito")

        st.write(
            "Busca un libro de nuestro catálogo para encontrar "
            "lecturas similares."
        )

        titulo_seleccionado = st.selectbox(
            "Selecciona un libro",
            options=libros["title"].tolist(),
            index=None,
            placeholder="Escribe un título...",
        )

        st.divider()

        st.markdown("### ¿Cómo funciona?")

        st.markdown(
            """
            1. Selecciona un libro.
            2. BookMatch analiza su contenido.
            3. Descubre tres recomendaciones similares.
            """
        )

        st.caption(
            "Las recomendaciones se basan en similitud de contenido "
            "mediante TF-IDF y similitud coseno."
        )

        if titulo_seleccionado is not None:
            indice_libro = libros.index[
                libros["title"] == titulo_seleccionado
            ][0]

            st.divider()
            st.markdown("### Libro seleccionado")

            mostrar_libro(
                libros.iloc[indice_libro]
            )

    # ----------------------------------------------
    # Recomendaciones
    # ----------------------------------------------

    if titulo_seleccionado is None:
        st.info(
            "Selecciona un libro en el panel izquierdo "
            "para descubrir tus próximas lecturas."
        )
        return

    indice_libro = libros.index[
        libros["title"] == titulo_seleccionado
    ][0]

    recomendaciones = recomendador.recomendar(
        indice_libro,
        cantidad=3,
    )

    st.markdown("## ✨ Recomendaciones para ti")

    st.write(
        "Estas lecturas comparten características de contenido "
        "con el libro que seleccionaste."
    )

    columnas = st.columns(3)

    for columna, (_, libro) in zip(
        columnas,
        recomendaciones.iterrows(),
    ):
        with columna:
            st.markdown(
                "<div class='book-card'>",
                unsafe_allow_html=True,
            )

            mostrar_libro(
                libro,
                mostrar_similitud=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    st.caption(
        "BookMatch · Sistema de recomendación de libros "
        "basado en contenido."
    )


if __name__ == "__main__":
    main()
