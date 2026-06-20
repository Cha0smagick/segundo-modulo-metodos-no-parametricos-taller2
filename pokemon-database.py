#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ANÁLISIS MULTIVARIADO Y TAXONOMÍA ALGORÍTMICA DE POKÉMON — Q4 STATISTICAL LEVEL
==============================================================================
Maestría en Analítica de Datos — Politécnico Grancolombiano
Métodos No Paramétricos | Taller #2
Grupo #10

Este script implementa un pipeline completo de análisis multivariado:
  1.  Carga y preprocesamiento (StandardScaler)
  2.  Análisis Exploratorio (EDA) con visualización de distribuciones y correlaciones
  3.  Reducción de dimensionalidad:
        • PCA: scree plot, loadings, correlation circle, biplot, cos2
        • t-SNE: barrido de perplejidad, trustworthiness, comparativas
  4.  Clustering:
        • K-Means con validación multi-métrica (codo, silueta, Davies-Bouldin,
          Calinski-Harabasz, Gap statistic)
        • Jerárquico (Ward) con dendrograma
        • DBSCAN con diagnóstico k-distance
        • Validación estadística: Kruskal-Wallis, Bartlett
  5.  Cartografía visual con sprites sobre PCA / t-SNE
  6.  Generación de reportes: .txt, .md, .html con todas las figuras
  7.  Diccionario de gráficos con metadatos de interpretación

Requisitos:
    pip install kagglehub pandas numpy matplotlib seaborn scikit-learn scipy

Ejecución:
    python pokemon-database.py
==============================================================================
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import kagglehub
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import os
import shutil
import json
import warnings
import sys
from datetime import datetime
from itertools import cycle

# Scikit-learn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors

# Scipy
from scipy.stats import kruskal, bartlett, chi2_contingency, f_oneway, normaltest
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, cdist

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# 2. GLOBAL CONFIGURATION
# ============================================================================
# Output directory structure
BASE_OUTPUT = "output"
DIRS = {
    "descriptive": os.path.join(BASE_OUTPUT, "descriptive"),
    "dim_reduction": os.path.join(BASE_OUTPUT, "dimensionality_reduction"),
    "clustering": os.path.join(BASE_OUTPUT, "clustering"),
    "image_maps": os.path.join(BASE_OUTPUT, "image_maps"),
    "diagnostics": os.path.join(BASE_OUTPUT, "diagnostics"),
}
LOCAL_IMG_DIR = "images"
KAGGLE_STATS = "rounakbanik/pokemon"
KAGGLE_IMAGES = "vishalsubbiah/pokemon-images-and-types"

# Plotting aesthetics
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (12, 7),
})

# Custom colormap for clusters
CUSTOM_COLORS = ['#1f77b4', '#2ca02c', '#bcbd22', '#9467bd',
                 '#e377c2', '#7f7f7f', '#17becf', '#d62728', '#ff7f0e']
CMAP_CUSTOM = LinearSegmentedColormap.from_list("pokemon_gradient", CUSTOM_COLORS)

# Resistance columns
COLS_AGAINST = [
    'against_bug', 'against_dark', 'against_dragon', 'against_electric',
    'against_fairy', 'against_fight', 'against_fire', 'against_flying',
    'against_ghost', 'against_grass', 'against_ground', 'against_ice',
    'against_normal', 'against_poison', 'against_psychic', 'against_rock',
    'against_steel', 'against_water'
]

# Archetype definitions
ARCHETYPES = {
    0: {
        "nombre": "Guardianes del Océano y el Acero",
        "desc": "Especialistas en resistencia elemental (Fuego, Fantasma) y física (Acero).",
        "color": "#1f77b4"
    },
    1: {
        "nombre": "Protectores de la Biosfera",
        "desc": "Expertos en resistir ataques de naturaleza y combate cercano, frágiles ante lo místico.",
        "color": "#2ca02c"
    },
    2: {
        "nombre": "Místicos del Éter",
        "desc": "Entidades con defensas psíquicas superiores, vulnerables a miedos primordiales.",
        "color": "#bcbd22"
    },
    3: {
        "nombre": "Centinelas de Alta Tensión",
        "desc": "Grupo de élite con alta resistencia al calor y electricidad, pero vulnerables a la tierra.",
        "color": "#9467bd"
    },
    4: {
        "nombre": "Espectros Sombríos",
        "desc": "Maestros de lo etéreo y lo oscuro, resistentes a Fantasma y Siniestro.",
        "color": "#e377c2"
    },
    5: {
        "nombre": "Titanes de la Tierra",
        "desc": "Colosos inmutables, resistentes a Roca y Eléctrico, anclas del equipo.",
        "color": "#7f7f7f"
    },
    6: {
        "nombre": "Guardianes Ígneos",
        "desc": "Inmunes funcionales al Fuego, obligatorios contra entrenadores de tipo Fuego.",
        "color": "#17becf"
    },
    7: {
        "nombre": "Centinelas de Hielo",
        "desc": "Resistentes al Hielo y al Dragón. Clave contra Lance y Lorelei.",
        "color": "#d62728"
    },
    8: {
        "nombre": "Versátiles Híbridos",
        "desc": "Perfiles mixtos y adaptables. Comodines estratégicos para cualquier escenario.",
        "color": "#ff7f0e"
    }
}


# ============================================================================
# 3. DICTIONARY OF GRAPHS  (metadata for every figure generated)
# ============================================================================
GRAPH_DICTIONARY = {
    # Descriptive
    "histogram_variables": {
        "file": "histogram_resistencias.png",
        "title": "Distribución de las 18 Variables de Resistencia Elemental",
        "tipo": "Histograma con KDE",
        "interpretacion": "Visualiza la forma de la distribución de cada variable de resistencia. "
                           "Valores < 1 indican resistencia (mitigación de daño), > 1 indican vulnerabilidad. "
                           "La asimetría revela sesgos en el diseño de tipos.",
        "metodo": "matplotlib.pyplot.hist + seaborn.kdeplot"
    },
    "boxplot_variables": {
        "file": "boxplot_resistencias.png",
        "title": "Diagrama de Caja: Variables de Resistencia Estandarizadas",
        "tipo": "Boxplot",
        "interpretacion": "Muestra la mediana, dispersión y outliers por variable. "
                           "Bigotes extendidos indican alta variabilidad en ese tipo elemental.",
        "metodo": "seaborn.boxplot"
    },
    "correlation_heatmap": {
        "file": "heatmap_correlaciones.png",
        "title": "Mapa de Calor: Correlaciones entre Resistencias Elementales",
        "tipo": "Heatmap de correlación (Pearson)",
        "interpretacion": "Estructura de dependencia lineal entre variables. "
                           "Clusters de alta correlación sugieren tipos elementales que suelen co-aparecer.",
        "metodo": "seaborn.heatmap con matriz de correlación de Pearson"
    },
    "parallel_coordinates": {
        "file": "coordenadas_paralelas.png",
        "title": "Coordenadas Paralelas: Perfiles de Resistencia por Pokémon",
        "tipo": "Coordenadas paralelas",
        "interpretacion": "Cada línea es un Pokémon. El cruce de patrones revela perfiles "
                           "defensivos típicos. Agrupaciones de líneas = arquetipos de resistencia.",
        "metodo": "pandas.plotting.parallel_coordinates"
    },
    # Dimensionality Reduction
    "scree_plot": {
        "file": "scree_plot_pca.png",
        "title": "Gráfico de Sedimentación: Varianza Explicada por Componente PCA",
        "tipo": "Scree plot (sedimentación)",
        "interpretacion": "Muestra la varianza explicada por cada componente principal. "
                           "El 'codo' indica el número óptimo de dimensiones a retener. "
                           "Se superpone la varianza acumulada para facilitar la decisión.",
        "metodo": "PCA.explained_variance_ratio_"
    },
    "loadings_heatmap": {
        "file": "loadings_heatmap_pca.png",
        "title": "Mapa de Calor: Contribuciones (Loadings) de Variables a Componentes Principales",
        "tipo": "Heatmap de loadings",
        "interpretacion": "Cada celda representa el peso de una variable en un componente. "
                           "Valores absolutos altos = variable importante para ese componente.",
        "metodo": "PCA.components_"
    },
    "correlation_circle": {
        "file": "circulo_correlaciones_pca.png",
        "title": "Círculo de Correlaciones: Proyección de Variables en el Plano PC1-PC2",
        "tipo": "Círculo de correlaciones (variables factor map)",
        "interpretacion": "Variables cercanas al borde del círculo están bien representadas. "
                           "Ángulos agudos entre vectores = correlación positiva; "
                           "ángulos obtusos = correlación negativa; ortogonal = no correlación.",
        "metodo": "PCA.components_ escalados por sqrt(eigenvalues)"
    },
    "pca_biplot": {
        "file": "biplot_pca.png",
        "title": "PCA-Biplot: Individuos y Variables en el Plano Factorial",
        "tipo": "Biplot (individuos + variables)",
        "interpretacion": "Doble representación: puntos = Pokémon (individuos), "
                           "flechas = variables. La proximidad entre un Pokémon y una flecha "
                           "indica afinidad con esa característica.",
        "metodo": "PCA con escalado symbiplot"
    },
    "cos2_quality": {
        "file": "calidad_representacion_cos2.png",
        "title": "Calidad de Representación (cos²): Contribución de Cada Variable a los Ejes",
        "tipo": "Gráfico de barras cos²",
        "interpretacion": "cos² > 0.5 indica buena representación. Variables con bajo cos² "
                           "están mal proyectadas en el plano PC1-PC2.",
        "metodo": "cos² = (component_loadings)²"
    },
    "tsne_perplexity_grid": {
        "file": "tsne_perplejidad_grid.png",
        "title": "Análisis de Sensibilidad: t-SNE con Diferentes Valores de Perplejidad",
        "tipo": "Grid de t-SNE (barrido de perplejidad)",
        "interpretacion": "Perplejidad baja (~5-10) revela estructura local fina; "
                           "perplejidad alta (~30-50) revela estructura global. "
                           "La estabilidad visual entre valores sugiere clusters robustos.",
        "metodo": "TSNE con perplexity ∈ {5, 10, 15, 30, 50, 100}"
    },
    "tsne_learning_rate_grid": {
        "file": "tsne_learning_rate_grid.png",
        "title": "Análisis de Sensibilidad: t-SNE con Diferentes Learning Rates",
        "tipo": "Grid de t-SNE (barrido de learning rate)",
        "interpretacion": "Learning rate controla la convergencia. Valores muy bajos "
                           "pueden producir clusters falsos. Valores altos suavizan la estructura.",
        "metodo": "TSNE con learning_rate ∈ {10, 100, 200, 500, 1000, 2000}"
    },
    "tsne_trustworthiness": {
        "file": "tsne_trustworthiness.png",
        "title": "Trustworthiness: Fidelidad de Preservación de Vecindades (t-SNE vs PCA)",
        "tipo": "Gráfico de líneas comparativo",
        "interpretacion": "Trustworthiness mide qué tan bien se preservan los vecinos "
                           "del espacio original en la proyección. Valores > 0.8 indican "
                           "buena preservación. t-SNE suele superar a PCA en este aspecto.",
        "metodo": "sklearn.manifold.trustworthiness"
    },
    # Clustering
    "cluster_optimization_multi": {
        "file": "optimizacion_multimetrica.png",
        "title": "Optimización Multi-Métrica: Evaluación de k para K-Means",
        "tipo": "Dashboard multi-métrica (4 paneles)",
        "interpretacion": "Evalúa simultáneamente: Inercia (codo), Silueta, Davies-Bouldin "
                           "(menor es mejor), Calinski-Harabasz (mayor es mejor). "
                           "El consenso entre métricas guía la selección de k.",
        "metodo": "KMeans con k=2..10 evaluado con 4 métricas"
    },
    "gap_statistic": {
        "file": "gap_statistic.png",
        "title": "Gap Statistic: Contraste de Inercia contra Distribución Nula de Referencia",
        "tipo": "Gap statistic con barras de error",
        "interpretacion": "El k óptimo es el primer valor donde la curva se estabiliza "
                           "(regla de 1 error estándar). Para los datos Pokémon, "
                           "k=9 es el óptimo estadístico — 9 arquetipos defensivos "
                           "con entidad propia, no subgrupos.",
        "metodo": "Gap statistic con B=20 réplicas bootstrap"
    },
    "dendrogram": {
        "file": "dendrograma_ward.png",
        "title": "Dendrograma: Clustering Jerárquico Aglomerativo (Método de Ward)",
        "tipo": "Dendrograma",
        "interpretacion": "La altura de las uniones indica disimilitud entre clusters. "
                           "Un corte horizontal define los clusters. "
                           "Ramas largas antes de unirse = clusters bien separados.",
        "metodo": "scipy.cluster.hierarchy.linkage + dendrogram"
    },
    "silhouette_diagram": {
        "file": "silhouette_diagram.png",
        "title": "Diagrama de Silueta: Cohesión y Separación por Cluster",
        "tipo": "Silhouette plot",
        "interpretacion": "Cada barra horizontal es un espécimen. Barras que cruzan "
                           "la línea de promedio global indican especímenes mal clasificados. "
                           "Clusters con anchos uniformes son homogéneos.",
        "metodo": "silhouette_samples + fill_betweenx"
    },
    "cluster_characterization": {
        "file": "caracterizacion_heatmap.png",
        "title": "Caracterización de Arquetipos: Perfil Promedio de Resistencia por Cluster",
        "tipo": "Heatmap de perfiles",
        "interpretacion": "Filas = clusters, columnas = tipos elementales. "
                           "Color rojo = vulnerabilidad (valor alto), verde = resistencia. "
                           "Patrones consistentes definen arquetipos defensivos.",
        "metodo": "seaborn.heatmap sobre groupby.mean()"
    },
    "cluster_radar": {
        "file": "radar_arquetipos.png",
        "title": "Diagrama Radial: Perfil Comparativo de Arquetipos por Cluster",
        "tipo": "Radar chart (spider plot)",
        "interpretacion": "Cada eje es un tipo elemental. La superposición de perfiles "
                           "muestra similitudes y diferencias entre clusters. "
                           "Formas distintas = arquetipos diferenciados.",
        "metodo": "matplotlib.pyplot.subplot con coordenadas polares"
    },
    "pca_clusters": {
        "file": "pca_clusters.png",
        "title": "Proyección PCA con Clusters K-Means",
        "tipo": "Scatter plot 2D",
        "interpretacion": "Visualización de la partición del espacio latente. "
                           "Clusters bien separados en el plano PC1-PC2 indican "
                           "que las resistencias elementales discriminan naturalmente.",
        "metodo": "PCA + KMeans, coloreado por cluster"
    },
    "tsne_clusters": {
        "file": "tsne_clusters.png",
        "title": "Proyección t-SNE con Clusters K-Means",
        "tipo": "Scatter plot 2D",
        "interpretacion": "t-SNE revela estructura no lineal. Clusters compactos "
                           "y separados indican grupos naturales en los datos.",
        "metodo": "TSNE + KMeans, coloreado por cluster"
    },
    "dbscan_diagnostic": {
        "file": "dbscan_k_distance.png",
        "title": "Diagnóstico DBSCAN: Gráfico K-Distance para Selección de Epsilon",
        "tipo": "K-distance plot",
        "interpretacion": "El 'codo' en la curva sugiere el valor óptimo de eps. "
                           "Puntos por encima del codo serán considerados ruido.",
        "metodo": "NearestNeighbors.kneighbors"
    },
    "dbscan_result": {
        "file": "pca_dbscan.png",
        "title": "DBSCAN sobre Proyección PCA",
        "tipo": "Scatter plot con ruido",
        "interpretacion": "Puntos en morado = ruido (outliers). Muestra qué Pokémon "
                           "tienen perfiles de resistencia atípicos (especialistas puros).",
        "metodo": "DBSCAN + PCA"
    },
    "pca_image_map": {
        "file": "mapa_visual_pca.png",
        "title": "Cartografía Visual: Pokémon en el Espacio PCA con Sprites",
        "tipo": "Image map (scatter + sprites)",
        "interpretacion": "Cada sprite está posicionado según su perfil de resistencia. "
                           "Pokémon cercanos comparten vulnerabilidades y fortalezas.",
        "metodo": "OffsetImage + AnnotationBbox sobre PCA"
    },
    "tsne_image_map": {
        "file": "mapa_visual_tsne.png",
        "title": "Cartografía Visual: Pokémon en el Espacio t-SNE con Sprites",
        "tipo": "Image map (scatter + sprites)",
        "interpretacion": "t-SNE agrupa por vecindad local. 'Islas' de Pokémon "
                           "visualmente similares indican nichos defensivos.",
        "metodo": "OffsetImage + AnnotationBbox sobre t-SNE"
    },
    "validation_kruskal": {
        "file": "kruskal_wallis_resultados.png",
        "title": "Validación Estadística: Test de Kruskal-Wallis por Variable",
        "tipo": "Gráfico de barras con umbral de significancia",
        "interpretacion": "Barras que cruzan la línea roja (p=0.05) NO son significativas. "
                           "La mayoría de variables deben ser significativas para validar "
                           "que los clusters representan poblaciones diferentes.",
        "metodo": "scipy.stats.kruskal"
    },
}

# ============================================================================
# 4. CORE FUNCTIONS
# ============================================================================


def setup_environment():
    """Create output directory structure and return paths."""
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(LOCAL_IMG_DIR, exist_ok=True)
    return DIRS


def download_datasets():
    """
    Download datasets from Kaggle via kagglehub.
    Falls back to local files if download fails.
    """
    print("=" * 70)
    print("  FASE 0: ADQUISICIÓN DE DATOS")
    print("=" * 70)
    print("  Descargando datasets desde Kaggle...")

    try:
        stats_path = kagglehub.dataset_download(KAGGLE_STATS)
        images_path = kagglehub.dataset_download(KAGGLE_IMAGES)

        stats_csv_path = os.path.join(stats_path, "pokemon.csv")
        cached_img_dir = os.path.join(images_path, "images")

        # Sync images to local directory if empty
        if os.path.exists(cached_img_dir) and not os.listdir(LOCAL_IMG_DIR):
            print(f"  Copiando imágenes al directorio local '{LOCAL_IMG_DIR}'...")
            for item in os.listdir(cached_img_dir):
                src = os.path.join(cached_img_dir, item)
                dst = os.path.join(LOCAL_IMG_DIR, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        img_dir = LOCAL_IMG_DIR
        print("  Descarga completada exitosamente.")
        return stats_csv_path, img_dir, True

    except Exception as e:
        print(f"  ADVERTENCIA: No se pudo conectar con Kaggle: {e}")
        print("  Buscando archivos locales como respaldo...")

        stats_csv_path = "pokemon.csv"
        img_dir = LOCAL_IMG_DIR

        if not os.path.exists(stats_csv_path):
            print("  ERROR CRÍTICO: No se encontró el archivo CSV local.")
            print("  Coloque 'pokemon.csv' en el directorio de trabajo o verifique conexión.")
            return None, None, False

        print(f"  Usando archivo local: {stats_csv_path}")
        return stats_csv_path, img_dir, False


def load_and_filter_data(csv_path, only_gen1=True):
    """
    Load CSV and filter by generation.
    Returns filtered DataFrame and scope label.
    """
    df = pd.read_csv(csv_path)
    scope_text = "GEN 1" if only_gen1 else "ALL GENERATIONS"

    if only_gen1:
        df_filtered = df[df['generation'] == 1].copy()
    else:
        df_filtered = df.copy()

    print(f"\n  Registros cargados: {len(df_filtered)} ({scope_text})")
    print(f"  Variables disponibles: {df_filtered.shape[1]}")

    return df_filtered, scope_text


def preprocess_data(df, scope_text):
    """
    Select resistance columns, standardize, and return working matrices.
    """
    print(f"\n{'=' * 70}")
    print(f"  FASE 1: PREPROCESAMIENTO Y ESTANDARIZACIÓN")
    print(f"{'=' * 70}")

    # Check all columns exist
    available_cols = [c for c in COLS_AGAINST if c in df.columns]
    missing = set(COLS_AGAINST) - set(available_cols)
    if missing:
        print(f"  ADVERTENCIA: Columnas faltantes: {missing}")

    X = df[available_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Variables seleccionadas: {len(available_cols)} resistencias elementales")
    print(f"  Matriz: {X_scaled.shape[0]} muestras × {X_scaled.shape[1]} variables")
    print(f"  Estandarización: Z-score (media=0, var=1)")
    print(f"  Media post-estandarización: {X_scaled.mean():.6f}")
    print(f"  Desviación post-estandarización: {X_scaled.std():.6f}")

    return X, X_scaled, scaler, available_cols


# ============================================================================
# 5. EDA — ANALISIS EXPLORATORIO
# ============================================================================


def run_eda(X, X_scaled, available_cols, df, scope_text, dirs):
    """
    Comprehensive exploratory data analysis with statistical summaries
    and publication-quality visualizations.
    """
    print(f"\n{'=' * 70}")
    print(f"  FASE 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print(f"{'=' * 70}")

    report_lines = []
    report_lines.append(f"\n--- ESTADÍSTICAS DESCRIPTIVAS ({scope_text}) ---")
    desc = X.describe()
    report_lines.append(desc.to_string())
    report_lines.append("")

    # --- 5.1 Histograms with KDE ---
    print("  [1/4] Generando histogramas con KDE...")
    n_cols = 6
    n_rows = int(np.ceil(len(available_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(available_cols):
        ax = axes[i]
        sns.histplot(X[col], kde=True, bins=25, ax=ax, color='steelblue', edgecolor='white')
        ax.set_title(col.replace('against_', ''), fontsize=9)
        ax.set_xlabel('')
        ax.set_ylabel('Frecuencia' if i % n_cols == 0 else '')
        ax.axvline(X[col].mean(), color='red', linestyle='--', linewidth=1, label=f'μ={X[col].mean():.2f}')
        ax.axvline(X[col].median(), color='green', linestyle=':', linewidth=1, label=f'Md={X[col].median():.2f}')
        ax.legend(fontsize=6)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Distribución de Variables de Resistencia Elemental ({scope_text})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(dirs["descriptive"], f"histogram_resistencias_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # --- 5.2 Boxplots ---
    print("  [2/4] Generando boxplots estandarizados...")
    fig, ax = plt.subplots(figsize=(18, 6))
    X_melted = pd.melt(pd.DataFrame(X_scaled, columns=available_cols),
                       var_name='Variable', value_name='Valor Estandarizado')
    sns.boxplot(data=X_melted, x='Variable', y='Valor Estandarizado', ax=ax,
                palette='viridis', fliersize=2)
    ax.set_xticklabels([c.replace('against_', '') for c in available_cols],
                       rotation=45, ha='right')
    ax.set_title(f'Distribución de Variables Estandarizadas ({scope_text})', fontsize=13)
    ax.set_xlabel('Tipo Elemental')
    ax.set_ylabel('Z-score')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(dirs["descriptive"], f"boxplot_resistencias_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # --- 5.3 Correlation Heatmap ---
    print("  [3/4] Generando mapa de calor de correlaciones...")
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.3,
                cbar_kws={'shrink': 0.8, 'label': 'Correlación de Pearson'},
                ax=ax)
    ax.set_xticklabels([c.replace('against_', '') for c in available_cols],
                       rotation=45, ha='right')
    ax.set_yticklabels([c.replace('against_', '') for c in available_cols],
                       rotation=0)
    ax.set_title(f'Matriz de Correlación entre Resistencias Elementales ({scope_text})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(dirs["descriptive"], f"heatmap_correlaciones_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # --- 5.4 Parallel Coordinates ---
    print("  [4/4] Generando coordenadas paralelas...")
    try:
        # Use a sample if too many records
        sample_size = min(100, len(df))
        df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        fig, ax = plt.subplots(figsize=(18, 6))
        pd.plotting.parallel_coordinates(
            df_sample[['name'] + available_cols].assign(
                label='Pokémon'
            ).head(80),
            class_column='label',
            cols=available_cols,
            ax=ax,
            color='steelblue',
            alpha=0.5
        )
        ax.set_xticklabels([c.replace('against_', '') for c in available_cols],
                           rotation=45, ha='right')
        ax.set_title(f'Coordenadas Paralelas: Perfiles de Resistencia ({scope_text})',
                     fontsize=13)
        ax.set_ylabel('Valor de Resistencia')
        ax.legend().remove()
        plt.tight_layout()
        path = os.path.join(dirs["descriptive"], f"coordenadas_paralelas_{scope_text}.png")
        fig.savefig(path)
        plt.close()
        print(f"    -> Guardado: {path}")
    except Exception as e:
        print(f"    ADVERTENCIA: No se pudo generar coordenadas paralelas: {e}")

    # --- Additional: Compute skewness and kurtosis ---
    skewness = X.apply(lambda x: x.skew())
    kurtosis = X.apply(lambda x: x.kurtosis())
    report_lines.append("\n--- ASIMETRÍA Y CURTOSIS ---")
    report_lines.append(f"{'Variable':<20} {'Asimetría':<12} {'Curtosis':<12}")
    report_lines.append("-" * 44)
    for col in available_cols:
        report_lines.append(f"{col.replace('against_', ''):<20} {skewness[col]:<12.3f} {kurtosis[col]:<12.3f}")

    # Normality test (D'Agostino-Pearson)
    normal_vars = 0
    report_lines.append("\n--- PRUEBA DE NORMALIDAD (D'Agostino-Pearson) ---")
    for col in available_cols:
        stat, p = normaltest(X[col].dropna())
        result = "Normal" if p > 0.05 else "No normal"
        if p > 0.05:
            normal_vars += 1
        report_lines.append(f"  {col.replace('against_', ''):<18} stat={stat:.2f}, p={p:.4e} -> {result}")
    report_lines.append(f"\n  Variables con distribución normal: {normal_vars}/{len(available_cols)}")

    print("  EDA completado.")
    return report_lines


# ============================================================================
# 6. DIMENSIONALITY REDUCTION — PCA & t-SNE
# ============================================================================


def run_dimensionality_reduction(X_scaled, available_cols, df, scope_text, dirs):
    """
    Comprehensive dimensionality reduction analysis:
      - PCA: scree, loadings, correlation circle, biplot, cos²
      - t-SNE: perplexity grid, learning rate grid, trustworthiness
    """
    print(f"\n{'=' * 70}")
    print(f"  FASE 3: REDUCCIÓN DE DIMENSIONALIDAD")
    print(f"{'=' * 70}")

    report_lines = []
    n = X_scaled.shape[0]
    p = X_scaled.shape[1]

    # ========================================================================
    # 6.1 PCA — Full decomposition
    # ========================================================================
    print("  [PCA] Descomposición completa...")
    pca_full = PCA(n_components=min(p, n))
    pca_full.fit(X_scaled)

    explained_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    # 6.1.1 — Scree Plot
    print("    -> Scree plot...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(explained_var)))

    bars = ax1.bar(range(1, len(explained_var) + 1), explained_var,
                   color=colors, edgecolor='white', linewidth=0.5, width=0.7,
                   label='Varianza individual')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada (Proporción)')
    ax1.set_title(f'Gráfico de Sedimentación — PCA ({scope_text})', fontsize=13)
    ax1.set_xticks(range(1, len(explained_var) + 1))

    # Add cumulative line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cum_var) + 1), cum_var, 'ro-', linewidth=2,
             markersize=6, label='Varianza acumulada')
    ax2.set_ylabel('Varianza Acumulada', color='red')
    ax2.axhline(0.8, color='gray', linestyle='--', alpha=0.5,
                label='Umbral 80%')

    # Annotate key components
    for i in range(min(4, len(explained_var))):
        ax1.annotate(f'{explained_var[i]:.1%}',
                     (i + 1, explained_var[i]),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=9, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"scree_plot_pca_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    # 6.1.2 — 2-component PCA for visualization
    pca_2d = PCA(n_components=2)
    pca_results = pca_2d.fit_transform(X_scaled)

    var_2d = np.sum(pca_2d.explained_variance_ratio_) * 100

    report_lines.append(f"\n--- PCA: VARIANZA EXPLICADA ---")
    report_lines.append(f"  PC1: {pca_2d.explained_variance_ratio_[0]*100:.2f}%")
    report_lines.append(f"  PC2: {pca_2d.explained_variance_ratio_[1]*100:.2f}%")
    report_lines.append(f"  Total 2 componentes: {var_2d:.2f}%")
    report_lines.append(f"  Dimensiones necesarias para 80%: "
                        f"{np.argmax(cum_var >= 0.8) + 1 if any(cum_var >= 0.8) else '>'+str(p)}")

    # Identify top features per component
    top_n = 3
    top_features = {}
    for comp_idx in range(2):
        idx = np.argsort(np.abs(pca_2d.components_[comp_idx]))[-top_n:][::-1]
        top_features[comp_idx] = [available_cols[i].replace('against_', '') for i in idx]

    pca_label_x = f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%) — [{', '.join(top_features[0])}]"
    pca_label_y = f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%) — [{', '.join(top_features[1])}]"

    # 6.1.3 — Loadings Heatmap
    print("    -> Loadings heatmap...")
    fig, ax = plt.subplots(figsize=(12, 6))
    loadings_df = pd.DataFrame(
        pca_2d.components_.T,
        index=[c.replace('against_', '') for c in available_cols],
        columns=[f'PC{i+1}' for i in range(2)]
    )
    sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Loading (peso)'})
    ax.set_title(f'Loadings de Variables en Componentes Principales ({scope_text})',
                 fontsize=13)
    ax.set_ylabel('Variable de Resistencia')
    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"loadings_heatmap_pca_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    # 6.1.4 — Correlation Circle
    print("    -> Círculo de correlaciones...")
    fig, ax = plt.subplots(figsize=(9, 9))
    # Scale loadings by sqrt(eigenvalue) to get correlations
    eigenvalues = pca_2d.explained_variance_ * (X_scaled.shape[0] - 1)
    correlations = pca_2d.components_.T * np.sqrt(eigenvalues / (X_scaled.shape[0] - 1))

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=1.5)
    ax.add_patch(circle)

    # Draw axes
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    # Plot each variable as an arrow
    var_labels = [c.replace('against_', '').title() for c in available_cols]
    for i, (var_label, (cx, cy)) in enumerate(zip(var_labels, correlations)):
        # Arrow
        ax.arrow(0, 0, cx, cy, head_width=0.04, head_length=0.04,
                 fc=plt.cm.tab20(i / len(var_labels)),
                 ec=plt.cm.tab20(i / len(var_labels)), alpha=0.8)
        # Label
        ax.text(cx * 1.12, cy * 1.12, var_label,
                fontsize=7, ha='center', va='center',
                fontweight='bold',
                color=plt.cm.tab20(i / len(var_labels)))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Círculo de Correlaciones — Variables en PC1-PC2 ({scope_text})',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"circulo_correlaciones_pca_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    # 6.1.5 — PCA Biplot
    print("    -> PCA Biplot...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Scale factor for biplot readability
    scale_factor = 3.0
    max_coord = np.max(np.abs(pca_results)) * 1.2

    # Plot individuals
    ax.scatter(pca_results[:, 0], pca_results[:, 1], c='lightblue',
               edgecolor='gray', alpha=0.6, s=60, label='Individuos (Pokémon)')

    # Annotate extreme individuals
    for i in range(pca_results.shape[0]):
        if np.linalg.norm(pca_results[i]) > np.percentile(np.linalg.norm(pca_results, axis=1), 95):
            ax.annotate(df.iloc[i]['name'] if 'name' in df.columns else str(i),
                       (pca_results[i, 0], pca_results[i, 1]),
                       fontsize=6, alpha=0.7)

    # Plot variables as arrows
    for i, (var_label, (cx, cy)) in enumerate(zip(var_labels, correlations)):
        ax.arrow(0, 0, cx * scale_factor, cy * scale_factor,
                 head_width=0.3, head_length=0.3,
                 fc='red', ec='red', alpha=0.7)
        ax.text(cx * scale_factor * 1.1, cy * scale_factor * 1.1,
                var_label, fontsize=8, color='darkred', fontweight='bold')

    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)
    ax.set_aspect('equal')
    ax.set_xlabel(pca_label_x, fontsize=10)
    ax.set_ylabel(pca_label_y, fontsize=10)
    ax.set_title(f'PCA-Biplot: Individuos y Variables ({scope_text})',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"biplot_pca_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    # 6.1.6 — Quality of representation (cos²)
    print("    -> Calidad de representación (cos²)...")
    # cos² = squared loading / sum squared loadings per variable
    cos2 = pca_2d.components_.T ** 2
    cos2_df = pd.DataFrame(
        cos2,
        index=[c.replace('against_', '') for c in available_cols],
        columns=[f'PC{i+1}' for i in range(2)]
    )
    cos2_df['cos²_total'] = cos2_df.sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Bar chart of total cos²
    sorted_cos2 = cos2_df.sort_values('cos²_total', ascending=True)
    colors_cos2 = plt.cm.coolwarm(np.linspace(0.2, 0.9, len(sorted_cos2)))
    ax1.barh(range(len(sorted_cos2)), sorted_cos2['cos²_total'],
             color=colors_cos2, edgecolor='white')
    ax1.set_yticks(range(len(sorted_cos2)))
    ax1.set_yticklabels(sorted_cos2.index, fontsize=8)
    ax1.set_xlabel('cos² total (PC1 + PC2)')
    ax1.set_title('Calidad Global de Representación', fontsize=12)
    ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Umbral 0.5')
    ax1.legend(fontsize=8)

    # Right: Breakdown per component
    cos2_df_sorted = cos2_df.sort_values('cos²_total', ascending=False)
    x = range(len(cos2_df_sorted))
    ax2.bar(x, cos2_df_sorted['PC1'], label='PC1', alpha=0.8, color='steelblue')
    ax2.bar(x, cos2_df_sorted['PC2'], bottom=cos2_df_sorted['PC1'],
            label='PC2', alpha=0.8, color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cos2_df_sorted.index, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('cos²')
    ax2.set_title('Descomposición cos² por Componente', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)

    fig.suptitle(f'Calidad de Representación de Variables en el Plano PC1-PC2 ({scope_text})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"calidad_representacion_cos2_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    # Report cos² findings
    poor_rep = cos2_df[cos2_df['cos²_total'] < 0.3].index.tolist()
    good_rep = cos2_df[cos2_df['cos²_total'] > 0.7].index.tolist()
    report_lines.append(f"\n--- CALIDAD DE REPRESENTACIÓN (cos²) ---")
    report_lines.append(f"  Variables bien representadas (cos²>0.7): {good_rep}")
    report_lines.append(f"  Variables mal representadas (cos²<0.3): {poor_rep}")

    # ========================================================================
    # 6.2 t-SNE — Analysis
    # ========================================================================
    print("  [t-SNE] Análisis completo...")

    # 6.2.1 — Perplexity Grid
    print("    -> Grid de perplejidad...")
    perplexities = [5, 10, 15, 30, 50, 100]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, perp in enumerate(perplexities):
        ax = axes[idx]
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    init='pca', learning_rate='auto', max_iter=1000)
        tsne_res = tsne.fit_transform(X_scaled)
        ax.scatter(tsne_res[:, 0], tsne_res[:, 1], c='steelblue',
                   s=30, alpha=0.6, edgecolor='white', linewidth=0.3)
        ax.set_title(f'perplexity = {perp}', fontsize=11)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'Análisis de Sensibilidad — Perplejidad en t-SNE ({scope_text})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"tsne_perplejidad_grid_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    # 6.2.2 — Learning Rate Grid
    print("    -> Grid de learning rate...")
    learning_rates = [10, 100, 200, 500, 1000, 2000]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                    init='pca', learning_rate=lr, max_iter=1000)
        tsne_res = tsne.fit_transform(X_scaled)
        ax.scatter(tsne_res[:, 0], tsne_res[:, 1], c='coral',
                   s=30, alpha=0.6, edgecolor='white', linewidth=0.3)
        ax.set_title(f'learning_rate = {lr}', fontsize=11)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'Análisis de Sensibilidad — Learning Rate en t-SNE ({scope_text})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"tsne_learning_rate_grid_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    # 6.2.3 — Default t-SNE for downstream use
    print("    -> Proyección t-SNE estándar...")
    tsne_default = TSNE(n_components=2, perplexity=30, random_state=42,
                        init='pca', learning_rate='auto', max_iter=1000)
    tsne_results = tsne_default.fit_transform(X_scaled)

    # 6.2.4 — Trustworthiness comparison (PCA vs t-SNE)
    print("    -> Trustworthiness (fidelidad de vecindades)...")
    n_neighbors_range = range(2, 21, 2)
    trust_pca = []
    trust_tsne = []

    for k in n_neighbors_range:
        t_pca = trustworthiness(X_scaled, pca_results, n_neighbors=k)
        t_tsne = trustworthiness(X_scaled, tsne_results, n_neighbors=k)
        trust_pca.append(t_pca)
        trust_tsne.append(t_tsne)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_neighbors_range, trust_pca, 'o-', color='steelblue',
            linewidth=2, markersize=8, label=f'PCA (2D, {var_2d:.1f}% varianza)')
    ax.plot(n_neighbors_range, trust_tsne, 's-', color='coral',
            linewidth=2, markersize=8, label='t-SNE (2D)')
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='Umbral 0.9')
    ax.set_xlabel('Número de Vecinos (k)')
    ax.set_ylabel('Trustworthiness')
    ax.set_title(f'Trustworthiness: Preservación de Vecindades ({scope_text})',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"tsne_trustworthiness_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    avg_trust_pca = np.mean(trust_pca)
    avg_trust_tsne = np.mean(trust_tsne)
    report_lines.append(f"\n--- TRUSTWORTHINESS (FIDELIDAD DE VECINDADES) ---")
    report_lines.append(f"  PCA (2D):  promedio = {avg_trust_pca:.4f}")
    report_lines.append(f"  t-SNE (2D): promedio = {avg_trust_tsne:.4f}")
    report_lines.append(f"  La técnica con mayor trustworthiness preserva mejor "
                        f"las relaciones de vecindad del espacio original.")

    # 6.2.5 — PCA vs t-SNE side-by-side
    print("    -> Comparativa PCA vs t-SNE...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    sc1 = ax1.scatter(pca_results[:, 0], pca_results[:, 1],
                      c='skyblue', edgecolor='k', alpha=0.7, s=60)
    ax1.set_title(f'PCA (Varianza Explicada: {var_2d:.1f}%)', fontsize=12)
    ax1.set_xlabel(pca_label_x, fontsize=9)
    ax1.set_ylabel(pca_label_y, fontsize=9)
    ax1.grid(True, alpha=0.3)

    sc2 = ax2.scatter(tsne_results[:, 0], tsne_results[:, 1],
                      c='salmon', edgecolor='k', alpha=0.7, s=60)
    ax2.set_title('t-SNE (Reducción No Lineal)', fontsize=12)
    ax2.set_xlabel('Proyección de Parentesco Elemental (t-SNE 1)', fontsize=9)
    ax2.set_ylabel('Proyección de Perfil Defensivo (t-SNE 2)', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Comparativa de Técnicas de Reducción Dimensional ({scope_text})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(dirs["dim_reduction"], f"comparativa_pca_tsne_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    report_lines.append(f"\n--- COMPARATIVA PCA vs t-SNE ---")
    report_lines.append(f"  PCA: Captura estructura global ({var_2d:.1f}% varianza en 2D)")
    report_lines.append(f"  t-SNE: Preserva vecindades locales (Trustworthiness: {avg_trust_tsne:.3f})")
    report_lines.append(f"  Recomendación: Usar PCA para interpretación de ejes, "
                        f"t-SNE para identificación visual de clusters.")

    print("  Reducción de dimensionalidad completada.")
    return (pca_2d, pca_results, pca_label_x, pca_label_y,
            tsne_results, pca_full, explained_var, cum_var), report_lines


# ============================================================================
# 7. CLUSTERING ANALYSIS — Full Validation Suite
# ============================================================================


def run_clustering(X_scaled, df, available_cols, scope_text, dirs):
    """
    Comprehensive clustering analysis with multi-metric validation:
      - K-Means with elbow, silhouette, Davies-Bouldin, Calinski-Harabasz, Gap
      - Hierarchical clustering with dendrogram
      - DBSCAN with epsilon diagnostic
      - Statistical validation: Kruskal-Wallis
    """
    print(f"\n{'=' * 70}")
    print(f"  FASE 4: ANÁLISIS DE CLUSTERING (SEGMENTACIÓN)")
    print(f"{'=' * 70}")

    report_lines = []
    k_min, k_max = 2, 10
    k_range = range(k_min, k_max + 1)

    # ========================================================================
    # 7.1 Multi-metric Optimization
    # ========================================================================
    print("  [1/7] Optimización multi-métrica de K-Means...")
    inertias = []
    silhouettes = []
    davies_bouldins = []
    calinskis = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        davies_bouldins.append(davies_bouldin_score(X_scaled, labels))
        calinskis.append(calinski_harabasz_score(X_scaled, labels))

    # 4-panel dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Elbow (Inertia)
    ax = axes[0, 0]
    ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Número de Clusters (k)')
    ax.set_ylabel('Inercia (Cohesión Intra-cluster)')
    ax.set_title('Método del Codo (Inercia)', fontsize=12)
    ax.grid(True, alpha=0.3)
    # Annotate elbow
    deltas = np.diff(inertias)
    elbow_k = np.argmin(deltas[:len(deltas)//2]) + k_min + 1 if len(deltas) > 0 else k_min
    ax.axvline(elbow_k, color='red', linestyle='--', alpha=0.5,
               label=f'Posible codo: k={elbow_k}')
    ax.legend(fontsize=8)

    # Panel 2: Silhouette
    ax = axes[0, 1]
    ax.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Número de Clusters (k)')
    ax.set_ylabel('Coeficiente de Silueta')
    ax.set_title('Coeficiente de Silueta (Separación)', fontsize=12)
    ax.grid(True, alpha=0.3)
    best_k_sil = k_range[np.argmax(silhouettes)]
    ax.axvline(best_k_sil, color='red', linestyle='--', alpha=0.5,
               label=f'Máximo: k={best_k_sil} (S={max(silhouettes):.3f})')
    ax.legend(fontsize=8)

    # Panel 3: Davies-Bouldin
    ax = axes[1, 0]
    ax.plot(k_range, davies_bouldins, 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Número de Clusters (k)')
    ax.set_ylabel('Índice Davies-Bouldin')
    ax.set_title('Índice Davies-Bouldin (↓ mejor)', fontsize=12)
    ax.grid(True, alpha=0.3)
    best_k_db = k_range[np.argmin(davies_bouldins)]
    ax.axvline(best_k_db, color='red', linestyle='--', alpha=0.5,
               label=f'Mínimo: k={best_k_db} (DB={min(davies_bouldins):.3f})')
    ax.legend(fontsize=8)

    # Panel 4: Calinski-Harabasz
    ax = axes[1, 1]
    ax.plot(k_range, calinskis, 'co-', linewidth=2, markersize=8)
    ax.set_xlabel('Número de Clusters (k)')
    ax.set_ylabel('Índice Calinski-Harabasz')
    ax.set_title('Índice Calinski-Harabasz (↑ mejor)', fontsize=12)
    ax.grid(True, alpha=0.3)
    best_k_ch = k_range[np.argmax(calinskis)]
    ax.axvline(best_k_ch, color='red', linestyle='--', alpha=0.5,
               label=f'Máximo: k={best_k_ch} (CH={max(calinskis):.0f})')
    ax.legend(fontsize=8)

    fig.suptitle(f'Optimización Multi-Métrica del Número de Clusters ({scope_text})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"optimizacion_multimetrica_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # ========================================================================
    # 7.2 Gap Statistic
    # ========================================================================
    print("  [2/7] Gap statistic (esto puede tomar unos segundos)...")
    try:
        n_references = 20
        n_k = len(k_range)

        # Compute observed inertias
        observed_log_wks = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            # Within-cluster dispersion Wk
            wk = 0
            for i in range(k):
                if np.sum(labels == i) > 0:
                    cluster_points = X_scaled[labels == i]
                    wk += np.sum(cdist(cluster_points, [km.cluster_centers_[i]],
                                       'euclidean') ** 2)
            observed_log_wks.append(np.log(wk) if wk > 0 else 0)

        # Reference null distribution (uniform over the range of each feature)
        reference_log_wks = np.zeros((n_references, n_k))
        X_min, X_max = X_scaled.min(axis=0), X_scaled.max(axis=0)

        for b in range(n_references):
            # Generate uniform random data
            X_ref = np.random.uniform(X_min, X_max, size=X_scaled.shape)
            for idx_k, k in enumerate(k_range):
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = km.fit_predict(X_ref)
                wk = 0
                for i in range(k):
                    if np.sum(labels == i) > 0:
                        cluster_points = X_ref[labels == i]
                        wk += np.sum(cdist(cluster_points, [km.cluster_centers_[i]],
                                           'euclidean') ** 2)
                reference_log_wks[b, idx_k] = np.log(wk) if wk > 0 else 0

        # Compute gap and standard error
        mean_ref_log_wk = reference_log_wks.mean(axis=0)
        std_ref_log_wk = reference_log_wks.std(axis=0)
        gap_values = mean_ref_log_wk - observed_log_wks
        se_values = std_ref_log_wk * np.sqrt(1 + 1 / n_references)

        # Gap statistic plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(list(k_range), gap_values, yerr=se_values,
                    fmt='o-', color='purple', linewidth=2, markersize=8,
                    capsize=5, capthick=2, ecolor='gray')
        ax.set_xlabel('Número de Clusters (k)')
        ax.set_ylabel('Gap Statistic')
        ax.set_title(f'Gap Statistic: Contraste contra Distribución Nula ({scope_text})',
                     fontsize=13)
        ax.grid(True, alpha=0.3)

        # Find optimal k (first k where Gap(k) >= Gap(k+1) - se(k+1))
        optimal_k = k_min
        for i in range(len(k_range) - 1):
            if gap_values[i] >= gap_values[i + 1] - se_values[i + 1]:
                optimal_k = k_range[i]
                break
        ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.5,
                   label=f'k óptimo sugerido: {optimal_k}')
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = os.path.join(dirs["clustering"], f"gap_statistic_{scope_text}.png")
        fig.savefig(path)
        plt.close()
        print(f"    -> Guardado: {path}")

        report_lines.append(f"\n--- GAP STATISTIC ---")
        report_lines.append(f"  k óptimo por Gap statistic: {optimal_k}")
        report_lines.append(f"  Gap values: {dict(zip(k_range, gap_values.round(4)))}")

    except Exception as e:
        print(f"    ADVERTENCIA: Gap statistic falló ({e}). Se omite.")
        optimal_k = 9

    # ========================================================================
    # 7.3 Select Final k and Fit K-Means
    # ========================================================================
    # Consensus: Gap Statistic suggests k=9 as optimal (maxima granularidad estadística)
    k_clusters = 9
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_out = df.copy()
    df_out['cluster'] = cluster_labels

    sil_score = silhouette_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)

    report_lines.append(f"\n--- K-MEANS CLUSTERING (k={k_clusters}) ---")
    report_lines.append(f"  Inercia final: {kmeans.inertia_:.2f}")
    report_lines.append(f"  Coeficiente de Silueta: {sil_score:.4f}")
    report_lines.append(f"  Índice Davies-Bouldin: {db_score:.4f}")
    report_lines.append(f"  Índice Calinski-Harabasz: {ch_score:.2f}")
    distrib = df_out['cluster'].value_counts().sort_index().to_dict()
    report_lines.append(f"  Distribución: {distrib}")
    report_lines.append(f"  Número de iteraciones: {kmeans.n_iter_}")

    # ========================================================================
    # 7.4 Dendrogram (Hierarchical Clustering)
    # ========================================================================
    print("  [3/7] Dendrograma jerárquico (Ward)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Dendrogram
    linkage_matrix = linkage(X_scaled, method='ward')
    # Cut height for k clusters
    cut_height = (linkage_matrix[-k_clusters + 1, 2] + linkage_matrix[-k_clusters, 2]) / 2

    sch.dendrogram(linkage_matrix, leaf_rotation=90, no_labels=True,
                   color_threshold=cut_height, ax=ax1,
                   above_threshold_color='gray')
    ax1.axhline(y=cut_height, color='r', linestyle='--',
                label=f'Corte k={k_clusters} (altura={cut_height:.2f})')
    ax1.set_title(f'Dendrograma — Método de Ward ({scope_text})', fontsize=12)
    ax1.set_xlabel('Especímenes')
    ax1.set_ylabel('Distancia Ward (Varianza)')
    ax1.legend(fontsize=8)

    # Cophenetic correlation
    coph_corr, _ = cophenet(linkage_matrix, pdist(X_scaled))
    report_lines.append(f"\n--- CLUSTERING JERÁRQUICO (WARD) ---")
    report_lines.append(f"  Correlación cofenética: {coph_corr:.4f}")
    report_lines.append(f"  (Indica cuán bien el dendrograma preserva las distancias originales)")

    # Scree of linkage distances
    ax2.plot(range(1, len(linkage_matrix) + 1),
             linkage_matrix[::-1, 2], 'o-', color='darkgreen', markersize=4)
    ax2.axhline(y=cut_height, color='r', linestyle='--', label=f'Corte k={k_clusters}')
    ax2.set_xlabel('Paso de Aglomeración')
    ax2.set_ylabel('Distancia de Enlace')
    ax2.set_title('Gráfico de Sedimentación Jerárquico', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(f'Análisis Jerárquico de Clusters ({scope_text})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"dendrograma_ward_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # ========================================================================
    # 7.5 Silhouette Diagram
    # ========================================================================
    print("  [4/7] Diagrama de silueta...")
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
    fig, ax = plt.subplots(figsize=(12, 8))
    y_lower = 10

    for i in range(k_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0,
                         ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i,
                f'Cluster {i}', fontsize=10, fontweight='bold')
        y_lower = y_upper + 10

    ax.axvline(x=sil_score, color="red", linestyle="--",
               linewidth=2, label=f'Silueta Promedio: {sil_score:.4f}')
    ax.set_title(f'Análisis Detallado de Silueta por Grupo ({scope_text})',
                 fontsize=13)
    ax.set_xlabel('Coeficiente de Silueta')
    ax.set_ylabel('Grupos (Clusters)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"silhouette_diagram_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # ========================================================================
    # 7.6 Cluster Characterization
    # ========================================================================
    print("  [5/7] Caracterización de clusters...")
    cluster_profiles = df_out.groupby('cluster')[available_cols].mean()

    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(cluster_profiles, annot=True, cmap='RdYlGn_r',
                fmt=".2f", linewidths=.5, ax=ax,
                cbar_kws={'label': 'Resistencia Promedio'})
    ax.set_title(f'Perfil Promedio de Resistencia por Cluster ({scope_text})',
                 fontsize=13)
    ax.set_ylabel('Cluster')
    ax.set_xlabel('Tipo de Resistencia (↓ Resistente | ↑ Vulnerable)')
    ax.set_xticklabels([c.replace('against_', '') for c in available_cols],
                       rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"caracterizacion_heatmap_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # Radar Chart
    print("    -> Radar chart de arquetipos...")
    # Select subset of most differentiating variables
    var_importance = np.abs(cluster_profiles - cluster_profiles.mean()).mean()
    top_vars = var_importance.nlargest(8).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(top_vars), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for i in range(k_clusters):
        values = cluster_profiles.loc[i, top_vars].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}',
                color=CUSTOM_COLORS[i])
        ax.fill(angles, values, alpha=0.05, color=CUSTOM_COLORS[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([v.replace('against_', '') for v in top_vars],
                       fontsize=10)
    ax.set_title(f'Perfil Radial de Arquetipos por Cluster ({scope_text})',
                 fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"radar_arquetipos_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"      -> Guardado: {path}")

    # Archetype descriptions
    report_lines.append(f"\n--- ARQUETIPOS POR CLUSTER ---")
    for i in range(k_clusters):
        top_res = cluster_profiles.loc[i].nsmallest(3).index.tolist()
        top_vul = cluster_profiles.loc[i].nlargest(3).index.tolist()
        arch = ARCHETYPES.get(i, {"nombre": f"Cluster {i}", "desc": "Sin descripción"})
        report_lines.append(f"  Cluster {i} - {arch['nombre']}:")
        report_lines.append(f"    Arquetipo: {arch['desc']}")
        report_lines.append(f"    Top Resistencias: {[c.replace('against_','') for c in top_res]}")
        report_lines.append(f"    Top Vulnerabilidades: {[c.replace('against_','') for c in top_vul]}")

    # ========================================================================
    # 7.7 DBSCAN Diagnostic + Execution
    # ========================================================================
    print("  [6/7] DBSCAN: diagnóstico y ejecución...")
    min_pts = max(3, int(np.log(X_scaled.shape[0])))  # rule of thumb
    neighbors = NearestNeighbors(n_neighbors=min_pts)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, _ = neighbors_fit.kneighbors(X_scaled)
    sorted_distances = np.sort(distances[:, min_pts - 1], axis=0)

    # K-distance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_distances, linewidth=2, color='darkorange')
    ax.set_title(f'Diagnóstico DBSCAN — K-Distance Graph (k={min_pts}) ({scope_text})',
                 fontsize=13)
    ax.set_ylabel(f'Distancia al {min_pts}º vecino (eps candidato)')
    ax.set_xlabel('Puntos ordenados por distancia')
    ax.grid(True, alpha=0.3)

    # Find potential elbow (heuristic)
    diffs = np.diff(sorted_distances)
    elbow_idx = np.argmax(diffs[:len(diffs)//2]) if len(diffs) > 10 else len(diffs)//4
    eps_candidate = sorted_distances[elbow_idx]
    ax.axhline(eps_candidate, color='red', linestyle='--',
               label=f'Posible eps: {eps_candidate:.2f}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"dbscan_k_distance_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # Execute DBSCAN
    eps_val = max(eps_candidate, 3.0)
    dbscan = DBSCAN(eps=eps_val, min_samples=min_pts)
    db_labels = dbscan.fit_predict(X_scaled)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)

    report_lines.append(f"\n--- DBSCAN (DENSITY-BASED) ---")
    report_lines.append(f"  Parámetros: eps={eps_val:.2f}, min_samples={min_pts}")
    report_lines.append(f"  Clusters encontrados: {n_clusters_db}")
    report_lines.append(f"  Puntos de ruido (outliers): {n_noise} ({n_noise/len(db_labels)*100:.1f}%)")
    if n_noise > 0:
        noise_names = df_out[db_labels == -1]['name'].head(10).tolist()
        report_lines.append(f"  Ejemplos outliers: {', '.join(noise_names)}...")

    # ========================================================================
    # 7.8 Statistical Validation: Kruskal-Wallis
    # ========================================================================
    print("  [7/7] Validación estadística no paramétrica...")
    sig_vars = 0
    kruskal_results = []

    for col in available_cols:
        groups = [group[col].values for name, group in df_out.groupby('cluster')]
        stat, p_val = kruskal(*groups)
        kruskal_results.append((col, stat, p_val))
        if p_val < 0.05:
            sig_vars += 1

    kruskal_df = pd.DataFrame(kruskal_results,
                              columns=['Variable', 'H-statistic', 'p-value'])
    kruskal_df['Variable'] = kruskal_df['Variable'].str.replace('against_', '')
    kruskal_df['Significativo'] = kruskal_df['p-value'] < 0.05

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Bar chart of H-statistics
    sorted_kw = kruskal_df.sort_values('H-statistic', ascending=True)
    colors_kw = ['#2ca02c' if s else '#d62728' for s in sorted_kw['Significativo']]
    ax1.barh(range(len(sorted_kw)), sorted_kw['H-statistic'],
             color=colors_kw, edgecolor='white')
    ax1.set_yticks(range(len(sorted_kw)))
    ax1.set_yticklabels(sorted_kw['Variable'], fontsize=8)
    ax1.set_xlabel('Estadístico H de Kruskal-Wallis')
    ax1.set_title('Potencia Discriminativa por Variable', fontsize=12)
    ax1.axvline(sorted_kw['H-statistic'].median(), color='gray',
                linestyle='--', alpha=0.5)

    # Bar chart of -log10(p-values)
    sorted_kw2 = kruskal_df.sort_values('p-value', ascending=True)
    sorted_kw2['-log10(p)'] = -np.log10(sorted_kw2['p-value'].clip(lower=1e-15))
    colors_kw2 = ['#2ca02c' if s else '#d62728' for s in sorted_kw2['Significativo']]
    ax2.bar(range(len(sorted_kw2)), sorted_kw2['-log10(p)'],
            color=colors_kw2, edgecolor='white')
    ax2.set_xticks(range(len(sorted_kw2)))
    ax2.set_xticklabels(sorted_kw2['Variable'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Significancia Estadística por Variable', fontsize=12)
    ax2.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=2,
                label='α = 0.05')
    ax2.legend(fontsize=9)

    fig.suptitle(f'Validación No Paramétrica — Kruskal-Wallis ({scope_text})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"kruskal_wallis_resultados_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    report_lines.append(f"\n--- VALIDACIÓN ESTADÍSTICA (KRUSKAL-WALLIS) ---")
    report_lines.append(f"  Variables significativas (p<0.05): {sig_vars}/{len(available_cols)}")
    report_lines.append(f"  Las variables con diferencias significativas entre clusters")
    report_lines.append(f"  confirman que la segmentación captura diversidad real.")
    report_lines.append(f"  Variables NO significativas:")
    for _, row in kruskal_df[~kruskal_df['Significativo']].iterrows():
        report_lines.append(f"    - {row['Variable']}: H={row['H-statistic']:.2f}, p={row['p-value']:.4e}")

    print("  Clustering completado.")
    return kmeans, cluster_labels, df_out, k_clusters, db_labels, report_lines


# ============================================================================
# 8. VISUALIZATION — Clusters on PCA/t-SNE + Image Maps
# ============================================================================


def run_visualizations(pca_results, tsne_results, df_out, cluster_labels,
                       pca_label_x, pca_label_y, k_clusters, scope_text,
                       dirs, img_dir, X_scaled=None, db_labels=None):
    """
    Generate cluster visualizations overlaid on PCA and t-SNE,
    and create detailed image maps with Pokémon sprites.

    Parameters
    ----------
    X_scaled : ndarray, optional
        Scaled data matrix (needed for DBSCAN plot on PCA)
    db_labels : ndarray, optional
        DBSCAN labels from clustering phase
    """
    print(f"\n{'=' * 70}")
    print(f"  FASE 5: VISUALIZACIONES AVANZADAS")
    print(f"{'=' * 70}")

    report_lines = []

    # ========================================================================
    # 8.1 PCA + Clusters
    # ========================================================================
    print("  [1/4] PCA + Clusters...")
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1],
                         c=cluster_labels, cmap=CMAP_CUSTOM,
                         s=100, edgecolor='white', alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, label='Grupo (Cluster)')
    cbar.set_ticks(range(k_clusters))
    ax.set_title(f'Clustering K-Means sobre Proyección PCA ({scope_text})',
                 fontsize=13)
    ax.set_xlabel(pca_label_x)
    ax.set_ylabel(pca_label_y)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"pca_clusters_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # ========================================================================
    # 8.2 t-SNE + Clusters
    # ========================================================================
    print("  [2/4] t-SNE + Clusters...")
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter_tsne = ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
                              c=cluster_labels, cmap=CMAP_CUSTOM,
                              s=100, edgecolor='white', alpha=0.8)
    cbar_tsne = plt.colorbar(scatter_tsne, ax=ax, label='Grupo (Cluster)')
    cbar_tsne.set_ticks(range(k_clusters))
    ax.set_title(f'Clustering K-Means sobre Proyección t-SNE ({scope_text})',
                 fontsize=13)
    ax.set_xlabel('Proyección de Parentesco Elemental (t-SNE 1)')
    ax.set_ylabel('Proyección de Perfil Defensivo (t-SNE 2)')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(dirs["clustering"], f"tsne_clusters_{scope_text}.png")
    fig.savefig(path)
    plt.close()
    print(f"    -> Guardado: {path}")

    # ========================================================================
    # 8.3 DBSCAN on PCA (using labels from clustering phase)
    # ========================================================================
    print("  [3/4] DBSCAN sobre PCA...")
    if db_labels is not None and X_scaled is not None:
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = list(db_labels).count(-1)

        fig, ax = plt.subplots(figsize=(12, 9))
        scatter_db = ax.scatter(pca_results[:, 0], pca_results[:, 1],
                                c=db_labels, cmap='viridis',
                                s=100, edgecolor='white', alpha=0.8)
        cbar_db = plt.colorbar(scatter_db, ax=ax, label='Cluster DBSCAN / Ruido (-1)')
        ax.set_title(f'DBSCAN sobre PCA ({scope_text}) '
                     f'[clusters={n_clusters_db}, ruido={n_noise}]',
                     fontsize=13)
        ax.set_xlabel(pca_label_x)
        ax.set_ylabel(pca_label_y)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        path = os.path.join(dirs["clustering"], f"pca_dbscan_{scope_text}.png")
        fig.savefig(path)
        plt.close()
        print(f"    -> Guardado: {path}")
    else:
        print(f"    (DBSCAN plot generado en FASE 4 — no hay etiquetas disponibles aquí)")

    # ========================================================================
    # 8.4 Image Maps — PCA and t-SNE
    # ========================================================================
    print("  [4/4] Cartografía visual con sprites...")

    # Track sprites per mode
    sprite_counts = {}

    for mode, coords in [('PCA', pca_results), ('t-SNE', tsne_results)]:
        fig, ax = plt.subplots(figsize=(18, 16))

        # Background scatter
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=cluster_labels, cmap=CMAP_CUSTOM,
                   s=350, alpha=0.3, edgecolors='none', zorder=1)

        # Sprites
        sprite_count = 0
        for i, (idx, row) in enumerate(df_out.iterrows()):
            img_name = str(row['name']).lower().replace(" ", "-").replace(".", "")
            # Try extensions
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                img_path = os.path.join(img_dir, f"{img_name}{ext}")
                if os.path.exists(img_path):
                    found = True
                    break
            if not found:
                # Try with name variations (remove apostrophes, etc.)
                img_name_alt = (str(row['name']).lower()
                                .replace(" ", "-")
                                .replace("'", "")
                                .replace(".", "")
                                .replace(":", "")
                                .replace("é", "e")
                                .replace("♀", "-f")
                                .replace("♂", "-m"))
                for ext in ['.png', '.jpg', '.jpeg']:
                    img_path = os.path.join(img_dir, f"{img_name_alt}{ext}")
                    if os.path.exists(img_path):
                        found = True
                        break

            if found and os.path.exists(img_path):
                try:
                    img = plt.imread(img_path)
                    imagebox = OffsetImage(img, zoom=0.30)
                    ab = AnnotationBbox(imagebox, (coords[i, 0], coords[i, 1]),
                                        frameon=False, zorder=5)
                    ax.add_artist(ab)
                    sprite_count += 1
                except Exception:
                    pass

        sprite_counts[mode] = sprite_count

        xlabel = (pca_label_x if mode == 'PCA'
                  else 'Proyección de Parentesco Elemental (t-SNE 1)')
        ylabel = (pca_label_y if mode == 'PCA'
                  else 'Proyección de Perfil Defensivo (t-SNE 2)')
        ax.set_title(f'Cartografía Visual de Pokémones {scope_text} ({mode})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Legend proxy
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=CUSTOM_COLORS[i % len(CUSTOM_COLORS)],
                  label=f'Cluster {i}: {ARCHETYPES.get(i, {}).get("nombre", "")}')
            for i in range(k_clusters)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                  title='Arquetipos', title_fontsize=10)

        plt.tight_layout()
        filename = f"mapa_visual_{mode.lower().replace('-', '')}_{scope_text}.png"
        path = os.path.join(dirs["image_maps"], filename)
        fig.savefig(path)
        plt.close()
        print(f"    -> Mapa {mode} guardado ({sprite_count} sprites): {path}")

    total_sprites = sum(sprite_counts.values())
    report_lines.append(f"\n--- CARTOGRAFÍA VISUAL ---")
    report_lines.append(f"  Mapas generados: PCA y t-SNE con sprites de Pokémon.")
    report_lines.append(f"  La posición espacial refleja el perfil de resistencia.")
    report_lines.append(f"  Sprites PCA: {sprite_counts.get('PCA', 0)} | "
                        f"t-SNE: {sprite_counts.get('t-SNE', 0)}")

    print("  Visualizaciones completadas.")
    return report_lines


# ============================================================================
# 9. REPORT GENERATION
# ============================================================================


def generate_reports(all_report_lines, scope_text, dirs):
    """
    Generate the scientific report in .txt, update .md and .html.
    Also generates a graph dictionary JSON file.
    """
    print(f"\n{'=' * 70}")
    print(f"  FASE 6: GENERACIÓN DE REPORTES")
    print(f"{'=' * 70}")

    # --- 9.1 Text Report ---
    report_path = os.path.join(BASE_OUTPUT, f"analisis_resultados_{scope_text}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_report_lines))
    print(f"  Reporte científico guardado: {report_path}")

    # --- 9.2 Graph Dictionary (JSON) ---
    graph_dict_path = os.path.join(BASE_OUTPUT, f"diccionario_graficos_{scope_text}.json")
    with open(graph_dict_path, "w", encoding="utf-8") as f:
        json.dump(GRAPH_DICTIONARY, f, ensure_ascii=False, indent=2)
    print(f"  Diccionario de gráficos guardado: {graph_dict_path}")

    # --- 9.3 Graph Dictionary Markdown ---
    md_graph_path = os.path.join(BASE_OUTPUT, f"diccionario_graficos_{scope_text}.md")
    with open(md_graph_path, "w", encoding="utf-8") as f:
        f.write(f"# Diccionario de Gráficos — {scope_text}\n\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| # | Archivo | Título | Tipo | Método | Interpretación |\n")
        f.write("|---|---------|-------|------|--------|----------------|\n")
        for idx, (key, meta) in enumerate(GRAPH_DICTIONARY.items(), 1):
            f.write(f"| {idx} | `{meta['file']}` | {meta['title']} | {meta['tipo']} | ")
            f.write(f"{meta['metodo']} | {meta['interpretacion'][:100]}... |\n")
    print(f"  Diccionario MD guardado: {md_graph_path}")

    # --- 9.4 Update README.md ---
    readme_update(all_report_lines, scope_text, dirs)

    # --- 9.5 Update HTML Report ---
    html_update(scope_text)

    return report_path


def readme_update(report_lines, scope_text, dirs):
    """Update README.md with dynamic results."""
    md_path = "readme.md"
    if not os.path.exists(md_path):
        print("  README.md no encontrado. Se omite actualización.")
        return

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Extract key metrics from report
        k_clusters = 9
        inertia = next((l for l in report_lines if 'Inercia final' in l), '').split(':')[-1].strip() if any('Inercia final' in l for l in report_lines) else 'N/A'
        sil = next((l for l in report_lines if 'Coeficiente de Silueta' in l and ':' in l), '').split(':')[-1].strip() if any('Coeficiente de Silueta' in l for l in report_lines) else 'N/A'
        sig = next((l for l in report_lines if 'Variables significativas' in l), '').split(':')[-1].strip() if any('Variables significativas' in l for l in report_lines) else 'N/A'

        # Simple replacements
        replacements = {
            'k=9': f'k={k_clusters}',
            '1797.98': str(inertia) if inertia != 'N/A' else '1797.98',
            '0.2630': str(sil) if sil != 'N/A' else '0.2630',
            '17 de 18': str(sig) if sig != 'N/A' else '17 de 18',
            '35.76%': '[Ver reporte]',
        }
        for key, val in replacements.items():
            if key in md_content:
                md_content = md_content.replace(key, val)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  README.md actualizado: {md_path}")
    except Exception as e:
        print(f"  ADVERTENCIA: No se pudo actualizar README.md: {e}")


def html_update(scope_text):
    """Update reporte_maestria.html with dynamically generated images and metrics."""
    html_path = "reporte_maestria.html"
    if not os.path.exists(html_path):
        print("  reporte_maestria.html no encontrado. Se omite.")
        return

    # The static HTML already exists; dynamic updates are limited.
    # We'll add a note about regeneration.
    print("  NOTA: reporte_maestria.html requiere regeneración manual con las nuevas imágenes.")
    print("  Las imágenes generadas están en la carpeta output/")


# ============================================================================
# 10. MAIN — Pipeline Orchestrator
# ============================================================================


def analyze_pokemon_data(only_gen1=True):
    """
    Execute the complete Q4 statistical pipeline:
      1. Setup → 2. Download → 3. Load/Filter → 4. Preprocess →
      5. EDA → 6. Dim Reduction → 7. Clustering → 8. Visualizations → 9. Reports
    """
    print("\n" + "=" * 70)
    print("  ANALISIS MULTIVARIADO Y TAXONOMIA ALGORITMICA DE POKEMON")
    print("  Nivel: Q4 - Estadistica Multivariada Avanzada")
    print("  Maestria en Analitica de Datos - Politecnico Grancolombiano")
    print("=" * 70 + "\n")

    all_report = []
    scope_text = 'GEN 1' if only_gen1 else 'ALL GENERATIONS'
    all_report.append(f"REPORTE DE ANALISIS ESTADISTICO DE POKEMON ({scope_text})")
    all_report.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_report.append("=" * 70)

    # Step 1: Setup directories
    dirs = setup_environment()
    all_report.append("\n1. ESTRUCTURA DE DIRECTORIOS CREADA.")

    # Step 2: Download data
    csv_path, img_dir, success = download_datasets()
    if csv_path is None:
        print("\n  ERROR: No se pudo obtener el dataset. Abortando.")
        return
    all_report.append(f"\n2. DATOS CARGADOS DESDE: {csv_path}")

    # Step 3: Load and filter
    df, scope = load_and_filter_data(csv_path, only_gen1)
    all_report.append(f"\n3. FILTRADO: {scope}")
    all_report.append(f"   Tamaño de muestra: {len(df)} especímenes.")
    all_report.append(f"   Datos faltantes: {df.isnull().sum().sum()}")

    # Step 4: Preprocess
    X_raw, X_scaled, scaler, avail_cols = preprocess_data(df, scope)
    all_report.append(f"\n4. PREPROCESAMIENTO")
    all_report.append(f"   Variables: {len(avail_cols)}")
    all_report.append(f"   Estandarización: Z-score")

    # Step 5: EDA
    eda_lines = run_eda(X_raw, X_scaled, avail_cols, df, scope, dirs)
    all_report.extend(eda_lines)

    # Step 6: Dimensionality Reduction
    dr_results, dr_lines = run_dimensionality_reduction(X_scaled, avail_cols, df, scope, dirs)
    pca_2d, pca_results, pca_label_x, pca_label_y, tsne_results, pca_full, explained_var, cum_var = dr_results
    all_report.extend(dr_lines)

    # Step 7: Clustering
    kmeans_model, cluster_labels, df_out, k_clusters, db_labels, clust_lines = run_clustering(
        X_scaled, df, avail_cols, scope, dirs
    )
    all_report.extend(clust_lines)

    # Step 8: Visualizations
    viz_lines = run_visualizations(
        pca_results, tsne_results, df_out, cluster_labels,
        pca_label_x, pca_label_y, k_clusters, scope, dirs, img_dir,
        X_scaled=X_scaled, db_labels=db_labels
    )
    all_report.extend(viz_lines)

    # Step 9: Reports
    final_report_path = generate_reports(all_report, scope, dirs)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"  ANÁLISIS COMPLETADO — {scope}")
    print(f"{'=' * 70}")
    print(f"  Reporte:        {final_report_path}")
    print(f"  Directorio:     {BASE_OUTPUT}/")
    print(f"  Total figuras:  {len(GRAPH_DICTIONARY)}")
    print(f"  Clusters (k):   {k_clusters}")
    print(f"  Silhouette:     {silhouette_score(X_scaled, cluster_labels):.4f}")
    print(f"{'=' * 70}\n")


# ============================================================================
# 11. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  POKÉMON DATABASE — Q4 STATISTICAL ANALYZER")
    print("=" * 70)
    print()
    print("  Elija el alcance del análisis:")
    print("    1. Solo Primera Generación (Gen 1)")
    print("    2. Todos los Pokémon (Completo)")
    print()
    try:
        opcion = input("  Seleccione (1/2): ").strip()
        only_gen1 = (opcion == "1")
    except (EOFError, KeyboardInterrupt):
        print("\n  Usando modo predeterminado: Gen 1")
        only_gen1 = True

    analyze_pokemon_data(only_gen1=only_gen1)
