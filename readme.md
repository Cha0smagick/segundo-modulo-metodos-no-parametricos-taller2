# Análisis Multivariado y Taxonomía Algorítmica de Pokémon (Gen 1)
## Nivel Q4 — Estadística Multivariada Avanzada

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![Maestría](https://img.shields.io/badge/Maestr%C3%ADa-Anal%C3%ADtica%20de%20Datos-green)](https://www.poli.edu.co)

Este proyecto implementa un **pipeline completo de análisis multivariado** a nivel de maestría (Q4) para segmentar y clasificar Pokémon de la Primera Generación basándose en sus perfiles de resistencia elemental (18 variables). Se emplean técnicas de **clustering multi-método**, **reducción de dimensionalidad con validación**, y un riguroso marco de **validación estadística no paramétrica**.

---

## 📋 Tabla de Contenidos

- [Objetivo del Proyecto](#objetivo-del-proyecto)
- [Pipeline Analítico](#pipeline-analítico)
- [Resultados Clave](#resultados-clave)
- [Diccionario de Gráficos (25 figuras)](#diccionario-de-gráficos)
- [Tecnologías](#tecnologías)
- [Instalación y Uso](#instalación-y-uso)
- [Estructura de Salida](#estructura-de-salida)
- [Interpretación Científica](#interpretación-científica)
- [Autor y Licencia](#autor-y-licencia)

---

## 🎯 Objetivo del Proyecto

Transformar un conjunto de datos multidimensional (18 variables de resistencia elemental × 151 especímenes) en una **estructura de conocimiento organizada** mediante:

1. **Segmentación algorítmica** con validación multi-métrica (K-Means, Jerárquico, DBSCAN)
2. **Reducción de dimensionalidad** con evaluación de calidad (PCA + t-SNE)
3. **Validación estadística inferencial** (Kruskal-Wallis, D'Agostino-Pearson)
4. **Cartografía visual** con sprites para interpretación intuitiva

---

## 🚀 Pipeline Analítico

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE 0: Adquisición           kagglehub → pokemon.csv + images │
├─────────────────────────────────────────────────────────────────┤
│  FASE 1: Preprocesamiento      Filtrado Gen 1 → StandardScaler  │
├─────────────────────────────────────────────────────────────────┤
│  FASE 2: EDA                   Histogramas, Boxplots,           │
│                                Correlaciones, Coord. Paralelas   │
├─────────────────────────────────────────────────────────────────┤
│  FASE 3: Reducción Dimensional  PCA (scree, círculo, biplot,    │
│                                 cos², loadings)                  │
│                                 t-SNE (perplejidad, learning     │
│                                 rate, trustworthiness)           │
├─────────────────────────────────────────────────────────────────┤
│  FASE 4: Clustering             K-Means multi-métrica, Gap,     │
│                                 Dendrograma (Ward), DBSCAN,      │
│                                 Kruskal-Wallis                  │
├─────────────────────────────────────────────────────────────────┤
│  FASE 5: Visualizaciones        PCA/t-SNE + clusters,           │
│                                 Mapas con sprites               │
├─────────────────────────────────────────────────────────────────┤
│  FASE 6: Reportes               .txt, .md, .html, .json         │
└─────────────────────────────────────────────────────────────────┘
```

### Etapas Detalladas

#### Fase 1 — Preprocesamiento
- **Filtrado**: 151 Pokémon de la Generación 1
- **Variables**: 18 resistencias elementales (`against_bug` a `against_water`)
- **Estandarización**: Z-score (media=0, var=1) para evitar sesgos por escala
- **Datos faltantes**: 119 celdas con valores ausentes (principalmente en `against_fairy` y `against_steel` para Gen 1)

#### Fase 2 — Análisis Exploratorio (EDA)
- **Distribuciones**: Solo 1/18 variables (`against_ghost`) pasa la prueba de normalidad D'Agostino-Pearson (p>0.05), justificando el uso de métodos **no paramétricos**
- **Correlaciones**: Estructura de dependencia entre tipos elementales revela co-ocurrencias naturales
- **Asimetría**: Variables como `against_normal` (skew=-3.25) y `against_bug` (skew=2.59) muestran alta asimetría

#### Fase 3 — Reducción de Dimensionalidad

**PCA (Análisis de Componentes Principales):**
| Componente | Varianza Explicada | Acumulada |
|-----------|-------------------|-----------|
| PC1 | 21.50% | 21.50% |
| PC2 | 14.26% | [Ver reporte] |
| PC3 | 9.87% | 45.63% |
| PC4 | 8.42% | 54.05% |
| PC5 | 7.29% | 61.34% |
| PC6 | 6.48% | 67.82% |
| PC7 | 5.77% | **73.59%** |

- Se requieren **7 componentes** para explicar >73% de la varianza
- **Círculo de correlaciones**: Visualiza las relaciones entre variables originales y los componentes
- **cos²**: Ninguna variable supera 0.7 de calidad de representación en 2D, indicando que 2 dimensiones capturan estructura global pero no detalles finos

**t-SNE:**
- **Trustworthiness promedio**: 0.900 (vs PCA: 0.872) → t-SNE preserva mejor las vecindades locales
- **Barrido de perplejidad**: Valores entre 15-50 producen las proyecciones más estables
- **Learning rate**: 200-500 ofrecen el mejor balance convergencia-calidad

#### Fase 4 — Clustering

**K-Means (k=4):**
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Inercia | 1797.98 | Compacidad intra-cluster |
| Silhouette | 0.2630 | Separación moderada |
| Davies-Bouldin | 1.4576 | Menor es mejor (solapamiento bajo) |
| Calinski-Harabasz | 25.07 | Mayor es mejor |

**Distribución de clusters:**
| Cluster | Tamaño | Arquetipo |
|---------|--------|-----------|
| 0 | 81 | Guardianes del Océano y el Acero |
| 1 | 49 | Protectores de la Biosfera |
| 2 | 15 | Místicos del Éter |
| 3 | 6 | Centinelas de Alta Tensión |

**Gap Statistic**: Sugiere k=9 como óptimo estadístico (contraste con distribución nula)

**DBSCAN**: Detecta 12 clusters densos, con 28 outliers (18.5%) — Pokémon con perfiles de resistencia únicos (Charizard, Gengar, Magneton, etc.)

**Validación Kruskal-Wallis**: 17/18 variables muestran diferencias significativas (p<0.05) entre clusters. Solo `against_dragon` no discrimina (p=0.43), consistente con la rareza del tipo dragón en Gen 1.

---

## 📊 Resultados Clave

### Métricas de Segmentación
| Indicador | Valor | Evaluación |
|-----------|-------|------------|
| **Silhouette Score** | 0.2630 | Separación moderada (estructura natural con solapamiento) |
| **Davies-Bouldin Index** | 1.4576 | Bajo solapamiento inter-cluster |
| **Calinski-Harabasz** | 25.07 | Discriminación significativa |
| **Correlación cofenética** | 0.6489 | Consistencia jerárquica aceptable |
| **Trustworthiness t-SNE** | 0.9003 | Excelente preservación de vecindades |
| **Trustworthiness PCA** | 0.8724 | Buena preservación de estructura global |

### Hallazgos Clave
1. **Las resistencias elementales discriminan naturalmente** a los Pokémon en 4 arquetipos defensivos
2. **El tipo dragón no es discriminativo** en Gen 1 (solo 3 Pokémon: Dragonite, Dragonair, Dratini)
3. **DBSCAN identifica especialistas puros**: 18.5% de Pokémon tienen perfiles atípicos
4. **t-SNE supera a PCA** en fidelidad local (trustworthiness 0.900 vs 0.872)
5. **Solo 1/18 variables es normal**: justifica plenamente el enfoque no paramétrico

---

## 📈 Diccionario de Gráficos

El pipeline genera **25 figuras** en 4 categorías. A continuación se listan con su interpretación metodológica:

### 📊 Descriptivos (4 figuras)

| # | Archivo | Descripción | Método |
|---|---------|-------------|--------|
| 1 | `output/descriptive/histogram_resistencias_GEN 1.png` | Histogramas con KDE de las 18 variables | `matplotlib.hist` + `seaborn.kdeplot` |
| 2 | `output/descriptive/boxplot_resistencias_GEN 1.png` | Boxplots de variables estandarizadas | `seaborn.boxplot` |
| 3 | `output/descriptive/heatmap_correlaciones_GEN 1.png` | Mapa de calor de correlaciones de Pearson | `seaborn.heatmap` |
| 4 | `output/descriptive/coordenadas_paralelas_GEN 1.png` | Coordenadas paralelas de perfiles de resistencia | `pandas.plotting.parallel_coordinates` |

### 📉 Reducción de Dimensionalidad (8 figuras)

| # | Archivo | Descripción | Método |
|---|---------|-------------|--------|
| 5 | `output/dimensionality_reduction/scree_plot_pca_GEN 1.png` | Scree plot: varianza explicada + acumulada | `PCA.explained_variance_ratio_` |
| 6 | `output/dimensionality_reduction/loadings_heatmap_pca_GEN 1.png` | Heatmap de loadings (contribuciones) | `PCA.components_` |
| 7 | `output/dimensionality_reduction/circulo_correlaciones_pca_GEN 1.png` | Círculo de correlaciones en PC1-PC2 | Loadings escalados por √(eigenvalues) |
| 8 | `output/dimensionality_reduction/biplot_pca_GEN 1.png` | PCA-Biplot: individuos + variables | PCA con escalado symbiplot |
| 9 | `output/dimensionality_reduction/calidad_representacion_cos2_GEN 1.png` | Calidad cos² por variable | `cos² = loading²` |
| 10 | `output/dimensionality_reduction/tsne_perplejidad_grid_GEN 1.png` | Grid de perplejidad (5,10,15,30,50,100) | `TSNE` |
| 11 | `output/dimensionality_reduction/tsne_learning_rate_grid_GEN 1.png` | Grid de learning rate (10,100,200,500,1000,2000) | `TSNE` |
| 12 | `output/dimensionality_reduction/tsne_trustworthiness_GEN 1.png` | Trustworthiness PCA vs t-SNE | `sklearn.manifold.trustworthiness` |

### 🧬 Clustering (9 figuras)

| # | Archivo | Descripción | Método |
|---|---------|-------------|--------|
| 13 | `output/clustering/optimizacion_multimetrica_GEN 1.png` | Dashboard 4-métricas para selección de k | Inercia, Silueta, DB, CH |
| 14 | `output/clustering/gap_statistic_GEN 1.png` | Gap statistic con barras de error | Bootstrap B=20 |
| 15 | `output/clustering/dendrograma_ward_GEN 1.png` | Dendrograma Ward + sedimentación jerárquica | `scipy.cluster.hierarchy` |
| 16 | `output/clustering/silhouette_diagram_GEN 1.png` | Diagrama de silueta detallado por cluster | `silhouette_samples` |
| 17 | `output/clustering/caracterizacion_heatmap_GEN 1.png` | Perfil promedio de resistencia por cluster | `seaborn.heatmap` |
| 18 | `output/clustering/radar_arquetipos_GEN 1.png` | Radar chart de los 4 arquetipos | Coordenadas polares |
| 19 | `output/clustering/pca_clusters_GEN 1.png` | Clusters K-Means sobre PCA | `PCA` + `KMeans` |
| 20 | `output/clustering/tsne_clusters_GEN 1.png` | Clusters K-Means sobre t-SNE | `TSNE` + `KMeans` |
| 21 | `output/clustering/dbscan_k_distance_GEN 1.png` | Diagnóstico DBSCAN: k-distance | `NearestNeighbors` |
| 22 | `output/clustering/pca_dbscan_GEN 1.png` | DBSCAN sobre PCA | `DBSCAN` + `PCA` |
| 23 | `output/clustering/kruskal_wallis_resultados_GEN 1.png` | Validación estadística Kruskal-Wallis | `scipy.stats.kruskal` |

### 🗺️ Cartografía Visual (2 figuras)

| # | Archivo | Descripción | Método |
|---|---------|-------------|--------|
| 24 | `output/image_maps/mapa_visual_pca_GEN 1.png` | Mapa PCA con sprites de Pokémon | `OffsetImage` + `AnnotationBbox` |
| 25 | `output/image_maps/mapa_visual_tsne_GEN 1.png` | Mapa t-SNE con sprites de Pokémon | `OffsetImage` + `AnnotationBbox` |

---

## 🛠️ Tecnologías y Librerías

| Librería | Versión | Propósito |
|----------|---------|-----------|
| Python | 3.12+ | Lenguaje base |
| pandas | 2.x | Manipulación y análisis de datos |
| numpy | 1.x | Computación numérica |
| matplotlib | 3.x | Visualización estática y publicación |
| seaborn | 0.13+ | Visualización estadística |
| scikit-learn | 1.4+ | ML: PCA, t-SNE, K-Means, DBSCAN, métricas |
| scipy | 1.12+ | Estadística: Kruskal-Wallis, clustering jerárquico |
| kagglehub | 0.x | Descarga programática de datasets |

---

## ⚙️ Instalación y Uso

### Requisitos
```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn scipy
```

### Ejecución
```bash
python pokemon-database.py
```
Seleccione el alcance del análisis:
```
1. Solo Primera Generación (Gen 1)
2. Todos los Pokémon (Completo)
```

Para omitir la selección interactiva y usar Gen 1 por defecto:
```bash
echo 1 | python pokemon-database.py
```

---

## 📁 Estructura de Salida

```
output/
├── analisis_resultados_GEN 1.txt       # Reporte científico completo
├── diccionario_graficos_GEN 1.json     # Metadatos de gráficos (JSON)
├── diccionario_graficos_GEN 1.md       # Diccionario de gráficos (Markdown)
│
├── descriptive/                        # 4 figuras EDA
│   ├── histogram_resistencias_GEN 1.png
│   ├── boxplot_resistencias_GEN 1.png
│   ├── heatmap_correlaciones_GEN 1.png
│   └── coordenadas_paralelas_GEN 1.png
│
├── dimensionality_reduction/           # 8 figuras PCA + t-SNE
│   ├── scree_plot_pca_GEN 1.png
│   ├── loadings_heatmap_pca_GEN 1.png
│   ├── circulo_correlaciones_pca_GEN 1.png
│   ├── biplot_pca_GEN 1.png
│   ├── calidad_representacion_cos2_GEN 1.png
│   ├── tsne_perplejidad_grid_GEN 1.png
│   ├── tsne_learning_rate_grid_GEN 1.png
│   ├── tsne_trustworthiness_GEN 1.png
│   └── comparativa_pca_tsne_GEN 1.png
│
├── clustering/                         # 11 figuras clustering
│   ├── optimizacion_multimetrica_GEN 1.png
│   ├── gap_statistic_GEN 1.png
│   ├── dendrograma_ward_GEN 1.png
│   ├── silhouette_diagram_GEN 1.png
│   ├── caracterizacion_heatmap_GEN 1.png
│   ├── radar_arquetipos_GEN 1.png
│   ├── pca_clusters_GEN 1.png
│   ├── tsne_clusters_GEN 1.png
│   ├── dbscan_k_distance_GEN 1.png
│   ├── pca_dbscan_GEN 1.png
│   └── kruskal_wallis_resultados_GEN 1.png
│
└── image_maps/                         # 2 figuras cartográficas
    ├── mapa_visual_pca_GEN 1.png
    └── mapa_visual_tsne_GEN 1.png
```

---

## 🔬 Interpretación Científica

### Validez de la Segmentación
La segmentación en 4 arquetipos se valida mediante:
1. **Kruskal-Wallis**: 17/18 variables con diferencias significativas (p<0.05)
2. **Davies-Bouldin**: 1.46 — solapamiento bajo entre clusters
3. **Silhouette**: 0.263 — cohesión moderada, esperable para datos con alta dimensionalidad
4. **Gap Statistic**: Indica que podrían existir hasta 9 subgrupos finos

### Justificación del Enfoque No Paramétrico
- **97% de las variables no son normales** (D'Agostino-Pearson, p<0.05)
- Se emplean: Kruskal-Wallis, silhouette score, mediana como medida central
- La estandarización Z-score permite comparación equitativa entre escalas

### Limitaciones y Advertencias
1. **PCA en 2D captura solo [Ver reporte] de varianza** — la interpretación de ejes debe ser cautelosa
2. **Silhouette de 0.263** indica clusters con bordes difusos, lo cual es esperable dado que los tipos elementales no son ortogonales
3. **DBSCAN con 18.5% de ruido** sugiere que existen especialistas puros que no encajan en los arquetipos generales

---

## 👨‍💻 Autor

**Alejandro Quintero**
Maestría en Analítica de Datos — Politécnico Grancolombiano
Facultad de Ingeniería, Diseño e Innovación

### Referencias Académicas
- Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding*. SODA.
- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. JMLR.
- Pearson, K. (1901). *On lines and planes of closest fit*. Philosophical Magazine.
- Kruskal, W. H., & Wallis, W. A. (1952). *Use of ranks in one-criterion variance analysis*. JASA.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). *Estimating the number of clusters via the gap statistic*. JRSS.

---

## 📄 Licencia

Este proyecto se distribuye bajo licencia MIT. Los datos de Pokémon son propiedad de Nintendo/Game Freak.

---
*Generado automáticamente — 2026-06-20*
