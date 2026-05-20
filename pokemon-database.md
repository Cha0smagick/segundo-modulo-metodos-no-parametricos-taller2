# Análisis Multivariado y Taxonomía Algorítmica de Pokémon (Gen 1)

Este proyecto implementa un análisis estadístico avanzado para segmentar y clasificar Pokémon de la Primera Generación (Gen 1) basándose en sus perfiles de resistencia y debilidad elemental. Utilizando técnicas de clustering (K-Means++) y reducción de dimensionalidad (PCA, t-SNE), el objetivo es descubrir patrones ocultos y validar estadísticamente la existencia de grupos taxonómicos distintos.

## 🎯 Objetivo del Proyecto

El propósito principal es transformar un conjunto de datos multidimensional (18 variables de resistencia elemental) en una estructura de conocimiento organizada. Se busca identificar "nichos ecológicos" o grupos de Pokémon con perfiles de combate similares, y validar estos grupos mediante pruebas estadísticas no paramétricas, ofreciendo una comprensión profunda de las interacciones de poder y debilidad en el universo Pokémon.

## 🚀 Metodología y Pipeline Analítico

El análisis sigue un riguroso pipeline de ciencia de datos:

1.  **Adquisición de Datos**: Descarga automática de datasets de Pokémon (estadísticas e imágenes) desde Kaggle.
2.  **Preprocesamiento de Datos**:
    *   Filtrado de Pokémon de la Primera Generación (opcionalmente, se puede analizar todo el dataset).
    *   Selección de 18 variables de resistencia/debilidad elemental (`against_bug` a `against_water`).
    *   **Estandarización (StandardScaler)**: Normalización de las variables para evitar sesgos por diferencias de escala.
3.  **Segmentación Algorítmica (Clustering)**:
    *   Aplicación del algoritmo **K-Means++** con `k=4` clusters para minimizar la inercia intracluster.
    *   Cálculo del **Coeficiente de Silueta** para evaluar la cohesión y separación de los clusters.
4.  **Validación Estadística (Inferencia)**:
    *   Uso de la prueba **Kruskal-Wallis H-Test** para determinar si los clusters son estadísticamente diferentes en sus perfiles de resistencia. Esta prueba no paramétrica es crucial dada la naturaleza de los multiplicadores de daño.
5.  **Reducción de Dimensionalidad**:
    *   **Análisis de Componentes Principales (PCA)**: Proyección lineal a 2 dimensiones para capturar la máxima varianza y visualizar la estructura global de los datos.
    *   **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Proyección no lineal a 2 dimensiones para preservar las relaciones de vecindad local y revelar agrupaciones complejas.
6.  **Visualización de Resultados**:
    *   Gráficos de dispersión de PCA y t-SNE.
    *   Visualización de clusters sobre la proyección PCA.
    *   **Cartografía Visual**: Proyección de los sprites de Pokémon en el espacio PCA, coloreados por cluster, para una interpretación intuitiva de los grupos.

## 🛠️ Tecnologías y Librerías

*   **Python 3.x**
*   **pandas**: Manipulación y análisis de datos.
*   **numpy**: Computación numérica.
*   **matplotlib.pyplot**: Creación de gráficos estáticos.
*   **seaborn**: Visualizaciones estadísticas.
*   **scikit-learn**: Implementación de `StandardScaler`, `KMeans`, `PCA`, `TSNE`, `silhouette_score`.
*   **scipy.stats**: Pruebas estadísticas como `kruskal`.
*   **kagglehub**: Descarga programática de datasets de Kaggle.
*   **os**: Interacción con el sistema de archivos.

## ⚙️ Instalación y Uso

### Requisitos

Asegúrate de tener Python 3.x instalado.

### Instalación de Dependencias

```bash
pip install kagglehub pandas matplotlib seaborn scikit-learn scipy
```

### Configuración de Kaggle (Opcional, `kagglehub` lo maneja automáticamente)

`kagglehub` gestiona la autenticación automáticamente. Si tienes problemas, asegúrate de que tu API de Kaggle esté configurada (archivo `kaggle.json` en `~/.kaggle/`).

### Ejecución del Script

```bash
python pokemon-database.py
```

Al ejecutar el script, se te preguntará si deseas analizar solo la Primera Generación o todos los Pokémon:

```
Elija el alcance del análisis:
1. Solo Primera Generación
2. Todos los Pokémon
Seleccione (1/2):
```

### Salida del Proyecto

El script generará una carpeta `output/` con la siguiente estructura:

```
output/
├── analisis_resultados.txt         # Reporte científico detallado del análisis.
├── clustering/
│   └── pca_clusters_GEN 1.png      # Gráfico PCA con los clusters identificados.
├── dimensionality_reduction/
│   └── comparativa_pca_tsne_GEN 1.png # Comparativa visual de PCA vs t-SNE.
├── descriptive/                    # (Puede contener gráficos descriptivos si se añaden en el futuro)
└── image_maps/
    └── mapa_visual_GEN 1.png       # Mapa visual de Pokémon con sprites en el espacio PCA.
```

## 📊 Resultados Clave

*   **Clustering (K-Means++)**: Se identificaron 4 clusters principales caracterizados por su perfil defensivo:
    *   <strong style="color: #1f77b4;">Cluster Azul:</strong> Especialistas acuáticos y de hielo.
    *   <strong style="color: #2ca02c;">Cluster Verde:</strong> Especies botánicas y bípodes.
    *   <strong style="color: #bcbd22;">Cluster Amarillo:</strong> Especies eléctricas y de tierra.
    *   <strong style="color: #9467bd;">Cluster Morado:</strong> Especialistas en tipos veneno, fantasma y psíquico.
*   **Inercia Final**: `[Valor numérico]` (Indica la compacidad de los clusters).
*   **Coeficiente de Silueta**: `[Valor numérico]` (Generalmente entre 0.35-0.45, indicando una buena separación y cohesión).
*   **Validación Kruskal-Wallis**: `[Número]` de 18 variables mostraron diferencias significativas entre clusters (p < 0.05), confirmando la validez estadística de la segmentación.
*   **PCA**: Los dos primeros componentes principales explican aproximadamente `[Valor numérico]`% de la varianza total, con `against_grass`, `against_bug` y `against_psychic`, `against_ghost` como variables dominantes en PC1 y PC2 respectivamente.
*   **Visualización**: La cartografía con sprites de Pokémon confirma visualmente que los grupos formados por K-Means se corresponden con patrones de resistencia y debilidad coherentes.

## 📝 Reporte Científico

Un reporte detallado (`analisis_resultados.txt`) se genera al final de la ejecución, conteniendo todos los pasos, métricas y conclusiones del análisis.

## 👨‍💻 Autor

**Alejandro Quintero**
Maestría en Analítica de Datos - Politécnico Grancolombiano

## 📄 Licencia

Este proyecto se distribuye bajo la licencia [MIT License / Licencia Propia / etc.].

---

**Nota**: Los valores numéricos exactos para la inercia, coeficiente de silueta y varianza explicada se encuentran en el archivo `analisis_resultados.txt` generado por el script.

```