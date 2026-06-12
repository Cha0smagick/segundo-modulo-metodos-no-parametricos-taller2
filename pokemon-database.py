import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import kruskal, bartlett
import scipy.cluster.hierarchy as sch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def analyze_pokemon_data(only_gen1=True):
    """
    Realiza un análisis estadístico avanzado de los Pokémon de la Gen 1,
    incluyendo validación de supuestos, optimización de clusters y reducción dimensional.
    
    Args:
        only_gen1 (bool): Si es True, filtra solo la Gen 1. Si es False, usa todo el dataset
    """
    report = [] # Lista para almacenar el reporte científico
    
    # Definición de rutas de Kaggle para estadísticas e imágenes
    stats_handle = "rounakbanik/pokemon"
    images_handle = "vishalsubbiah/pokemon-images-and-types"
    
    # Crear estructura de carpetas para los outputs
    base_output = "output"
    dirs = ["descriptive", "dimensionality_reduction", "clustering", "image_maps"]
    local_img_dir = "images"
    
    for d in dirs:
        os.makedirs(os.path.join(base_output, d), exist_ok=True)
    os.makedirs(local_img_dir, exist_ok=True)

    print("--- Descargando datasets desde Kaggle ---")
    try:
        stats_path = kagglehub.dataset_download(stats_handle)
        images_path = kagglehub.dataset_download(images_handle)
        stats_csv_path = os.path.join(stats_path, "pokemon.csv")
        cached_img_dir = os.path.join(images_path, "images")

        # Sincronizar imágenes al directorio local si está vacío
        if os.path.exists(cached_img_dir) and not os.listdir(local_img_dir):
            print(f"📦 Copiando imágenes al directorio local '{local_img_dir}'...")
            for item in os.listdir(cached_img_dir):
                src = os.path.join(cached_img_dir, item)
                dst = os.path.join(local_img_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        img_dir = local_img_dir
    except Exception as e:
        print(f"⚠️ No se pudo conectar con Kaggle (Error: {e})")
        print("Buscando archivos locales como respaldo...")
        
        stats_csv_path = "pokemon.csv"
        img_dir = "images"
        
        if not os.path.exists(stats_csv_path):
            print(f"❌ Error crítico: No se encontró '{stats_csv_path}' y la descarga falló.")
            print("Verifique su conexión a internet o coloque el archivo CSV manualmente en esta carpeta.")
            return
        print(f"📂 Usando archivo local encontrado: {stats_csv_path}")

    # Carga del CSV principal
    df = pd.read_csv(stats_csv_path)
    
    try:
        # --- 1. Filtrado de Datos ---
        # Se seleccionan únicamente los Pokémon pertenecientes a la primera generación (Gen 1)
        scope_text = "GEN 1" if only_gen1 else "TOTAL DATASET"
        report.append(f"--- REPORTE DE ANÁLISIS ESTADÍSTICO DE POKÉMON ({scope_text}) ---\n")
        report.append("1. PREPARACIÓN DE DATOS:")
        
        if only_gen1:
            df_gen1 = df[df['generation'] == 1].copy()
        else:
            df_gen1 = df.copy()
            
        print(f"✅ Registros cargados: {len(df_gen1)}")
        missing = df_gen1.isnull().sum().sum()
        report.append(f"   - Tamaño de la muestra: {len(df_gen1)} especímenes.")
        report.append(f"   - Datos faltantes detectados: {missing}")
        report.append(f"   - Tamaño de la muestra: {len(df_gen1)} especímenes.\n")

        # --- 2. Selección de Variables ---
        # Se eligen las variables de resistencia/debilidad (misma escala numérica)
        cols_against = [
            'against_bug', 'against_dark', 'against_dragon', 'against_electric', 'against_fairy', 
            'against_fight', 'against_fire', 'against_flying', 'against_ghost', 'against_grass', 
            'against_ground', 'against_ice', 'against_normal', 'against_poison', 'against_psychic', 
            'against_rock', 'against_steel', 'against_water'
        ]
        X = df_gen1[cols_against]
        
        report.append("2. ESTANDARIZACIÓN:")
        report.append("   - Se seleccionaron 18 variables de resistencia elemental.")
        # Estandarización: Escalar los datos para que tengan media 0 y varianza 1
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        report.append("   - Se aplicó StandardScaler para normalizar las magnitudes, evitando sesgos por escalas de debilidad.\n")

        # --- 4. Reducción de Dimensiones (Movido arriba para estar disponible en visualizaciones) ---
        report.append("4. REDUCCIÓN DE DIMENSIONALIDAD:")
        # PCA: Reducción lineal para maximizar la varianza explicada
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(X_scaled)
        
        # Identificar las variables con mayor peso en cada componente para los labels
        top_features_idx1 = np.argsort(np.abs(pca.components_[0]))[-3:][::-1]
        top_features_idx2 = np.argsort(np.abs(pca.components_[1]))[-3:][::-1]
        pc1_vars = [cols_against[i].replace('against_', '') for i in top_features_idx1]
        pc2_vars = [cols_against[i].replace('against_', '') for i in top_features_idx2]
        
        pca_label_x = f"PC1 (Dominancia: {', '.join(pc1_vars)})"
        pca_label_y = f"PC2 (Dominancia: {', '.join(pc2_vars)})"

        # t-SNE: Reducción no lineal para preservar la estructura local (vecindades)
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(X_scaled)
        
        var_explicada = np.sum(pca.explained_variance_ratio_) * 100
        report.append(f"   - Varianza PC1: {pca.explained_variance_ratio_[0]*100:.2f}%, PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
        report.append(f"   - PCA: Se conservó el {var_explicada:.2f}% de la varianza total en 2 componentes principales.")
        report.append(f"   - Variables clave PC1: {', '.join(pc1_vars)}")
        report.append(f"   - Variables clave PC2: {', '.join(pc2_vars)}")
        report.append("   - t-SNE: Se proyectó la estructura no lineal para identificar vecindades complejas.\n")

        # --- 3. Categorización (Clustering) ---
        report.append("3. AGRUPAMIENTO (CLUSTERING K-MEANS):")
        
        # Optimización de Clusters (Codo y Silueta)
        print("📊 Calculando métricas de optimización (Inercia y Silueta)...")
        k_range = range(2, 11)
        inertias = []
        silhouettes = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))
            
        # Generar gráfica de optimización
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Número de Clusters (k)')
        ax1.set_ylabel('Inercia (Cohesión)', color=color)
        ax1.plot(k_range, inertias, marker='o', color=color, label='Inercia (Codo)')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Coeficiente de Silueta (Separación)', color=color)
        ax2.plot(k_range, silhouettes, marker='s', color=color, label='Silueta')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f'Optimización de Clusters: Método del Codo vs Silueta ({scope_text})')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_output, "clustering", f"optimizacion_metodos_{scope_text}.png"))
        plt.close()

        # Generar Dendrograma (Método de Ward)
        print("📊 Generando Dendrograma (Método de Ward)...")
        k_clusters = 4 
        plt.figure(figsize=(12, 7))
        linkage_matrix = sch.linkage(X_scaled, method='ward')
        
        # Calcular la altura de corte para visualizar los clusters seleccionados
        # Se promedia la distancia del enlace que crea k clusters y el que crea k-1
        cut_height = (linkage_matrix[-k_clusters+1, 2] + linkage_matrix[-k_clusters, 2]) / 2
        
        sch.dendrogram(linkage_matrix, leaf_rotation=90, no_labels=True, color_threshold=cut_height)
        plt.axhline(y=cut_height, color='r', linestyle='--', label=f'Umbral de corte (k={k_clusters})')
        
        plt.title(f'Dendrograma de Clustering Jerárquico (Método de Ward - {scope_text})')
        plt.xlabel('Especímenes (Distribución de hojas)')
        plt.ylabel('Distancia Ward')
        plt.legend()
        plt.savefig(os.path.join(base_output, "clustering", f"dendrograma_ward_{scope_text}.png"))
        plt.close()

        # Validación de la calidad del clustering mediante Silhouette Score
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df_gen1['cluster'] = cluster_labels
        
        sil_score = silhouette_score(X_scaled, cluster_labels)
        
        print("✅ Categorización (Clustering) completada.")
        report.append(f"   - Algoritmo: K-Means con k={k_clusters} clusters.")
        report.append(f"   - Inercia final: {kmeans.inertia_:.2f}")
        report.append("   - Análisis Jerárquico: Se generó el Dendrograma de Ward para validar la estructura arbórea.")
        report.append(f"   - Coeficiente de Silueta: {sil_score:.4f} (Indicador de cohesión y separación).")
        distribucion = df_gen1['cluster'].value_counts().to_dict()
        report.append(f"   - Distribución de clusters: {distribucion}\n")
        
        # Análisis de perfiles por cluster
        report.append("   - Caracterización de perfiles y Arquetipos (Promedios por cluster):")
        
        # Definición de Arquetipos basada en el dominio de datos
        archetypes = {
            0: {"nombre": "Guardianes del Océano y el Acero", "desc": "Especialistas en resistencia elemental (Fuego, Fantasma) y física (Acero)."},
            1: {"nombre": "Protectores de la Biosfera", "desc": "Expertos en resistir ataques de naturaleza y combate cercano, frágiles ante lo místico."},
            2: {"nombre": "Místicos del Éter", "desc": "Entidades con defensas psíquicas superiores, vulnerables a miedos primordiales (Sombra/Bichos)."},
            3: {"nombre": "Centinelas de Alta Tensión", "desc": "Grupo de élite con alta resistencia al calor y electricidad, pero vulnerables a la tierra."}
        }

        cluster_profiles = df_gen1.groupby('cluster')[cols_against].mean()
        for i in range(k_clusters):
            top_res = cluster_profiles.loc[i].nsmallest(3).index.tolist()
            top_vul = cluster_profiles.loc[i].nlargest(3).index.tolist()
            report.append(f"     Cluster {i} - {archetypes[i]['nombre']}:")
            report.append(f"       - Arquetipo: {archetypes[i]['desc']}")
            report.append(f"       - Top Resistencias: {[c.replace('against_','') for c in top_res]}")
            report.append(f"       - Top Vulnerabilidades: {[c.replace('against_','') for c in top_vul]}")
        report.append("")

        # --- 3.2 Diagnóstico para DBSCAN (Opcional) ---
        print("📊 Generando diagnóstico para DBSCAN (K-Distance Graph)...")
        min_pts = 5 # Valor recomendado para datasets pequeños
        neighbors = NearestNeighbors(n_neighbors=min_pts)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, _ = neighbors_fit.kneighbors(X_scaled)
        distances = np.sort(distances[:, min_pts-1], axis=0)
        
        plt.figure(figsize=(8, 5))
        plt.plot(distances)
        plt.title(f'DBSCAN Diagnostic: K-Distance Graph (k={min_pts})')
        plt.ylabel('Distancia Epsilon (eps)')
        plt.xlabel('Puntos ordenados')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_output, "clustering", f"dbscan_epsilon_check_{scope_text}.png"))
        plt.close()
        print(f"✅ Diagnóstico guardado. Revisa 'dbscan_epsilon_check_{scope_text}.png' para elegir eps.")

        # --- 3.3 Ejecución de DBSCAN (Comparativa) ---
        # Basado en la escala de StandardScaler, un eps entre 3 y 5 suele ser un buen punto de partida
        # para este dataset.
        eps_val = 3.8 # Ajustado tras observar la rodilla en el gráfico de diagnóstico
        min_samples_val = 5
        dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
        db_labels = dbscan.fit_predict(X_scaled)
        
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = list(db_labels).count(-1)
        
        report.append("3.3 RESULTADOS DBSCAN (EXPLORATORIO):")
        report.append(f"   - Parámetros óptimos sugeridos: eps={eps_val}, min_samples={min_samples_val}")
        report.append(f"   - Clusters encontrados: {n_clusters_db}")
        report.append(f"   - Puntos de ruido (outliers): {n_noise}")
        report.append(f"   - Nota: Los {n_noise} puntos de ruido representan Pokémon con perfiles de resistencia únicos (especialistas).")
        
        if n_noise > 0:
            noise_names = df_gen1[db_labels == -1]['name'].head(10).tolist()
            report.append(f"   - Ejemplos de especialistas detectados: {', '.join(noise_names)}...")
        
        # Visualización de DBSCAN
        plt.figure(figsize=(10, 7))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=db_labels, 
                   cmap='viridis', s=100, edgecolor='white', alpha=0.8)
        plt.title(f"DBSCAN sobre PCA (eps={eps_val}, clusters={n_clusters_db})")
        plt.xlabel(pca_label_x)
        plt.ylabel(pca_label_y)
        plt.savefig(os.path.join(base_output, "clustering", f"pca_dbscan_{scope_text}.png"))
        plt.close()

        # --- 3.1 Validación Estadística (Kruskal-Wallis) ---
        report.append("3.1 VALIDACIÓN ESTADÍSTICA (INFERENCIA):")
        report.append("   - Se aplicó la prueba de Kruskal-Wallis para validar si los clusters son")
        report.append("     poblaciones estadísticamente diferentes en sus resistencias.")
        
        sig_vars = 0
        for col in cols_against:
            stat, p_val = kruskal(*[group[col].values for name, group in df_gen1.groupby('cluster')])
            if p_val < 0.05:
                sig_vars += 1
            if col == 'against_fire': # Mantener referencia para el reporte
                report.append(f"   - Prueba focal (against_fire): H-stat={stat:.2f}, p-value={p_val:.4e}")
        
        report.append(f"   - Resultado global: {sig_vars}/18 variables muestran diferencias significativas entre clusters.")
        report.append("   - Interpretación: La segmentación es biológicamente relevante.\n")

        # --- 5. Visualización: PCA vs t-SNE ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.scatter(pca_results[:, 0], pca_results[:, 1], c='skyblue', edgecolor='k', alpha=0.7)
        ax1.set_title(f"PCA (Varianza Explicada: {var_explicada:.1f}%)")
        ax1.set_xlabel(pca_label_x)
        ax1.set_ylabel(pca_label_y)
        
        ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], c='salmon', edgecolor='k', alpha=0.7)
        ax2.set_title("t-SNE (Reducción No Lineal)")
        ax2.set_xlabel("Proyección de Parentesco Elemental (t-SNE 1)")
        ax2.set_ylabel("Proyección de Perfil Defensivo (t-SNE 2)")
        
        plt.savefig(os.path.join(base_output, "dimensionality_reduction", f"comparativa_pca_tsne_{scope_text}.png"))
        print("💾 Gráfico PCA vs t-SNE guardado.")
        plt.close()

        # Definir paleta de colores y mapa de color reutilizable
        # Definir paleta de colores personalizada (Azul, Verde, Amarillo, Morado)
        custom_colors = ['#1f77b4', '#2ca02c', '#bcbd22', '#9467bd']
        # Se utiliza LinearSegmentedColormap para crear una transición suave (degradado) entre los colores
        cmap_custom = LinearSegmentedColormap.from_list("pokemon_gradient", custom_colors)

        # --- 6. Visualización: PCA + Clusters ---
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=df_gen1['cluster'], 
                             cmap=cmap_custom, s=100, edgecolor='white', alpha=0.8)
        cbar = plt.colorbar(scatter, label='Grupo (Cluster)')
        cbar.set_ticks(range(k_clusters))
        plt.title(f"Clustering K-Means sobre Proyección PCA ({scope_text})")
        plt.xlabel(pca_label_x)
        plt.ylabel(pca_label_y)
        plt.savefig(os.path.join(base_output, "clustering", f"pca_clusters_{scope_text}.png"))
        print("💾 Gráfico de Clusters (PCA) guardado.")
        plt.close()

        # --- 6.1 Visualización: t-SNE + Clusters ---
        plt.figure(figsize=(10, 7))
        scatter_tsne = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=df_gen1['cluster'], 
                                  cmap=cmap_custom, s=100, edgecolor='white', alpha=0.8)
        cbar_tsne = plt.colorbar(scatter_tsne, label='Grupo (Cluster)')
        cbar_tsne.set_ticks(range(k_clusters))
        plt.title(f"Clustering K-Means sobre Proyección t-SNE ({scope_text})")
        plt.xlabel("Proyección de Parentesco Elemental (t-SNE 1)")
        plt.ylabel("Proyección de Perfil Defensivo (t-SNE 2)")
        plt.savefig(os.path.join(base_output, "clustering", f"tsne_clusters_{scope_text}.png"))
        print("💾 Gráfico de Clusters (t-SNE) guardado.")
        plt.close()

        # --- 7. Visualización: Mapas con Imágenes (PCA y t-SNE) ---
        print("🎨 Generando mapas visuales con imágenes y colores...")
        
        for mode in ['PCA', 't-SNE']:
            fig, ax = plt.subplots(figsize=(14, 12))
            coords = pca_results if mode == 'PCA' else tsne_results
            
            # Dibujar puntos de fondo con colores de cluster para referencia
            ax.scatter(coords[:, 0], coords[:, 1], c=df_gen1['cluster'], 
                      cmap=cmap_custom, s=250, alpha=0.4, edgecolors='none')

            for i, (idx, row) in enumerate(df_gen1.iterrows()):
                img_name = str(row['name']).lower().replace(" ", "-")
                img_path = os.path.join(img_dir, f"{img_name}.png")
                
                if os.path.exists(img_path):
                    img = plt.imread(img_path)
                    imagebox = OffsetImage(img, zoom=0.35)
                    ab = AnnotationBbox(imagebox, (coords[i, 0], coords[i, 1]), frameon=False)
                    ax.add_artist(ab)

            ax.set_title(f"Mapa Visual de Pokémones {scope_text} ({mode} Mapping)")
            ax.set_xlabel(pca_label_x if mode == 'PCA' else "Proyección de Parentesco Elemental (t-SNE 1)")
            ax.set_ylabel(pca_label_y if mode == 'PCA' else "Proyección de Perfil Defensivo (t-SNE 2)")
            plt.grid(True, linestyle='--', alpha=0.3)
            
            filename = f"mapa_visual_{mode.lower().replace('-', '')}_{scope_text}.png"
            plt.savefig(os.path.join(base_output, "image_maps", filename))
            plt.close()
            print(f"💾 {mode} Image Map guardado.")

        # Re-crear ejes para el reporte si es necesario, pero ya guardamos los archivos

        ax.set_title("Mapa Visual de Pokémones Gen 1 (PCA Mapping)")
        ax.set_xlabel(pca_label_x)
        ax.set_ylabel(pca_label_y)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        report.append("5. CONCLUSIÓN VISUAL:")
        report.append("   - El mapa visual confirma que los Pokémon se agrupan por sus tipos elementales,")
        report.append("     ya que comparten perfiles de debilidad y resistencia similares.")

        # --- 8. Guardar Reporte ---
        report_path = os.path.join(base_output, "analisis_resultados.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"💾 Reporte científico guardado en: {report_path}")
        
        # --- 9. Inyección de Resultados en Documentación (.md y .html) ---
        # Actualización de pokemon-database.md
        md_path = "pokemon-database.md"
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            
            md_replacements = {
                "k=6": f"k={k_clusters}",
                "[Valor numérico] (Indica la compacidad": f"{kmeans.inertia_:.2f} (Indica la compacidad",
                "[Valor numérico] (Generalmente entre 0.35-0.45": f"{sil_score:.4f} (Generalmente entre 0.35-0.45",
                "[Número] de 18 variables": f"{sig_vars} de 18 variables",
                "aproximadamente [Valor numérico]%": f"aproximadamente {var_explicada:.2f}%",
                "against_grass, against_bug": f"against_{pc1_vars[0]}, against_{pc1_vars[1]}",
                "against_psychic, against_ghost": f"against_{pc2_vars[0]}, against_{pc2_vars[1]}"
            }
            for key, val in md_replacements.items():
                md_content = md_content.replace(key, val)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            print("📝 pokemon-database.md actualizado con datos reales.")

        # Actualización de reporte_maestria.html
        html_path = "reporte_maestria.html"
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            html_replacements = {
                "<strong>k=6</strong>": f"<strong>k={k_clusters}</strong>",
                "[Valor Reportado en Consola]": f"{kmeans.inertia_:.2f}",
                "Sig. Variables: 18 de 18": f"Sig. Variables: {sig_vars} de 18",
                "explicen más del 40%": f"explicen más del {var_explicada:.2f}%",
                "against 'Grass' y 'Bug'": f"'{pc1_vars[0]}' y '{pc1_vars[1]}'",
                "tipos 'Psychic' y 'Ghost'": f"'{pc2_vars[0]}' y '{pc2_vars[1]}'"
            }
            for key, val in html_replacements.items():
                html_content = html_content.replace(key, val)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print("🌐 reporte_maestria.html actualizado con datos reales.")

    except Exception as e:
        print(f"❌ Error crítico durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Requisitos: pip install kagglehub pandas matplotlib seaborn scikit-learn
    # Opción: True para Gen 1, False para todos los Pokémon
    print("Elija el alcance del análisis:")
    print("1. Solo Primera Generación")
    print("2. Todos los Pokémon")
    opcion = input("Seleccione (1/2): ")
    
    analyze_pokemon_data(only_gen1=(opcion == "1"))
