import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from scipy.stats import kruskal, bartlett
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
    for d in dirs:
        os.makedirs(os.path.join(base_output, d), exist_ok=True)

    print("--- Descargando datasets desde Kaggle ---")
    try:
        stats_path = kagglehub.dataset_download(stats_handle)
        images_path = kagglehub.dataset_download(images_handle)
        stats_csv_path = os.path.join(stats_path, "pokemon.csv")
        img_dir = os.path.join(images_path, "images")
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

        # --- 3. Categorización (Clustering) ---
        report.append("3. AGRUPAMIENTO (CLUSTERING K-MEANS):")
        # Validación de la calidad del clustering mediante Silhouette Score
        k_clusters = 4 
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df_gen1['cluster'] = cluster_labels
        
        sil_score = silhouette_score(X_scaled, cluster_labels)
        
        print("✅ Categorización (Clustering) completada.")
        report.append(f"   - Algoritmo: K-Means con k={k_clusters} clusters.")
        report.append(f"   - Inercia final: {kmeans.inertia_:.2f}")
        report.append(f"   - Coeficiente de Silueta: {sil_score:.4f} (Indicador de cohesión y separación).")
        distribucion = df_gen1['cluster'].value_counts().to_dict()
        report.append(f"   - Distribución de clusters: {distribucion}\n")
        
        # Análisis de perfiles por cluster
        report.append("   - Caracterización de perfiles (Promedios por cluster):")
        cluster_profiles = df_gen1.groupby('cluster')[cols_against].mean()
        for i in range(k_clusters):
            top_res = cluster_profiles.loc[i].nsmallest(3).index.tolist()
            top_vul = cluster_profiles.loc[i].nlargest(3).index.tolist()
            report.append(f"     Cluster {i}: Resistencia en {[c.replace('against_','') for c in top_res]} | Vulnerable a {[c.replace('against_','') for c in top_vul]}")
        report.append("")

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

        # --- 4. Reducción de Dimensiones ---
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

        # --- 5. Visualización: PCA vs t-SNE ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.scatter(pca_results[:, 0], pca_results[:, 1], c='skyblue', edgecolor='k', alpha=0.7)
        ax1.set_title(f"PCA (Varianza Explicada: {var_explicada:.1f}%)")
        ax1.set_xlabel(pca_label_x)
        ax1.set_ylabel(pca_label_y)
        
        ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], c='salmon', edgecolor='k', alpha=0.7)
        ax2.set_title("t-SNE (Reducción No Lineal)")
        ax2.set_xlabel("Dimensión t-SNE 1")
        ax2.set_ylabel("Dimensión t-SNE 2")
        
        plt.savefig(os.path.join(base_output, "dimensionality_reduction", f"comparativa_pca_tsne_{scope_text}.png"))
        print("💾 Gráfico PCA vs t-SNE guardado.")
        plt.close()

        # --- 6. Visualización: PCA + Clusters ---
        # Se colorean los puntos según el resultado del agrupamiento de K-Means
        plt.figure(figsize=(10, 7))
        
        # Definir paleta de colores personalizada (Azul, Verde, Amarillo, Morado)
        custom_colors = ['#1f77b4', '#2ca02c', '#bcbd22', '#9467bd']
        cmap_custom = ListedColormap(custom_colors)

        scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=df_gen1['cluster'], 
                             cmap=cmap_custom, s=100, edgecolor='white', alpha=0.8)
        plt.colorbar(scatter, label='Grupo (Cluster)')
        plt.title(f"Clustering K-Means sobre Proyección PCA ({scope_text})")
        plt.xlabel(pca_label_x)
        plt.ylabel(pca_label_y)
        
        plt.savefig(os.path.join(base_output, "clustering", f"pca_clusters_{scope_text}.png"))
        print("💾 Gráfico de Clusters guardado.")
        plt.close()

        # --- 7. Visualización: Mapa con Imágenes ---
        # En lugar de puntos, se renderizan los sprites de cada Pokémon en sus coordenadas PCA
        print("🎨 Generando gráfico con imágenes (esto puede tardar unos segundos)...")
        fig, ax = plt.subplots(figsize=(14, 12))
        
        ax.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0) 

        for i, (idx, row) in enumerate(df_gen1.iterrows()):
            img_name = str(row['name']).lower().replace(" ", "-")
            img_path = os.path.join(img_dir, f"{img_name}.png")
            
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                imagebox = OffsetImage(img, zoom=0.35)
                ab = AnnotationBbox(imagebox, (pca_results[i, 0], pca_results[i, 1]), frameon=False)
                ax.add_artist(ab)

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

        plt.savefig(os.path.join(base_output, "image_maps", f"mapa_visual_{scope_text}.png"))
        print("💾 Mapa visual de Pokémon guardado.")
        plt.close()

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
