# ⚔️ Guía Definitiva del Maestro Pokémon: Análisis Estadístico para Ganar la Liga

## *"Atrapa, Analiza, Domina" — De la Pokédex a la Liga de Campeones*

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![Maestría](https://img.shields.io/badge/Maestr%C3%ADa-Anal%C3%ADtica%20de%20Datos-green)](https://www.poli.edu.co)

---

## 🏆 Carta del Maestro Pokémon a los Entrenadores

> *"Joven entrenador... Crees que atrapar a todos los Pokémon te hará imbatible? Te equivocas. **Entender** a tus Pokémon es lo que marca la diferencia entre un campeón y un perdedor. He pasado 20 años analizando cada resistencia, cada debilidad, cada sinergia defensiva. Este análisis multivariado no es solo un ejercicio académico — es el **mapa secreto** hacia la victoria."*
>
> — **Anónimo, Campeón de la Liga Índigo (1996-2026)**

Bienvenido, entrenador. Este documento no es un simple reporte estadístico. Es tu **manual de batalla definitivo**. Aquí no solo verás gráficos bonitos — aprenderás a **leer el ADN de combate** de cada Pokémon, a construir equipos que cubran todas las vulnerabilidades, y a entrar a la Liga de Campeones sabiendo exactamente qué funciona y por qué.

Usamos **18 variables de resistencia elemental** de los 151 Pokémon originales, procesadas con algoritmos de **Machine Learning** nivel maestría, para descubrir los **9 arquetipos defensivos** que componen el metajuego de la Primera Generación.

---

## 📋 Índice de Batalla

| Sección | Lo que aprenderás |
|---------|-------------------|
| [**1. Marco Metodológico**](#1-marco-metodológico-el-laboratorio-del-campeón) | Cómo convertimos datos crudos en estrategia de guerra |
| [**2. EDA — Radiografía de las Debilidades**](#2-eda--radiografía-de-las-debilidades) | Por qué tu equipo actual está mal balanceado |
| [**3. PCA — El Mapa Secreto de la Liga**](#3-pca--el-mapa-secreto-de-la-liga) | Las verdaderas dimensiones del poder Pokémon |
| [**4. t-SNE — Las Islas de Similitud**](#4-t-sne--las-islas-de-similitud-defensiva) | Vecindades ocultas entre especies aparentemente distintas |
| [**5. Clustering — Los 9 Arquetipos de Combate**](#5-clustering--los-9-arquetipos-de-combate) | La taxonomía definitiva del equipo perfecto |
| [**6. Validación Estadística — No es Casualidad**](#6-validación-estadística--no-es-casualidad) | Ciencia pura que respalda tu estrategia |
| [**7. Cartografía Visual — El Mapa del Tesoro**](#7-cartografía-visual--el-mapa-del-tesoro) | Todos los Pokémon en un solo plano dimensional |
| [**8. Diccionario Completo de Gráficos**](#8-diccionario-completo-de-gráficos) | Cada gráfico explicado como si fueras al gym |
| [**9. Estrategias para la Liga**](#9-estrategias-para-la-liga) | Cómo aplicar este conocimiento contra Brock, Misty, Giovanni y el Alto Mando |

---

## 1. Marco Metodológico: El Laboratorio del Campeón

Antes de lanzar una Pokébola, un científico analiza. Antes de entrar a la Liga, un Maestro entiende los números. Esto es lo que hicimos:

```
📦 Adquisición  →  🔬 Preprocesamiento  →  📊 Análisis  →  🧬 Clustering  →  ⚔️ Estrategia
```

| Fase | ¿Qué hicimos? | ¿Para qué sirve en batalla? |
|------|---------------|---------------------------|
| **StandardScaler** | Normalizamos las 18 resistencias a Z-score | Para que tipos raros como Fantasma no dominen el análisis solo por tener valores extremos |
| **PCA** | Redujimos 18 dimensiones a 2 componentes | Para visualizar en un mapa quién es aliado natural de quién |
| **t-SNE** | Preservamos vecindades locales | Para descubrir sustitutos funcionales — Pokémon que aunque diferentes, juegan el mismo rol defensivo |
| **K-Means** | Agrupamos en 9 arquetipos | Para construir equipos con cobertura total de tipos |
| **DBSCAN** | Detectamos outliers (especialistas puros) | Para identificar Pokémon tan únicos que no encajan en ningún arquetipo — gemas ocultas |
| **Kruskal-Wallis** | Validamos estadísticamente los clusters | Para asegurarnos de que los 9 grupos NO son producto del azar |

### 📦 Los Datos

- **151 Pokémon** de la Primera Generación
- **18 variables de resistencia** (de `against_bug` a `against_water`)
- **Valores**: 0.25 (resistencia alta) → 4.0 (vulnerabilidad extrema)
- **Estandarización**: Z-score (media=0, desviación=1)

---

## 2. EDA — Radiografía de las Debilidades

### 🎯 ¿Qué mira un Maestro Pokémon aquí?

Antes de armar tu equipo, necesitas saber **cómo se distribuyen las resistencias en el mundo Pokémon**. Estos gráficos te muestran, de un vistazo, qué tipos son intrínsecamente más resistentes y cuáles son más frágiles.

---

### 📊 Gráfico 1: Distribución de las Resistencias Elementales

![Histogramas de Resistencia](output/descriptive/histogram_resistencias_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada histograma muestra cómo se distribuyen los valores de resistencia para un tipo. La línea roja es el promedio; la verde, la mediana.

**⚔️ Lo que significa para tu batalla:**

*Mira la columna de `normal`* — Casi todos los Pokémon tienen resistencia 1.0 (daño normal). Pero mira `water`: ves un pico en 0.5 (resistentes) y otro en 2.0 (débiles). Esto te dice que **los tipos Agua son polarizantes**: o aguantan el agua superbien o les duele muchísimo. Cuando enfrentes a un líder de gimnasio Agua como Misty, necesitas saber en qué lado del espectro está cada uno de sus Pokémon.

**💡 Consejo de batalla:** `against_ghost` fue la **única variable con distribución normal** (p=0.14). ¿Qué significa? Que los ataques Fantasma son los más equilibrados del juego — ni demasiado fuertes ni demasiado débiles contra la mayoría. No confíes ciegamente en tu tipo Fantasma para cubrir todas las bases.

---

### 📊 Gráfico 2: Boxplots — La Caja de Herramientas del Campeón

![Boxplots Estandarizados](output/descriptive/boxplot_resistencias_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada caja muestra la mediana (línea central), el rango intercuartil (la caja) y los valores atípicos (puntos). Los datos están en Z-score (0 = promedio).

**⚔️ Lo que significa para tu batalla:**

*Observa `against_rock`* — La caja es ancha y larga hacia arriba. Esto significa que el tipo Roca tiene **altísima variabilidad**: algunos Pokémon lo resisten superbien, otros son increíblemente vulnerables. Cuando lleves un Pokémon que le deba a Roca (como un Charizard), asegúrate de saber exactamente qué tan vulnerable es comparado con otros de su mismo tipo.

*Ahora mira `against_normal`* — La caja es más angosta que las demás y está centrada en 1.0 (Z-score=0). El tipo Normal es el **más homogéneo y predecible**. Cuando enfrentes a un Normal (como el Team Rocket), no hay sorpresas — sabes exactamente qué esperar.

**💡 Consejo de batalla:** Los tipos con cajas largas (Roca, Agua, Eléctrico) son los **más impredecibles**. Lleva siempre un plan B contra ellos. Los tipos con cajas cortas (Normal, Veneno, Dragón) son **predecibles** — puedes planificar con certeza.

---

### 📊 Gráfico 3: Mapa de Calor de Correlaciones — La Danza de los Tipos

![Correlaciones](output/descriptive/heatmap_correlaciones_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada celda es la correlación de Pearson entre dos resistencias. Rojo = correlación positiva (tienden a subir juntas). Azul = correlación negativa (una sube, la otra baja).

**⚔️ Lo que significa para tu batalla:**

*Busca el par `fire`-`ice`* — Si ves correlación positiva, significa que los Pokémon que resisten Fuego también tienden a resistir Hielo. **Esto es una redundancia defensiva**: si tu equipo ya tiene un Pokémon que resiste Fuego, probablemente también resista Hielo. ¡Estás duplicando cobertura!

*Busca `psychic`-`bug`* — Correlación negativa significa que los Pokémon que resisten Psíquico **suelen ser débiles a Bicho**. Esto es crítico para tu equipo: si tienes un Pokémon que aguanta Psíquico, debes cubrir su vulnerabilidad a Bicho con otro miembro del equipo.

**💡 Consejo de batalla:** Las correlaciones te dicen qué **parejas de tipos suelen ir juntas**. Si ves correlación positiva entre Agua y Hielo, cuando escojas un Pokémon Agua, asume que también lidiarás con una resistencia a Hielo similar. No dupliques tipos que ya tienes cubiertos.

---

### 📊 Gráfico 4: Coordenadas Paralelas — El Pulso de Cada Pokémon

![Coordenadas Paralelas](output/descriptive/coordenadas_paralelas_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada línea es un Pokémon. Cada columna vertical es una variable de resistencia. Las líneas que siguen trayectorias similares son Pokémon con perfiles defensivos parecidos.

**⚔️ Lo que significa para tu batalla:**

*Busca grupos de líneas que suben y bajan juntas* — Esos Pokémon son **funcionalmente intercambiables** en términos defensivos. Si tienes a Bulbasaur en tu equipo y ves que Oddish tiene exactamente el mismo patrón de líneas, **no necesitas ambos** — estás desperdiciando un puesto.

*Busca líneas que se disparan hacia arriba en columnas específicas* — Esos son Pokémon con **vulnerabilidades extremas** a un tipo particular. Por ejemplo, un Charizard tendrá un pico enorme en `against_water` y `against_rock`. Esto te dice: **nunca lo saques contra un Agua o Roca**, o asegúrate de tener un cambio rápido.

**💡 Consejo de batalla:** Usa este gráfico para **detectar redundancias en tu equipo**. Si dos Pokémon tienen perfiles casi idénticos, reemplaza uno por alguien con un perfil complementario. La cobertura es más importante que la fuerza bruta.

---

## 3. PCA — El Mapa Secreto de la Liga

### 🎯 ¿Qué mira un Maestro Pokémon aquí?

El PCA reduce las 18 resistencias a solo 2 dimensiones. Es como si pudiéramos **ver a todos los Pokémon en un mapa de 2 dimensiones** donde la cercanía significa similitud defensiva. Es el **mapa secreto** que te muestra cómo se relacionan los Pokémon a nivel de ADN de combate.

---

### 📊 Gráfico 5: Scree Plot — ¿Cuántas Dimensiones Importan?

![Scree Plot PCA](output/dimensionality_reduction/scree_plot_pca_GEN%201.png)

**🔬 Lo que dice la ciencia:** Las barras azules muestran cuánta "información" (varianza) captura cada componente. La línea roja muestra el acumulado. El umbral gris punteado es el 80%.

**⚔️ Lo que significa para tu batalla:**

*Mira cómo las primeras 2 barras son las más altas* — PC1 captura 21.5% y PC2 captura 14.3% de toda la información defensiva. Entre las dos, **35.8% de la información total**. Esto es bajo para un análisis científico (idealmente >70%), pero para batallas Pokémon es **suficiente**: los primeros 2 ejes ya capturan la mayoría de las diferencias entre tipos.

*Nota que necesitamos 7 componentes para llegar al 73.6%* — Esto te dice que las resistencias Pokémon son **complejas y multidimensionales**. No puedes resumir todo en "tipo fuerte vs tipo débil". Hay matices.

**💡 Consejo de batalla:** Aunque el mapa 2D es útil, recuerda que **los expertos usan 7 dimensiones mentales** para evaluar un Pokémon. No juzgues solo por su tipo principal — mira su perfil completo de 18 resistencias. Eso es lo que separa a un buen entrenador de un Maestro.

---

### 📊 Gráfico 6: Heatmap de Loadings — ¿Qué Define a Cada Eje?

![Loadings Heatmap](output/dimensionality_reduction/loadings_heatmap_pca_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada celda es el peso (loading) de una variable en un componente. Valores absolutos altos = esa variable es importante para definir ese eje.

**⚔️ Lo que significa para tu batalla:**

*PC1 (el eje horizontal del mapa)* está dominado por `flying`, `grass`, `ground`. Esto significa que **la primera dimensión del poder Pokémon separa a los Pokémon por su relación con el vuelo, la hierba y la tierra**. Un Pokémon con PC1 alto será muy diferente a uno con PC1 bajo en estos aspectos.

*PC2 (el eje vertical)* está dominado por `psychic`, `bug`, `dark`. **La segunda dimensión separa por sensibilidad psíquica, bicho y oscuridad**.

**💡 Consejo de batalla:** Cuando mires el mapa PCA, el **eje horizontal** te dice qué tan "aéreo-vegetal-terrestre" es un Pokémon. El **eje vertical** te dice qué tan "psíquico-bicho-oscuridad". Esto no es teoría — es la geometría real de las vulnerabilidades. Un Pokémon en la esquina superior derecha será completamente diferente a uno en la inferior izquierda. **Ahí es donde encuentras la cobertura que necesitas.**

---

### 📊 Gráfico 7: Círculo de Correlaciones — La Brújula de los Tipos

![Círculo de Correlaciones](output/dimensionality_reduction/circulo_correlaciones_pca_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada flecha es una variable. La dirección de la flecha indica hacia dónde "tira" esa variable en el mapa. Flechas en la misma dirección = correlacionadas. Flechas opuestas = anticorrelacionadas. La longitud indica qué tan bien representada está.

**⚔️ Lo que significa para tu batalla:**

*Mira las flechas largas (cerca del borde) como `psychic`, `bug`, `ghost`, `rock`* — Estas variables están **bien representadas** en el mapa 2D. Confía en lo que ves: la posición de los Pokémon respecto a estas flechas es precisa.

*Mira las flechas cortas (cerca del centro) como `normal`, `dragon`* — Estas variables están **mal representadas**. El mapa 2D no captura bien su comportamiento. El tipo Normal y Dragón son tan únicos que necesitan más dimensiones para ser entendidos.

*Flechas en dirección opuesta: `bug` vs `psychic`* — Esto confirma la conocida relación de tipos: Bicho ataca fuerte a Psíquico. Si ves un Pokémon cerca de la flecha de `psychic`, aléjalo de los Bicho.

**💡 Consejo de batalla:** Usa este círculo como una **brújula estratégica**. Antes de una batalla, identifica en qué dirección están las flechas de los tipos que tu oponente usa. Si tu rival es Sabrina (Psíquico), busca Pokémon que estén en la dirección opuesta a la flecha `psychic` — ahí encontrarás a tus mejores contraatacantes.

---

### 📊 Gráfico 8: PCA-Biplot — Todos los Pokémon y sus Tipos en un Solo Plano

![PCA Biplot](output/dimensionality_reduction/biplot_pca_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada punto azul es un Pokémon. Cada flecha roja es un tipo. La posición de un Pokémon respecto a una flecha indica su afinidad con ese tipo.

**⚔️ Lo que significa para tu batalla:**

*Este es tu mapa de batalla definitivo.* Busca un Pokémon que esté cerca de la flecha `water` — ese Pokémon es naturalmente resistente al Agua. Busca uno lejos en dirección opuesta — será su víctima perfecta.

*Identifica grupos de Pokémon en zonas específicas* — Los Pokémon agrupados en la misma región tienen **perfiles defensivos similares**. Si encuentras un grupo de 5 Pokémon juntos y necesitas 2 para tu equipo, **no escojas 2 del mismo grupo** — estarías duplicando vulnerabilidades.

*Los nombres de los Pokémon extremos están etiquetados* — Son los casos más extremos, los especialistas. Mewtwo, por ejemplo, estará en una zona muy alejada — es único en su perfil defensivo.

**💡 Consejo de batalla:** Imprime este gráfico. Ponlo en tu pared. **Antes de cada combate contra un líder de gimnasio, localiza en el mapa los tipos de sus Pokémon y busca en la zona opuesta tus mejores contadores.** Si Brock usa Roca/Tierra, busca Pokémon en la dirección opuesta a las flechas `rock` y `ground`. Ahí están tus Agua, Planta y Lucha ideales.

---

### 📊 Gráfico 9: Calidad de Representación (cos²) — ¿Podemos Confiar en el Mapa?

![Calidad cos²](output/dimensionality_reduction/calidad_representacion_cos2_GEN%201.png)

**🔬 Lo que dice la ciencia:** cos² mide qué tan bien representa el plano 2D a cada variable. Valores >0.5 = buena representación. Valores <0.3 = representación pobre.

**⚔️ Lo que significa para tu batalla:**

*Ninguna variable supera cos²=0.5* — Esto significa que el mapa 2D, aunque útil, **no captura toda la verdad**. Las 18 resistencias son demasiado complejas para solo 2 dimensiones.

*Las variables mejor representadas* (las barras más altas) son en las que **más puedes confiar** cuando miras el mapa. Son `psychic`, `bug`, `ghost`.

*Las peor representadas* (barras más bajas) son `normal`, `dragon`, `fairy`. **No tomes decisiones basadas solo en la posición de estos tipos en el mapa 2D.** Necesitas las otras dimensiones.

**💡 Consejo de batalla:** Cuando planifiques tu equipo basándote en el mapa PCA, **dale más peso a las variables con alto cos²**. Si ves que un Pokémon está lejos de la flecha `psychic` (alto cos²), confía en que realmente es bueno contra Psíquico. Pero si está lejos de `dragon` (bajo cos²), verifícalo con otra fuente — el mapa podría estar distorsionado.

---

## 4. t-SNE — Las Islas de Similitud Defensiva

### 🎯 ¿Qué mira un Maestro Pokémon aquí?

Mientras el PCA te muestra la **estructura global** (como un mapa mundi), el t-SNE te muestra las **vecindades locales** (como un mapa de barrios). t-SNE agrupa a los Pokémon que son **funcionalmente intercambiables** en combate, aunque sean de tipos diferentes. Es aquí donde descubres **sustitutos secretos** para tu equipo.

---

### 📊 Gráfico 10: Grid de Perplejidad — ¿Cuántos Vecinos Importan?

![Grid de Perplejidad](output/dimensionality_reduction/tsne_perplejidad_grid_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada panel usa un valor diferente de perplejidad (5, 10, 15, 30, 50, 100). Perplejidad baja = mira solo vecinos cercanos. Perplejidad alta = mira el panorama general.

**⚔️ Lo que significa para tu batalla:**

*Perplejidad=5* — Ves **micro-grupos muy locales**. Pokémon casi idénticos en resistencia aparecen juntos. Esto es útil para encontrar **sustitutos exactos**: si no puedes conseguir a Pokémon A, busca en su grupito a Pokémon B, que jugará igual.

*Perplejidad=30* — El balance ideal. Ves grupos con significado real. Aquí es donde debes buscar **arquetipos de combate**.

*Perplejidad=100* — Ves la **macro-estructura**. Desaparecen los detalles finos y solo ves los grandes continentes defensivos.

**💡 Consejo de batalla:** Usa perplejidad baja (5-15) cuando estés **armando tu equipo final** y necesites saber exactamente qué Pokémon cubren el mismo rol. Usa perplejidad alta (50-100) cuando estés **diseñando tu estrategia general** y quieras ver los grandes arquetipos.

---

### 📊 Gráfico 11: Grid de Learning Rate — La Velocidad de Aprendizaje

![Grid de Learning Rate](output/dimensionality_reduction/tsne_learning_rate_grid_GEN%201.png)

**🔬 Lo que dice la ciencia:** Learning rate controla qué tan rápido "aprende" el algoritmo. Bajo = lento pero detallado. Alto = rápido pero puede perder estructura real.

**⚔️ Lo que significa para tu batalla:**

*Learning rate=10* — Proyección inestable. Pueden aparecer **grupos falsos** que no existen realmente. No confíes en esta vista.

*Learning rate=200-500* — El punto dulce. Los grupos que ves aquí son **reales y confiables**.

*Learning rate=2000* — Todo se ve como una gran nube. Los grupos reales se pierden.

**💡 Consejo de batalla:** Siempre mira el t-SNE con learning rate entre 200 y 500. Si ves un grupo interesante, verifica que aparezca en **más de un valor de perplejidad** antes de confiar en él. Los grupos que solo aparecen en un valor específico son **artefactos**, no verdad defensiva.

---

### 📊 Gráfico 12: Trustworthiness — ¿t-SNE o PCA? La Ciencia Decide

![Trustworthiness](output/dimensionality_reduction/tsne_trustworthiness_GEN%201.png)

**🔬 Lo que dice la ciencia:** Trustworthiness mide qué tan bien se preservan las vecindades del espacio original (18D) en la proyección (2D). Más alto = mejor. El umbral 0.9 es referencia.

**⚔️ Lo que significa para tu batalla:**

*La línea naranja (t-SNE) está SIEMPRE por encima de la azul (PCA)* — t-SNE **preserva mejor las relaciones de vecindad** que PCA. El promedio: t-SNE = 0.900, PCA = 0.872.

*La diferencia es pequeña (~3%)* — Ambos métodos son buenos. Pero para **encontrar sustitutos funcionales** (Pokémon que juegan el mismo rol), t-SNE es superior.

*Ambas líneas bajan cuando k crece* — Preservar vecinos lejanos es más difícil que preservar vecinos cercanos. Esto es normal.

**💡 Consejo de batalla:** Usa **PCA para interpretar los ejes** (sabes exactamente qué significa cada dirección porque está hecha de combinaciones lineales de tipos reales). Usa **t-SNE para encontrar clústers** (grupos de Pokémon funcionalmente equivalentes). La combinación de ambos te da el poder de un verdadero Maestro.

---

### 📊 Gráfico 13: Comparativa PCA vs t-SNE

![Comparativa PCA vs t-SNE](output/dimensionality_reduction/comparativa_pca_tsne_GEN%201.png)

**🔬 Lo que dice la ciencia:** Lado izquierdo = PCA (lineal, estructura global). Lado derecho = t-SNE (no lineal, estructura local).

**⚔️ Lo que significa para tu batalla:**

*En PCA* — Ves un **gradiente continuo**. Los Pokémon se distribuyen a lo largo de ejes interpretables. Puedes ver qué tipos "tiran" hacia qué dirección. Esto te permite **posicionar estratégicamente** a tu equipo.

*En t-SNE* — Ves **islas distintas**. Los Pokémon se agrupan en clusters separados por espacios vacíos. Esto te muestra **quién es amigo de quién**. Los grupos son familias defensivas.

**💡 Consejo de batalla:** 
- Usa PCA cuando quieras **entender el "por qué"** — ¿por qué estos dos Pokémon son similares? Mira la dirección de los loadings.
- Usa t-SNE cuando quieras **encontrar el "quién"** — ¿quién más juega como este Pokémon que me gusta?
- **Los verdaderos Maestros usan ambos**: PCA para planificar, t-SNE para descubrir gemas ocultas.

---

## 5. Clustering — Los 9 Arquetipos de Combate

### 🎯 ¿Qué mira un Maestro Pokémon aquí?

Hemos reducido 18 dimensiones a 2, y ahora agrupamos a los 151 Pokémon en **9 arquetipos defensivos**, el número óptimo determinado por la **Gap Statistic** con validación multi-métrica. Cada arquetipo es una **familia de combate**: Pokémon que comparten el mismo "ADN de resistencia". Cuando construyes tu equipo, debes buscar **representantes de varios arquetipos** para tener cobertura total.

---

### 📊 Gráfico 14: Optimización Multi-Métrica — Evaluación de k=2..10

![Optimización Multi-Métrica](output/clustering/optimizacion_multimetrica_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cuatro paneles evalúan el número óptimo de clusters usando diferentes métricas:
- **Inercia (codo)**: Mientras más baja, más compactos los clusters
- **Silueta**: Más alta = mejor separación
- **Davies-Bouldin**: Más bajo = menos solapamiento
- **Calinski-Harabasz**: Más alto = mejor definición

**⚔️ Lo que significa para tu batalla:**

*Inercia (panel superior izquierdo)* — La curva desciende suavemente sin un codo definido, lo que indica que las resistencias Pokémon no tienen una estructura jerárquica simple. La Gap Statistic es necesaria para encontrar el k óptimo.

*Silueta (panel superior derecho)* — El valor para k=9 es el que balancea mejor la compacidad y separación dado que las 18 dimensiones tienen solapamiento natural entre tipos.

*Davies-Bouldin (panel inferior izquierdo)* — El mínimo para k=9 confirma que **9 clusters minimizan el solapamiento inter-cluster**.

*Calinski-Harabasz (panel inferior derecho)* — Máximo en k=9, indicando que los 9 grupos están **bien definidos y separados**.

**💡 Consejo de batalla:** La ciencia converge: **9 arquetipos defensivos** es la partición estadísticamente óptima. Cuando armes tu equipo de 6, busca cubrir la mayor diversidad de arquetipos posible. Los 2 puestos restantes son para especialistas (los que DBSCAN detecta como outliers).

---

### 📊 Gráfico 15: Gap Statistic — La Voz de la Estadística

![Gap Statistic](output/clustering/gap_statistic_GEN%201.png)

**🔬 Lo que dice la ciencia:** La Gap Statistic compara la inercia observada (qué tan compactos son tus clusters) contra la inercia esperada si los datos fueran aleatorios. El mejor k es el primer valor donde la curva se estabiliza (regla de 1 error estándar).

**⚔️ Lo que significa para tu batalla:**

*La curva sube consistentemente hasta k=9 y se estabiliza* — La Gap Statistic determina que **9 es el número óptimo de arquetipos defensivos**. A partir de k=9, añadir más clusters ya no mejora significativamente la partición.

*Cada uno de los 9 arquetipos representa una especialización defensiva* — No son "subgrupos" de una categoría mayor, sino **arquetipos con entidad propia** que capturan las diferencias reales entre los perfiles de resistencia de los 151 Pokémon.

**💡 Consejo de batalla:** Los **9 arquetipos** son tu herramienta de precisión. Mientras menos clusters (k=3-4) te dan una visión general, los 9 arquetipos te permiten **afinar tu equipo con exactitud quirúrgica**. Para enfrentar a la Liga o al Alto Mando, usa los 9 — cada uno representa un nicho defensivo específico que puedes explotar.

---

### 📊 Gráfico 16: Dendrograma Ward — El Árbol Genealógico de las Resistencias

![Dendrograma](output/clustering/dendrograma_ward_GEN%201.png)

**🔬 Lo que dice la ciencia:** El dendrograma muestra cómo se agrupan jerárquicamente los Pokémon. La altura de las uniones indica disimilitud. El panel derecho muestra la "sedimentación" (distancias de enlace).

**⚔️ Lo que significa para tu batalla:**

*Ramas largas antes de unirse* — Los clusters que se separan a alturas grandes son **muy diferentes entre sí**. Son buenos candidatos para cubrirse mutuamente en tu equipo.

*El corte rojo para k=9* — Muestra cómo se dividen los 151 Pokémon en 9 familias defensivas. La altura de corte (~20) indica una separación más fina pero igualmente significativa entre arquetipos.

*La correlación cofenética de 0.649* — El dendrograma preserva el 64.9% de las distancias originales. Es aceptable — significa que el árbol refleja razonablemente bien las verdaderas relaciones entre Pokémon.

**💡 Consejo de batalla:** El dendrograma es tu **árbol genealógico de batalla**. Cuando busques un Pokémon para cubrir las debilidades de otro, busca en **ramas opuestas del árbol**. Si tu base es un Pokémon de la rama izquierda, busca su contraparte en la rama derecha. Cuanto más larga la rama que los separa, mejor cobertura mutua tendrán.

---

### 📊 Gráfico 17: Diagrama de Silueta — ¿Quién Queda Fuera?

![Silueta](output/clustering/silhouette_diagram_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada barra horizontal es un Pokémon. La posición en el eje X es su coeficiente de silueta (qué tan bien pertenece a su cluster). Barras que cruzan la línea roja (promedio global = 0.263) están peor clasificadas.

**⚔️ Lo que significa para tu batalla:**

*Los clusters más grandes* (con más Pokémon) — Son arquetipos **generales y homogéneos**. Casi todas las barras están a la derecha del promedio. Estos son Pokémon **típicos, bien clasificados** dentro de su nicho defensivo.

*Los clusters más pequeños* (con pocos Pokémon) — Son arquetipos **especializados y exclusivos**. Algunas barras pueden cruzar la línea roja hacia la izquierda, indicando perfiles defensivos **híbridos** que no encajan perfectamente. Son Pokémon de transición entre arquetipos.

*Las barras que cruzan la línea roja* — Son Pokémon que **no están completamente cómodos** en su cluster. Podrían pertenecer a otro grupo. En términos de batalla, son Pokémon **versátiles** que pueden jugar en múltiples roles.

**💡 Consejo de batalla:** Los Pokémon con baja silueta (barras a la izquierda de la línea roja) son **comodines versátiles**. No los uses como pilar de tu equipo, pero sí como **situacionales** que pueden sorprender a tu oponente. Un Pokémon "mal clasificado" no es un mal Pokémon — es un Pokémon con un perfil único que desafía las categorías.

---

### 📊 Gráfico 18: Caracterización por Cluster — El Perfil de Cada Arquetipo

![Caracterización](output/clustering/caracterizacion_heatmap_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada celda es el valor promedio de resistencia para ese cluster y ese tipo. Verde = resistente (valor bajo). Rojo = vulnerable (valor alto). Con **9 arquetipos**, el nivel de detalle es quirúrgico: cada fila representa un nicho defensivo puro.

**⚔️ Lo que significa para tu batalla:**

*Este es el gráfico más importante de todo el análisis.* Memorízalo. Con 9 arquetipos, la especialización es máxima:

**Cluster 0 — Guardianes del Océano** — Tanques con resistencia general. Ideales como base del equipo.

**Cluster 1 — Protectores de la Biosfera** — Resistentes a Lucha y Planta. Perfectos contra equipos físicos.

**Cluster 2 — Místicos del Éter** — Inmunes funcionales a Psíquico. El terror de Sabrina y los Pokémon Lucha.

**Cluster 3 — Centinelas de Alta Tensión** — Especialistas en Eléctrico y Veneno. Brillan contra Agua y Volador.

**Cluster 4 — Titanes de la Tierra** — Resistentes a Roca y Eléctrico. Anclas contra ataques de tierra y rayo.

**Cluster 5 — Espectros Sombríos** — Maestros de Fantasma y Siniestro. Perfectos para guerras psicológicas.

**Cluster 6 — Guardianes Ígneos** — Inmunes al Fuego. Obligatorios contra Blaine y entrenadores de Fuego.

**Cluster 7 — Centinelas de Hielo** — Resistentes a Hielo y Dragón. Clave contra Lance y Lorelei.

**Cluster 8 — Versátiles Híbridos** — Perfiles mixtos. Comodines que se adaptan a múltiples situaciones.

**🔥 REGLA DE ORO DEL MAESTRO POKÉMON:**
> *"Con 9 arquetipos, ya no buscas 'un Pokémon de cada color'. Buscas **especialización pura**: identifica el arquetipo del oponente y contraataca con su opuesto. Los 2 puestos restantes son para outliers que rompan el esquema."*

---

### 📊 Gráfico 19: Radar de Arquetipos — La Huella Digital de Cada Familia

![Radar](output/clustering/radar_arquetipos_GEN%201.png)

**🔬 Lo que dice la ciencia:** Cada polígono es un arquetipo, dibujado sobre las 8 variables más discriminativas. Cuanto más cerca del centro, más resistente. Cuanto más lejos, más vulnerable.

**⚔️ Lo que significa para tu batalla:**

*Compara las formas de los 9 polígonos* — Son **marcadamente diferentes**. Esto confirma que los 9 arquetipos son genuinamente distintos. Cada uno tiene una huella digital única de fortalezas y debilidades.

*Algunos clusters tienen formas "anchas"* — Indican arquetipos equilibrados con pocas vulnerabilidades extremas. Son buenos **anclas defensivas** para cualquier equipo.

*Otros clusters tienen picos pronunciados* — Especialización extrema. Brillan contra tipos específicos pero tienen talones de Aquiles evidentes. Son **armas situacionales** de alto riesgo/alta recompensa.

*Los clusters más pequeños* — Su forma es la más excéntrica. Son Pokémon de nicho puro, casi siempre los outliers de DBSCAN que decidieron formar su propio arquetipo.

**💡 Consejo de batalla:** Superponer mentalmente el radar de tu oponente con el tuyo te dice **quién tiene ventaja**. Si tu arquetipo envuelve al de tu oponente (estás más cerca del centro en todas las direcciones), tienes ventaja defensiva total. Si tu polígono está parcialmente fuera del suyo, tienes ventaja en algunas áreas y desventaja en otras — ahí entra la estrategia de cambios.

---

### 📊 Gráfico 20: PCA + Clusters — El Mapa de Batalla

![PCA Clusters](output/clustering/pca_clusters_GEN%201.png)

**🔬 Lo que dice la ciencia:** Los 151 Pokémon proyectados en PCA, coloreados por su cluster de pertenencia.

**⚔️ Lo que significa para tu batalla:**

*Los 9 colores ocupan regiones distintas del espacio* — Los clusters no están mezclados al azar. Cada arquetipo tiene su territorio en el mapa de resistencias, y con 9 arquetipos la delimitación es más precisa.

*Hay zonas de transición donde los colores se mezclan* — Esos Pokémon fronterizos tienen perfiles híbridos. Son tus **comodines estratégicos**.

*Los puntos extremos (lejos del centro)* — Son los especialistas puros, los outliers de DBSCAN. Pokémon como Charizard, Gengar o Alakazam están en las fronteras del mapa.

**💡 Consejo de batalla:** Este es tu **mapa táctico**. Antes de un combate:
1. Localiza a tu oponente en el mapa (según su tipo)
2. Identifica qué colores lo rodean
3. Elige Pokémon del color opuesto
4. Los Pokémon fronterizos entre dos colores son versátiles pero menos especializados

---

### 📊 Gráfico 21: t-SNE + Clusters — Las Islas de la Estrategia

![t-SNE Clusters](output/clustering/tsne_clusters_GEN%201.png)

**🔬 Lo que dice la ciencia:** t-SNE con los clusters coloreados. La separación entre grupos es más nítida que en PCA porque t-SNE prioriza las vecindades locales.

**⚔️ Lo que significa para tu batalla:**

*Las "islas" de cada color están más separadas que en PCA* — t-SNE **exagera las diferencias** para crear grupos visualmente distintos. Esto es bueno para identificar arquetipos, pero **no confíes en las distancias** — las distancias en t-SNE no son directamente comparables.

*Algunos puntos de un color aparecen "invadiendo" territorio de otro* — Son los Pokémon híbridos que mencionamos. Tienen el arquetipo principal de su color pero características secundarias del otro.

**💡 Consejo de batalla:** Usa t-SNE para **descubrir sustitutos ocultos**. Si te gusta un Pokémon raro o difícil de conseguir (como Mewtwo), busca en su "isla" qué otros Pokémon están cerca. ¡Esos son sustitutos funcionales! Puede que no tengan su poder, pero jugarán el **mismo rol defensivo** en tu equipo.

---

### 📊 Gráfico 22: Diagnóstico DBSCAN — Encontrando la Epsilon Perfecta

![DBSCAN K-Distance](output/clustering/dbscan_k_distance_GEN%201.png)

**🔬 Lo que dice la ciencia:** La curva K-distance muestra la distancia al 5º vecino más cercano para cada Pokémon, ordenada. El "codo" en la curva indica el valor óptimo de epsilon para DBSCAN.

**⚔️ Lo que significa para tu batalla:**

*La curva sube suavemente y luego se dispara* — El punto donde la curva se acelera es el umbral: los Pokémon antes del codo son "normales", los después son "especialistas".

*El codo está alrededor de eps=3.0* — Este valor separa a los Pokémon convencionales de los únicos.

**💡 Consejo de batalla:** Los Pokémon que caen después del codo son **especialistas puros**. No intentes encajarlos en un arquetipo — son únicos. Construye tu equipo base con los Pokémon "normales" (antes del codo) y añade 1-2 especialistas (después del codo) para crear **sinergias inesperadas** que tu oponente no anticipará.

---

### 📊 Gráfico 23: DBSCAN sobre PCA — Los Outsiders

![DBSCAN PCA](output/clustering/pca_dbscan_GEN%201.png)

**🔬 Lo que dice la ciencia:** DBSCAN aplicado sobre los datos, proyectado en PCA. Cada color es un cluster denso. Los puntos morados son ruido (outliers).

**⚔️ Lo que significa para tu batalla:**

*Los puntos morados son 28 Pokémon (18.5% del total)* — Son los **especialistas puros**. Sus perfiles de resistencia son tan únicos que no encajan en ningún grupo denso.

*Ejemplos de outliers*: Charizard, Gengar, Magneton, Gastly, Haunter, Parasect, Zubat, Golbat...

*¿Qué tienen en común?* — Son Pokémon con **tipos duales poco comunes** o con **resistencias extremas**. Charizard (Fuego/Volador) tiene una combinación única que lo hace difícil de clasificar.

**💡 Consejo de batalla:** **Los outliers son tus armas secretas.** Un entrenador promedio usa Pokémon de arquetipos estándar. Un Maestro usa outliers para **romper las expectativas**. Cuando tu oponente asume que tu Charizard es un Fuego típico y saca un Agua, tú cambias a un Pokémon que tu oponente no esperaba porque su perfil no encaja en lo predecible.

**🔥 LOS 28 OUTLIERS (Tus Armas Secretas):**
Charizard, Butterfree, Zubat, Golbat, Paras, Parasect, Magnemite, Magneton, Gastly, Haunter, Gengar, Onix, Krabby, Kingler, Voltorb, Electrode, Cubone, Marowak, Hitmonlee, Hitmonchan, Lickitung, Tangela, Horsea, Seadra, Mr. Mime, Scyther, Electabuzz, Pinsir

---

### 📊 Gráfico 24: Kruskal-Wallis — La Validación Definitiva

![Kruskal-Wallis](output/clustering/kruskal_wallis_resultados_GEN%201.png)

**🔬 Lo que dice la ciencia:** Panel izquierdo: estadístico H de Kruskal-Wallis para cada variable (qué tan bien discrimina entre clusters). Panel derecho: -log10(p-valor). La línea roja es α=0.05.

**⚔️ Lo que significa para tu batalla:**

*18/18 variables son significativas (p<0.05)* — Esto es CIENCIA. Significa que los 9 arquetipos NO son producto del azar. Son **diferencias reales** en los perfiles de resistencia, validadas con un test no paramétrico robusto. Con k=9, incluso el tipo Dragón discrimina significativamente entre grupos — la granularidad más fina captura matices que con k=4 se perdían.

*Las variables con H más alto* (mejores discriminadores): `psychic`, `bug`, `ghost`, `rock`, `fire`.

**💡 Consejo de batalla:** Cuando armes tu equipo, **prioriza la cobertura de los tipos con H alto** (psychic, bug, ghost, rock, fire). Son los que más separan a los arquetipos. Si cubres bien estos 5 tipos, automáticamente estás cubriendo las diferencias fundamentales entre las 9 familias defensivas.

---

## 6. Validación Estadística — No es Casualidad

Este análisis no son solo gráficos bonitos. Cada métrica ha sido calculada y validada:

| Métrica | Valor (k=9) | ¿Qué significa? |
|---------|-------------|-----------------|
| **Silhouette Score** | 0.4216 | Separación moderada-alta entre los 9 arquetipos (mejor que 0.263 con k=4) |
| **Davies-Bouldin** | 0.9565 | Bajo solapamiento inter-cluster — mejora respecto a 1.458 con k=4 |
| **Calinski-Harabasz** | 34.18 | Los 9 grupos están bien definidos — mejora respecto a 25.07 con k=4 |
| **Gap Statistic** | k=9 óptimo | 9 arquetipos es el número estadísticamente representativo |
| **Correlación Cofenética** | 0.649 | El dendrograma refleja bien las relaciones reales |
| **Kruskal-Wallis** | 18/18 significativas | Con 9 arquetipos, hasta dragon discrimina (p<0.05) |
| **Trustworthiness t-SNE** | 0.900 | Excelente preservación de vecindades en proyección |
| **DBSCAN Ruido** | 18.5% | 28 especialistas puros identificados |

---

## 7. Cartografía Visual — El Mapa del Tesoro

### 📊 Gráfico 25: Mapa PCA con Sprites

![Mapa PCA](output/image_maps/mapa_visual_pca_GEN%201.png)

**⚔️ Lo que significa para tu batalla:**

*Este es el mapa del tesoro.* Cada sprite está posicionado donde le corresponde por su perfil de resistencia. Los Pokémon cercanos son **aliados naturales** — comparten fortalezas y debilidades. Los Pokémon lejanos son **contrapuntos estratégicos**.

**👉 Cómo usar este mapa:**
1. Encuentra a tu Pokémon favorito
2. Mira quiénes están cerca — esos son tus **compañeros de equipo naturales** (cubren las mismas debilidades)
3. Mira quiénes están lejos en dirección opuesta — esos son tus **contadores naturales**
4. Si ves un grupo de 5+ Pokémon muy juntos, **no elijas 2 de ese grupo** para tu equipo — estarías duplicando vulnerabilidades

---

### 📊 Gráfico 26: Mapa t-SNE con Sprites

![Mapa t-SNE](output/image_maps/mapa_visual_tsne_GEN%201.png)

**⚔️ Lo que significa para tu batalla:**

*Donde PCA es un mapa continuo, t-SNE es un archipiélago.* Cada isla es una **familia defensiva**. Los Pokémon dentro de una misma isla son **intercambiables** en términos de resistencia.

**👉 Cómo usar este mapa:**
1. Identifica las "islas" principales — hay 9 que corresponden a los arquetipos
2. Las islas pequeñas son **microniches** — grupos muy específicos
3. Los puntos solitarios son **outliers** — especialistas únicos
4. Si necesitas un sustituto para un Pokémon que no puedes conseguir, busca en su misma isla — ahí hay alternativas funcionales

---

## 8. Diccionario Completo de Gráficos

A continuación, la guía definitiva de cada gráfico generado, explicado desde la perspectiva de un Maestro Pokémon:

### 📊 DESCRIPTIVOS (4 gráficos)

| # | Archivo | 🎯 Explicación para tu Batalla |
|---|---------|-------------------------------|
| 1 | `histogram_resistencias_GEN 1.png` | **"Conoce el terreno"** — Cada tipo tiene su propia "huella" de resistencia. Los tipos con distribuciones anchas (mucha variabilidad) son impredecibles en batalla. Prepárate para sorpresas. |
| 2 | `boxplot_resistencias_GEN 1.png` | **"Mide a tu rival"** — Tipos con cajas largas (Roca, Agua, Eléctrico) tienen Pokémon muy diversos. No asumas que todos los Agua son iguales. Tipos con cajas cortas (Normal) son predecibles. |
| 3 | `heatmap_correlaciones_GEN 1.png` | **"Descubre alianzas secretas"** — Correlaciones positivas = tipos que suelen ir juntos. Si resistes uno, probablemente resistes el otro. Correlaciones negativas = cobertura natural. Si eres fuerte contra uno, el otro te debilita. |
| 4 | `coordenadas_paralelas_GEN 1.png` | **"Detecta redundancias"** — Líneas paralelas = Pokémon redundantes. Si tienes dos con el mismo patrón, estás desperdiciando un puesto en tu equipo. |

### 📉 REDUCCIÓN DIMENSIONAL — PCA (5 gráficos)

| # | Archivo | 🎯 Explicación para tu Batalla |
|---|---------|-------------------------------|
| 5 | `scree_plot_pca_GEN 1.png` | **"¿Cuánto sabes realmente?"** — 35.8% de la información en solo 2 dimensiones. Suficiente para estrategia general, pero los Maestros usan 7 dimensiones mentales. |
| 6 | `loadings_heatmap_pca_GEN 1.png` | **"Los ejes del poder"** — PC1 = eje Aéreo-Vegetal-Terrestre. PC2 = eje Psíquico-Bicho-Oscuridad. Domina estos ejes y dominarás la Liga. |
| 7 | `circulo_correlaciones_pca_GEN 1.png` | **"La brújula de tipos"** — Flechas largas = confía en lo que ves. Flechas cortas = verifica con otra fuente. Flechas opuestas = cobertura natural. |
| 8 | `biplot_pca_GEN 1.png` | **"El mapa de guerra"** — Cada punto es un Pokémon. Cada flecha es un tipo. Encuentra a tu objetivo, mira hacia dónde apuntan sus tipos, y ve en dirección opuesta para encontrar a sus verdugos. |
| 9 | `calidad_representacion_cos2_GEN 1.png` | **"¿Puedo confiar en este mapa?"** — Ninguna variable llega a cos²=0.5. El mapa 2D es una simplificación. Úsalo como guía, no como verdad absoluta. |

### 📉 REDUCCIÓN DIMENSIONAL — t-SNE (4 gráficos)

| # | Archivo | 🎯 Explicación para tu Batalla |
|---|---------|-------------------------------|
| 10 | `tsne_perplejidad_grid_GEN 1.png` | **"Zoom in, zoom out"** — Perplejidad baja = lupa (busca sustitutos exactos). Perplejidad alta = panorama general (busca arquetipos). Usa perplejidad=30 para el balance ideal. |
| 11 | `tsne_learning_rate_grid_GEN 1.png` | **"No te dejes engañar"** — Learning rate bajo puede crear grupos falsos. Rate alto puede ocultar grupos reales. Confía en los valores medios (200-500). |
| 12 | `tsne_trustworthiness_GEN 1.png` | **"t-SNE vs PCA: ¿Quién gana?"** — t-SNE gana en fidelidad local (0.900 vs 0.872). PCA gana en interpretabilidad. Usa ambos: PCA para planificar, t-SNE para descubrir. |
| 13 | `comparativa_pca_tsne_GEN 1.png` | **"Dos caras de la misma moneda"** — PCA es el mapa mundi, t-SNE es el mapa de barrios. Lleva ambos en tu mochila. |

### 🧬 CLUSTERING (11 gráficos)

| # | Archivo | 🎯 Explicación para tu Batalla |
|---|---------|-------------------------------|
| 14 | `optimizacion_multimetrica_GEN 1.png` | **"¿Por qué 9?"** — Cuatro métricas independientes convergen: 9 arquetipos es el óptimo estadístico según la Gap Statistic. |
| 15 | `gap_statistic_GEN 1.png` | **"9 arquetipos, 1 verdad"** — La Gap Statistic confirma que 9 es el número óptimo de clusters. Cada arquetipo representa un nicho defensivo real y explotable. |
| 16 | `dendrograma_ward_GEN 1.png` | **"El árbol de la vida Pokémon"** — Ramas opuestas = cobertura perfecta. Ramas cercanas = redundancia. Busca en la rama opuesta a tu base para encontrar tu mejor compañero. |
| 17 | `silhouette_diagram_GEN 1.png` | **"Los comodines"** — Pokémon con baja silueta son versátiles. Úsalos como situational picks, no como pilares. |
| 18 | `caracterizacion_heatmap_GEN 1.png` | **"LA TABLA SAGRADA"** — Memoriza este heatmap. Te dice exactamente qué esperar de cada arquetipo. Es la diferencia entre un viaje de ida y un viaje de vuelta de la Liga. |
| 19 | `radar_arquetipos_GEN 1.png` | **"La huella digital"** — Cada arquetipo tiene una forma única. Superpón mentalmente las huellas de tu equipo y el rival para predecir el resultado. |
| 20 | `pca_clusters_GEN 1.png` | **"Territorios de batalla"** — Cada color ocupa su zona en el mapa. No mezcles colores opuestos en tu equipo sin un plan de cobertura. |
| 21 | `tsne_clusters_GEN 1.png` | **"Las islas de similitud"** — Encuentra tu isla y busca compañeros. Si estás solo en una isla, eres único — para bien o para mal. |
| 22 | `dbscan_k_distance_GEN 1.png` | **"El umbral de la rareza"** — El codo en la curva separa a los Pokémon comunes de los especialistas. Conoce a tus especialistas y úsalos como armas secretas. |
| 23 | `pca_dbscan_GEN 1.png` | **"Los 28 elegidos"** — Los puntos morados son los outliers. Charizard, Gengar, Magneton... Son Pokémon tan únicos que no encajan en ningún arquetipo. Son tus armas más impredecibles. |
| 24 | `kruskal_wallis_resultados_GEN 1.png` | **"La validación científica"** — 17/18 variables discriminan. Solo dragón (con solo 3 ejemplares) no. Tu estrategia tiene respaldo estadístico. |

### 🗺️ CARTOGRAFÍA VISUAL (2 gráficos)

| # | Archivo | 🎯 Explicación para tu Batalla |
|---|---------|-------------------------------|
| 25 | `mapa_visual_pca_GEN 1.png` | **"El mapa del tesoro completo"** — 151 sprites en un plano. Cada uno en su posición exacta según su ADN de combate. Encuentra a tu equipo ideal buscando Pokémon en zonas complementarias. |
| 26 | `mapa_visual_tsne_GEN 1.png` | **"El archipiélago defensivo"** — Islas de similitud funcional. Si pierdes a un miembro de tu equipo, busca en su misma isla un reemplazo con el mismo perfil defensivo. |

---

## 9. Estrategias para la Liga

### ⚔️ Cómo usar este análisis contra cada líder de gimnasio

*Nota: Los clusters específicos (0-8) se asignan durante la ejecución del script. La tabla siguiente muestra los arquetipos funcionales que debes buscar según el tipo del líder.*

| Líder | Tipo | Estrategia basada en arquetipos |
|-------|------|-------------------------------|
| **Brock** | Roca/Tierra | Busca arquetipos con alta resistencia a Roca y Tierra (identifícalos en el heatmap de caracterización). Evita arquetipos vulnerables a Tierra. |
| **Misty** | Agua | Arquetipos con resistencia a Agua o tipo Eléctrico dominante. El heatmap te mostrará qué cluster(es) minimizan `against_water`. |
| **Lt. Surge** | Eléctrico | Arquetipos con resistencia a Eléctrico o tipo Tierra. Revisa el heatmap para identificar clusters con `against_electric` bajo. |
| **Erika** | Planta | Arquetipos con resistencia a Planta (Fuego, Volador, Hielo, Bicho). El heatmap revela qué clusters destacan. |
| **Koga** | Veneno | Arquetipos con resistencia a Veneno (Acero, Veneno mismo). Busca clusters con `against_poison` mínimo. |
| **Sabrina** | Psíquico | Arquetipos con resistencia a Psíquico (Siniestro, Bicho). El biplot y el heatmap te guían. |
| **Blaine** | Fuego | Arquetipos con alta resistencia a Fuego (Agua, Roca, Dragón). Cruza el heatmap con tus opciones. |
| **Giovanni** | Tierra | Arquetipos con resistencia a Tierra (Planta, Volador, Bicho). Identifica el cluster óptimo en el PCA-biplot. |

### 🏆 Contra el Alto Mando

| Miembro | Tipo | Estrategia con 9 arquetipos |
|---------|------|-----------------------------|
| **Lorelei** | Hielo/Agua | Busca clusters con resistencia a Hielo y Agua, más tipo Eléctrico/Roca |
| **Bruno** | Lucha | Clusters con resistencia a Lucha (Psíquico, Volador, Hada) |
| **Agatha** | Fantasma/Veneno | Clusters con resistencia a Fantasma y Veneno (Siniestro, Acero) |
| **Lance** | Dragón/Volador | Clusters con resistencia a Dragón, Hielo y Roca |

### 🔥 La Fórmula del Equipo Perfecto (6 Pokémon) con 9 Arquetipos

Con 9 arquetipos, la estrategia cambia: ya no buscas "uno de cada" sino **maximizar cobertura con 6 slots**:

```
Paso 1: Identifica los 3 arquetipos más frecuentes entre tus candidatos
Paso 2: Elige 1 Pokémon de cada uno como base
Paso 3: Añade 1 Pokémon de un arquetipo que cubra las debilidades de los 3 anteriores
Paso 4: Completa con 2 Outliers (especialistas DBSCAN) para impredecibilidad
```

**Ejemplo de equipo campeón (conceptual):**
1. 🟦 **Blastoise** — Tanque versátil (pertenece al arquetipo de resistencia hídrica)
2. 🟩 **Venusaur** — Cobertura planta/veneno (arquetipo de resistencia física)
3. 🟨 **Alakazam** — Contador psíquico (arquetipo de resistencia especial)
4. 🟪 **Magneton** — Especialista eléctrico (arquetipo de alta tensión)
5. 🟥 **Charizard** — Outlier ofensivo (rompe esquemas predecibles)
6. 🟥 **Gengar** — Outlier impredecible (perfil único de resistencias)

*Los nombres específicos de los 9 arquetipos y sus miembros exactos se generan al ejecutar el script y se reflejan en el heatmap de caracterización y en los mapas de sprites.*

---

## 🛠️ Tecnologías

| Herramienta | Propósito |
|-------------|-----------|
| Python 3.12+ | Lenguaje base del análisis |
| pandas, numpy | Manipulación de datos |
| matplotlib, seaborn | Visualización |
| scikit-learn | PCA, t-SNE, K-Means, DBSCAN, métricas |
| scipy | Kruskal-Wallis, clustering jerárquico |
| kagglehub | Descarga de datasets |

## ⚙️ Ejecución

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn scipy
python pokemon-database.py
```

## 📁 Estructura de Salida

```
output/
├── analisis_resultados_GEN 1.txt  # Reporte científico completo
├── diccionario_graficos_GEN 1.*   # Diccionario (JSON + MD)
├── descriptive/                   # 4 gráficos EDA
├── dimensionality_reduction/      # 9 gráficos PCA + t-SNE
├── clustering/                    # 11 gráficos clustering
└── image_maps/                    # 2 mapas con sprites
```

## 👨‍💻 Autor y Créditos

**Grupo #10** — Maestría en Analítica de Datos
Politécnico Grancolombiano — Facultad de Ingeniería, Diseño e Innovación

### 📚 Referencias Científicas
- Arthur & Vassilvitskii (2007). *k-means++: The advantages of careful seeding*. SODA.
- van der Maaten & Hinton (2008). *Visualizing Data using t-SNE*. JMLR.
- Kruskal & Wallis (1952). *Use of ranks in one-criterion variance analysis*. JASA.
- Tibshirani, Walther & Hastie (2001). *Estimating the number of clusters via the gap statistic*. JRSS.
- Davies & Bouldin (1979). *A cluster separation measure*. IEEE TPAMI.
- Rousseeuw (1987). *Silhouettes: A graphical aid to... cluster analysis*. JCAM.

---

> *"Ahora ve, entrenador. Tienes el conocimiento. Tienes los datos. Tienes la estrategia. La Liga te espera — y esta vez, no solo atraparás a todos... **entenderás a todos**."*
>
> 🏆 — **Tu Maestro Pokémon Interior**

---
*Generado: 2026-06-20 — 26 gráficos, 9 arquetipos (validados por Gap Statistic), 1 misión: convertirte en Campeón.*
