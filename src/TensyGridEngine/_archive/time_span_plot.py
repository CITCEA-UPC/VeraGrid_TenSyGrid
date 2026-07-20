import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def generar_grafics_final(nom_fitxer='time_span.md'):
    if not os.path.exists(nom_fitxer):
        print(f"Error: No s'ha trobat el fitxer '{nom_fitxer}'")
        return

    dades = []
    with open(nom_fitxer, 'r', encoding='utf-8') as f:
        for linia in f:
            if '|' in linia and '---' not in linia:
                cols = [c.strip() for c in linia.split('|')]
                cols = [c for c in cols if c != '']

                grid_name = ""
                t_zero = None
                t_pre = None
                state_space = None

                extensions = ['.gridcal', '.xlsx', '.veragrid', '.json', '.xml', '.ods', '.dgs', '.md']

                for i, c in enumerate(cols):
                    if any(ext in c.lower() for ext in extensions):
                        grid_name = c
                        if i + 1 < len(cols):
                            t_zero = pd.to_numeric(cols[i + 1], errors='coerce')
                        if i + 2 < len(cols):
                            t_pre = pd.to_numeric(cols[i + 2], errors='coerce')
                        if i + 3 < len(cols):
                            # La nova columna "State-space dimension" és just després de T_pre
                            state_space = pd.to_numeric(cols[i + 3], errors='coerce')
                        break

                if grid_name:
                    dades.append({'Grid': grid_name, 'T_zero': t_zero, 'T_pre': t_pre, 'State_space': state_space})

    if not dades:
        print("No s'han trobat dades vàlides.")
        return

    df = pd.DataFrame(dades)

    # ---------------------------------------------------------
    # GRÀFIC 1: Barres horitzontals (Comparativa de temps)
    # ---------------------------------------------------------
    # Alçada dinàmica: 0.4 polzades per fila per donar espai a les etiquetes de text
    alcada = max(10, len(df) * 0.4)
    fig1, ax1 = plt.subplots(figsize=(14, alcada))

    y_pos = np.arange(len(df))
    width = 0.35

    # Barres
    barres_zero = ax1.barh(y_pos + width / 2, df['T_zero'], width, label='Temps des de zero (s)', color='red')
    barres_pre = ax1.barh(y_pos - width / 2, df['T_pre'], width, label='Temps precomputat (s)', color='royalblue')

    # Afegir els valors de temps al costat de cada barra
    def afegir_etiquetes(rects, ax):
        for rect in rects:
            w = rect.get_width()
            if not np.isnan(w) and w > 0:  # Només si hi ha dades
                ax.annotate(f'{w:.2f}s',
                            xy=(w, rect.get_y() + rect.get_height() / 2),
                            xytext=(5, 0),  # 5 punts de desplaçament a la dreta
                            textcoords="offset points",
                            ha='left', va='center', fontsize=8)

    afegir_etiquetes(barres_zero, ax1)
    afegir_etiquetes(barres_pre, ax1)

    # Configuració d'eixos
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['Grid'], fontsize=9)
    ax1.invert_yaxis()  # De dalt a baix

    ax1.set_xlabel('Temps (segons)')
    ax1.set_title('Comparativa d\'Execució: time_span.md', pad=20, fontsize=14)

    # Llegenda a la part superior dreta
    ax1.legend(loc='upper right', frameon=True, shadow=True)

    # Grid vertical per ajudar a llegir els segons
    ax1.grid(axis='x', linestyle='--', alpha=0.4)

    # Donar espai extra a la dreta perquè el text dels segons no quedi tallat
    max_val = df[['T_zero', 'T_pre']].max().max()
    if not np.isnan(max_val):
        ax1.set_xlim(0, max_val * 1.15)

    fig1.tight_layout()
    fig1.savefig('grafic_time_span_detallat.png', dpi=300)
    print("Gràfic de barres generat: 'grafic_time_span_detallat.png'")

    # ---------------------------------------------------------
    # GRÀFIC 2: Dispersió (State-space vs T_precomputat)
    # ---------------------------------------------------------
    # Filtrem només les files que tinguin dades en ambdues columnes per al gràfic
    df_scatter = df.dropna(subset=['State_space', 'T_pre'])

    if not df_scatter.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        # Crear el gràfic de dispersió
        ax2.scatter(df_scatter['State_space'], df_scatter['T_pre'],
                    color='mediumseagreen', alpha=0.8, edgecolors='black', s=80)

        # Afegir les etiquetes amb el nom de la xarxa i alpha=0.5
        for idx, row in df_scatter.iterrows():
            ax2.annotate(row['Grid'],
                         (row['State_space'], row['T_pre']),
                         xytext=(5, 5),
                         textcoords='offset points',
                         alpha=0.5,
                         fontsize=8)

        # Configuració d'eixos i títol
        ax2.set_xlabel('Dimensió del State-Space (nre. variables)', fontsize=12)
        ax2.set_ylabel('Temps precomputat [builder guardat] (segons)', fontsize=12)
        ax2.set_title('Impacte del State-Space en el Temps d\'Execució', pad=20, fontsize=14)

        # Grid
        ax2.grid(True, linestyle='--', alpha=0.6)

        fig2.tight_layout()
        fig2.savefig('grafic_dispersio_statespace.png', dpi=300)
        print("Gràfic de dispersió generat: 'grafic_dispersio_statespace.png'")
    else:
        print(
            "Avís: No s'han trobat dades vàlides per generar el gràfic de dispersió (falten valors a 'State-space' o 'T_pre').")

    # Mostrar tots dos gràfics
    plt.show()


# Execució
generar_grafics_final('time_span.md')