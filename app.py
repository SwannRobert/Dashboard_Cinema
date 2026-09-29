import streamlit as st
import pandas as pd
import kagglehub
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Dashboard Cinéma", layout="wide")

# Style pour barre de gauche en bleu foncé
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #2c3e50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def charger_donnees():
    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    fichiers = os.listdir(path)
    fichier_csv = next((f for f in fichiers if f.endswith('.csv')), "")
            
    path = os.path.join(path, fichier_csv)
    
    # Utilisation du moteur pyarrow pour une lecture ultra-rapide
    df = pd.read_csv(path, engine='pyarrow')
    
    # On enlève les données qui faussent les résultats
    df = df[df['status']== 'Released']
    df = df[df['vote_count']>= 15]
    
    # Spécification explicite du traitement des dates pour accélérer la conversion
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    df['year'] = df['release_date'].dt.year.astype(int)
    
    df['genres'] = df['genres'].fillna('Inconnu')
    df['runtime'] = df['runtime'].fillna(0)
    df['production_companies'] = df['production_companies'].fillna('Inconnu')
    
    # Pré-calculer les genres une seule fois et les stocker en cache
    liste_genres = set()
    for g in df['genres'].unique():
        if isinstance(g, str):
            liste_genres.update(g.split(', '))
    
    genres_tries = sorted(list(liste_genres))
    
    return df, genres_tries


# Mise en cache des calculs lourds de machine learning
@st.cache_resource
def calculer_tfidf(df_reco):
    df_reco['overview'] = df_reco['overview'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    matrice_tfidf = tfidf.fit_transform(df_reco['overview'])
    return tfidf, matrice_tfidf


# Chargement initial
df, genres_tries = charger_donnees()

st.sidebar.header("Filtres")
annee_min = int(df['year'].min())
annee_max = int(df['year'].max())
valeur_fin = min(annee_max, 2021)

annees_choisies = st.sidebar.slider(
    "Période", annee_min, annee_max, (annee_min, valeur_fin)
)

genres_choisis = st.sidebar.multiselect(
    "Genres choisis", genres_tries, 
    default=[]
)

# Application des filtres
df_filtered = df[
    (df['year'] >= annees_choisies[0]) & 
    (df['year'] <= annees_choisies[1])
]

if len(genres_choisis) > 0:
    pattern = '|'.join(genres_choisis)
    df_filtered = df_filtered[df_filtered['genres'].str.contains(pattern, case=False, regex=True)]


st.title("Comparaison des genres cinématographiques dans le temps")

# --- NOUVEAU TEXTE D'INTRODUCTION ---
st.markdown("""
### Bienvenue sur mon outil d'analyse cinématographique 🎬

👈 **Commencez par utiliser la barre latérale à gauche** pour configurer vos données : 
il est **indispensable** de sélectionner une période temporelle et de choisir un ou plusieurs genres pour que les graphiques se mettent à jour et aient du sens. 
N'hésitez pas à vous concentrer sur un genre de film en particulier, ou en sélectionner plusieurs pour les comparer !

Naviguez ensuite à travers les différents onglets ci-dessous :
- **Infos Diverses** : Vue d'ensemble de votre sélection, évolution des durées et distribution des notes.
- **Economie du cinéma** : Analyse financière, rentabilité des films et performances des studios majeurs.
- **Données** : Accès direct au tableau des données brutes (les 1000 premières lignes) avec possibilité de les modifier.
- **Recommandations** : Un moteur intelligent vous suggérant des œuvres similaires basées sur les résumés des films.

Les données proviennent de la base de donnée TMDB, qui est légèrement biaisée : les films américains récents y sont sur-représentés par rapport aux films anciens ou internationaux. Cela explique souvent certains graphiques étonnants !
""")
st.write("---")
# ------------------------------------

onglet1, onglet2, onglet3, onglet4 = st.tabs(["Infos Diverses", "Economie du cinéma", "Données", "Recommandations"])

with onglet1:
    st.header("Statistiques Générales")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Films", len(df_filtered))
    moyenne_note = df_filtered['vote_average'].mean()
    col2.metric("Note Moy.", round(moyenne_note, 2))
    moyenne_duree = df_filtered['runtime'].mean()
    col3.metric("Durée Moy.", f"{int(moyenne_duree)} min")
    
    st.write("---")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Langues parlées")
        compte_langues = df_filtered['original_language'].value_counts().reset_index().head(10)
        compte_langues.columns = ['Langue', 'Nombre']
        fig = px.pie(compte_langues, values='Nombre', names='Langue', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ce graphique en anneau montre la répartition des 10 langues originales les plus fréquentes. La taille de chaque portion est proportionnelle au nombre de films. Vous pourrez voir l'importance de chaque pays pour chaque genre : les Westerns sont par exemple souvent en anglais ou en italien !")
        
    with col_g2:
        st.subheader("Evolution durée moyenne")
        duree_par_an = df_filtered.groupby('year')['runtime'].mean().reset_index()
        fig_ligne = px.line(duree_par_an, x='year', y='runtime')
        fig_ligne.update_yaxes(range=[60, 160])
        st.plotly_chart(fig_ligne, use_container_width=True)
        st.caption("Observez ici comment la durée moyenne des films (en minutes) a évolué au fil des années sélectionnées. Vous remarquerez une augmentation de cette durée au fur et à mesure des innovations technologiques. Certains genre (comme l'animation) sont aussi souvent plus courts que les autres !")

    st.write("---")
    st.subheader("Nombre de films par genre et par an")
    if len(genres_choisis) > 0:
        base = df[(df['year'] >= annees_choisies[0]) & (df['year'] <= annees_choisies[1])]
        liste_data = []
        
        for g in genres_choisis:
            temp = base[base['genres'].str.contains(g, case=False, regex=False)]
            compte = temp.groupby('year').size().reset_index(name='count')
            compte['Genre'] = g
            liste_data.append(compte)
            
        if len(liste_data) > 0:
            final = pd.concat(liste_data)
            fig_genres = px.line(final, x='year', y='count', color='Genre')
            st.plotly_chart(fig_genres, use_container_width=True)
            st.caption("Comparez la popularité des genres sélectionnés dans le temps. Chaque courbe représente le volume de films sortis par an pour un genre donné. Certains genres ont perdu en popularité avec le temps, d'autres en ont gagné, mais le volume de film par an a en moyenne considérablement augmenté (à noter que cela s'explique aussi par le fait que la base de donnée est plus riche pour les années récentes...)")
    else:
        st.info("Sélectionnez des genres à gauche pour voir ce graphique.")
    
    st.write("---")
    st.subheader("Distribution des Notes")
    fig_notes = px.histogram(df_filtered, x="vote_average", nbins=40)
    st.plotly_chart(fig_notes, use_container_width=True)
    st.caption("Ce diagramme illustre la répartition des notes moyennes. Les barres les plus hautes indiquent les notes les plus fréquemment attribuées par le public. Certains genre (comme l'horreur) sont souvent mal notés, d'autres (documentaires par exemple) sont souvent très bien notés")


with onglet2:
    st.header("Analyse Financière")

    st.subheader("Top 10 : Plus gros revenus")
    choix = st.radio("Métrique à observer :", ["Note", "Budget"], horizontal=True)
    colonne_couleur = 'budget' if choix == "Budget" else 'vote_average'
    
    top_10 = df_filtered.nlargest(10, 'revenue').sort_values('revenue', ascending=True)
    
    if len(top_10) > 0:
        fig_top = px.bar(top_10, x='revenue', y='title', orientation='h', color=colonne_couleur)
        st.plotly_chart(fig_top, use_container_width=True)
        st.caption("Ce classement affiche les 10 films ayant généré le plus de revenus au box-office. La couleur des barres vous donne une indication supplémentaire sur le critère choisi au-dessus (leur Note ou leur Budget initial).")

    st.write("---")
    
    st.subheader("Revenus cumulés par budget")
    df_eco = df_filtered[df_filtered['budget'] > 0].copy()
    
    if len(df_eco) > 50:
        df_eco['decile'] = pd.qcut(df_eco['budget'], q=10, labels=False, duplicates='drop') + 1
        par_decile = df_eco.groupby('decile')['revenue'].sum().reset_index()

        fig_dec = px.bar(par_decile, x='decile', y='revenue')
        fig_dec.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig_dec, use_container_width=True)
        st.caption("Les films sont divisés en 10 groupes égaux (déciles) classés selon leur budget de production (1 = les 10% de films les moins chers, 10 = les 10% les plus chers). Les barres représentent les revenus totaux générés par l'ensemble des films de chaque groupe.")
    else:
        st.warning("Pas assez de films sélectionnés --> Changez les filtres")

    st.write("---")

    st.subheader("Rentabilité: Budget vs Revenus")
    df_scatter = df_filtered[
        (df_filtered['budget'] > 1000) & 
        (df_filtered['revenue'] > 1000)].copy()

    if len(df_scatter) > 0:
        fig_roi = px.scatter(
            df_scatter, x='budget', y='revenue', hover_name='title',
            log_x=True, log_y=True, color='vote_average',
            title="Nuage de points")
        
        mini = min(df_scatter['budget'].min(), df_scatter['revenue'].min())
        maxi = max(df_scatter['budget'].max(), df_scatter['revenue'].max())
        
        fig_roi.add_shape(type="line", x0=mini, y0=mini, x1=maxi, y1=maxi,
                          line=dict(color="Red", width=2, dash="dash"))
        
        st.plotly_chart(fig_roi, use_container_width=True)
        st.caption("Chaque point représente un film. S'il se trouve au-dessus de la ligne rouge en pointillés, cela signifie que ses revenus ont dépassé son budget de production (le film a été rentable). Les échelles sont logarithmiques pour mieux gérer les extrêmes. Les couleurs représentent la note moyenne du film.")

    st.write("---")

    st.subheader("Performance des studios")
    
    def clean_nom(texte):
        if pd.isna(texte): 
            return "Inconnu"
        propre = texte.replace("['", "").replace("']", "").replace("', '", ",")
        premier = propre.split(',')[0]
        return premier.strip('"').strip("'")

    df_studios = df_filtered[
        (df_filtered['budget'] > 5000000) & 
        (df_filtered['revenue'] > 0)].copy()
    
    if len(df_studios) > 0:
        df_studios['studio_principal'] = df_studios['production_companies'].apply(clean_nom)
        
        compte = df_studios['studio_principal'].value_counts()
        gros_studios = compte[compte > 30].index
        
        if len(gros_studios) > 0:
            df_top = df_studios[df_studios['studio_principal'].isin(gros_studios)]
            
            stats_studios = df_top.groupby('studio_principal').agg({
                'budget': 'mean', 
                'revenue': 'mean', 
                'title': 'count'
            }).reset_index()

            fig_stud = px.scatter(
                stats_studios, x="budget", y="revenue", size="title", text="studio_principal",
                log_x=True, log_y=True,
                labels={'budget': 'Budget Moyen', 'revenue': 'Revenu Moyen'}
            )
            fig_stud.update_traces(textposition='top center')
            st.plotly_chart(fig_stud, use_container_width=True)
            st.caption("Ce graphique situe les studios majeurs (ayant produit plus de 30 films dans votre sélection) en fonction de leur budget moyen investi et de leur revenu moyen généré. La taille des bulles correspond au nombre de films qu'ils ont produits. On retrouve ainsi en haut à droite les gros studios qui produisent des gros blockbusters rentables, et en bas à gauche les petits studios qui produisent des films plus confidentiels.")
        else:
            st.warning("Aucun studio avec plus de 30 films")
    else:
        st.warning("Pas assez de données")

    st.write("---")

    st.subheader("Budget moyen par note")
    df_note = df_filtered.copy()

    df_note['note_arrondie'] = df_note['vote_average'].round(1)
    df_note = df_note[(df_note['note_arrondie'] >= 2) & (df_note['note_arrondie'] <= 9)]
    
    budget_par_note = df_note.groupby('note_arrondie')['budget'].mean().reset_index()
    fig_barre_note = px.bar(budget_par_note, x='note_arrondie', y='budget')
    st.plotly_chart(fig_barre_note, use_container_width=True)
    st.caption("Ce diagramme croise la note attribuée par le public (arrondie) avec le budget moyen correspondant. Cela permet d'identifier si les films mieux notés sont généralement ceux qui ont coûté le plus cher à produire.")


with onglet3:
    st.header("Editeur de Données")
    st.write("Vous pouvez modifier le tableau ci-dessous :")

    limite_lignes = 1000
    st.info(f"Pour des raisons de fluidité du navigateur, seules les {limite_lignes} premières lignes sont éditables.")
    df_edited = st.data_editor(df_filtered.head(limite_lignes))
    
    st.write("---")
    if st.button("Bouton magique"):
        st.balloons()


with onglet4:
    st.header("Moteur de Recommandations")
    st.write("Sélectionnez un film, et on trouvera les 5 films les plus similaires basés sur l'analyse linguistique de leur résumé.")

    liste_titres = df_filtered['title'].unique()
    choix_film = st.selectbox("Choisissez un film :", liste_titres)

    if st.button("Lancer la recherche"):
        df_reco = df_filtered.reset_index(drop=True)
        
        with st.spinner("Calcul des similarités "):
            tfidf, matrice_tfidf = calculer_tfidf(df_reco)
            
            indices = pd.Series(df_reco.index, index=df_reco['title']).drop_duplicates()
            idx = indices[choix_film]
            
            score_sim = linear_kernel(matrice_tfidf[idx], matrice_tfidf)
            scores = list(enumerate(score_sim[0]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in scores[1:6]]
            
        st.subheader(f"Si vous avez aimé {choix_film}, vous aimerez peut-être :")

        cols = st.columns(5)
        for i, col in enumerate(cols):
            film_idx = top_indices[i]
            titre = df_reco['title'].iloc[film_idx]
            date = str(df_reco['release_date'].iloc[film_idx])[:4]
            note = df_reco['vote_average'].iloc[film_idx]
            
            with col:
                st.info(f"{titre}")
                st.caption(f"Année : {date}")
                st.caption(f"Note : {note}/10")
                with st.expander("Résumé"):
                    st.write(df_reco['overview'].iloc[film_idx])