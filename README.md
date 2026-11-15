# Trading AI Project

## Installation

Make sure python3.13 is installed on your machine.

### Mac OS
if python3.13 is not installed, run the following command:
```bash
brew install python@3.13
```

then
``` bash
alias python="python3.13"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### Windows
```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## To-Do

### Latest discussion

- chaque heure, fetch une grande liste d'article

- pour fetcher les urls, on utiliser DDGs, mais aussi les classes news_scrapper
- checker si l'article a deja ete extrait, avant de refaire l'extraction, pour ne pas re-extracter plusieurs fois

    - pour chaque article fetcher, on vas determiner si l'article est important ou pas par rapport au theme 
    
    - pour savoir pour si l'article est important, on demande a un llm:
        -  "le theme du system est Trading News, determine si cet article merite d'etre sauvegardé ou pas. Si il doit etre sauvegardé, return True, sinon return False "
        - si oui, sauvegarder l'article dans la bd, si non, ignorer l'article
        
    - pour chaque article importants, on va l'enregistrer dans une base de données
    - chaque object d'article dans la bd doit avoir:
        - titre
        - url
        - resume
        - resume_auto
        - auteurs
        - date
        - motscles
    
    - a chaque que le pipeline est lancé, on va aussi stocker dans la bd des logs par rapport au pipeline
    
Base de données:
- MongoDB
 - collection: articles
    - document:
        - titre
        - url
        - resume
        - resume_auto
        - auteurs
        
- collection: pipeline_log
    - document:
        - heure à laquelle on fetch les articles
        - le nombre d'url fetcher
        - le nombre article extrait
        - le nombre d'article importants et enregistrer

- Pour plus tard, il faudra prevoir comment tout deployer sur le cloud (soit GCP "google cloud run")

Cron Job:
- run the pipeline every {period}


### Chatgpt 
Voici un résumé en français clair et concis :

Le système exécute un **pipeline automatique chaque heure** pour récupérer une grande liste d’articles.
Les **URLs** sont obtenues à la fois via **DDGS** et les **classes `news_scraper`**. Avant d’extraire un article, le système vérifie s’il a déjà été traité afin d’éviter les duplications.

Chaque article est ensuite évalué par un **LLM** pour déterminer sa pertinence par rapport au **thème “Trading News”**.

* Si l’article est jugé pertinent, il est **enregistré dans la base de données**.
* Sinon, il est ignoré.

Les **articles importants** sauvegardés contiennent :

* titre
* URL
* résumé
* résumé automatique
* auteurs
* date
* mots-clés

Chaque exécution du pipeline génère aussi des **logs** enregistrés dans la base de données.

### Structure de la base de données (MongoDB)

* **Collection `articles`** : stocke les informations sur les articles.
* **Collection `pipeline_log`** : conserve les données d’exécution (heure, nombre d’URLs récupérées, d’articles extraits et d’articles retenus).

Enfin, le pipeline sera **déployé sur le cloud (ex. : GCP – Cloud Run)** et exécuté automatiquement via un **cron job** à intervalle régulier.

Catalyst clustering also admits tickerless corporate, regulatory, geopolitical and macroeconomic events with a summary and an importance score of at least 0.4. Preview the past 30 days with `python -m scripts.backfill_tickerless_events`. Add `--apply` to process eligible articles that still lack a cluster.
