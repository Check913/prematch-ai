# Prematch AI — Agent de value betting

Agent autonome qui détecte les **erreurs de cotes** sur le football (avant
match), calcule la mise optimale et vous envoie les paris retenus sur
**Telegram**. Il apprend en continu : ré-entraînement automatique, promotion
uniquement si le nouveau modèle est meilleur, backtest honnête sur cotes de
fermeture.

## Ce qu'il fait, concrètement

1. **Scrape** l'historique réel de 16 championnats (4 saisons, ~17 000 matchs,
   ~800 000 cotes) depuis football-data.co.uk — résultats, xG, cotes
   d'ouverture et de fermeture de ~10 bookmakers.
2. **Entraîne deux modèles** et les mélange :
   * *Dixon-Coles* (Poisson bivarié pondéré dans le temps) — structure du jeu,
     matrice de scores complète ;
   * *Gradient boosting* calibré (isotonique out-of-fold) — forme, Elo, xG.
3. **Calcule la probabilité juste** de chaque issue (1N2, Double Chance,
   Plus/Moins 2.5, BTTS), la compare au marché (dévig *Shin* par défaut) et
   retient les sélections où `p_modèle × cote − 1` dépasse le seuil d'edge.
4. **Mise en Kelly fractionné** (10 % de Kelly, plafond 1 % de bankroll par
   pari, 10 % d'exposition journalière).
5. **Envoie le digest Telegram** (matin et soir) avec les sélections, les
   alternatives et le raisonnement ; alerte immédiate si un edge ≥ 2× le seuil.
6. **Règle les paris**, mesure le ROI réel, **ré-entraîne** la nuit, ne promeut
   un modèle que s'il bat l'ancien en logloss, et **durcit le seuil d'edge**
   automatiquement si le modèle dérive du marché.

## Installation (Windows)

```bat
cd prematch-ai
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env    :: puis remplissez TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID
```

Python ≥ 3.10 requis. Aucune clé d'API payante n'est nécessaire pour la source
principale (football-data.co.uk). Optionnel : une clé gratuite
[The Odds API](https://the-odds-api.com) (500 requêtes/mois) ajoute des cotes
multi-bookmakers en temps réel.

## Bot Telegram (2 minutes)

1. Dans Telegram, parlez à **@BotFather** → `/newbot` → récupérez le token.
2. Envoyez `/start` à votre bot, puis ouvrez
   `https://api.telegram.org/bot<VOTRE_TOKEN>/getUpdates` → copiez
   `result[0].message.chat.id`.
3. Collez les deux valeurs dans `.env`.

## Utilisation

```bat
:: 1. donner a manger (la premiere fois, ~2 min grace au cache)
python -m prematch ingest-history     :: 17 000 matchs + 800 000 cotes
python -m prematch ingest-fixtures    :: matchs a venir + cotes du moment

:: 2. entrainer
python -m prematch train

:: 3. valider (backtest walk-forward honnete, ~10 min)
python -m prematch backtest --save
python -m prematch backtest --sweep   :: teste plusieurs seuils d'edge

:: 4. les paris du jour
python -m prematch today
python -m prematch show arsenal       :: analyse detaillee d'un match

:: 5. l'agent complet 24/7 (bot Telegram + scheduler + auto-apprentissage)
python -m prematch bot

:: sante du systeme
python -m prematch status
```

Démarrage automatique au boot : `scripts\install_task.ps1` (Planificateur
Windows) ou `scripts/start.bat` pour une session manuelle.

## Mode démo (sans réseau)

```bat
python -m prematch demo
```
Génère un championnat synthétique dont on connaît les vraies forces + des
cotes bruitées, et fait tourner tout le pipeline. Sert aussi de test de
non-régression (`python -m pytest tests/ -q`).

## Méthodologie — pourquoi c'est sérieux

* **Aucune fuite temporelle** : les features d'un match ne voient que le
  passé ; le backtest ré-entraîne par blocs avec une fenêtre glissante.
* **Cotes de fermeture** pour le backtest : c'est la meilleure estimation du
  marché au coup d'envoi — battre la fermeture est LE standard de la value.
* **Diagnostic intégré** : `scripts/eval_vs_market.py` compare logloss du
  modèle vs marché déviigé et calibration sur les derniers matchs.
* **Dévig Shin** (et options multiplicative/power) plutôt qu'une simple
  normalisation : retire la marge du bookmaker en modélisant le biais des
  parieurs.
* **Calibrage par température** : mesuré sur un holdout interne à chaque
  entraînement, il corrige la sous-confiance typique des modèles
  régularisés (T≈0.65–0.90 en pratique) — sans lui, les longshots génèrent
  des edges fantômes.
* **Plafond d'edge** (`max_edge: 0.60`) : un « edge » énorme est un signal
  d'alerte modèle, pas de l'argent libre — ces sélections sont écartées.
* **Kelly fractionné** : la mise croît avec l'edge mais reste plafonnée ;
  jamais plus de 1 % par pari (10 % de Kelly).
* **Sélection conservatrice** : edge ≥ 3 %, cotes 1.40–8.00, accord des deux
  modèles ≥ 55 %, max 1 pari par match et 8 par jour.
* **Auto-amélioration mesurée** : promotion sur logloss holdout, dérive
  modèle/marché surveillée, seuil d'edge durci si dégradation.

## Résultats de validation (novembre 2026)

Données réelles : 17 446 matchs, 814 103 cotes, 16 championnats, saisons
2023-24 → 2026-27. Backtest walk-forward corrigé (labels de test réparés,
plafond d'edge, calibrage par température, holdout interne sans fuite) :

```
4 347 paris sur 223 jours | ROI -3.53% | reussite 35.4%
P&L -950.75 EUR | drawdown max -95.4% | Sharpe -1.62
```

**Lecture honnête** : le modèle calibré atteint un logloss de 0.9335 contre
0.9296 pour le marché déviigé (cotes de fermeture) sur 200 matchs hors
échantillon — il est *quasi au niveau du marché* mais ne le bat pas. Parier
avec 3 % d'edge supposé contre la fermeture perd donc ~la marge du
bookmaker : c'est le résultat attendu d'un modèle construit sur données
publiques. Le framework (scraping, apprentissage, garde-fous, Telegram) est
opérationnel ; la rentabilité, elle, n'est PAS démontrée.

Enseignements appliqués : Kelly réduit à 10 % et mise max 1 % (un drawdown
de -95 % avec Kelly 25 % montre qu'un système sans edge est ruinant en mise
pleine) ; plafond d'edge à 60 % pour filtrer les artefacts.

Prochaines pistes pour tenter de dépasser le marché : cotes d'ouverture au
lieu de fermeture (le marché est moins efficace tôt), divisions moins
couvertes, The Odds API pour trouver les meilleurs prix multi-bookmakers,
et exploitation des baisses de cotes (steam) plutôt que des niveaux.

## Limites honnêtes

* Un ROI positif en backtest n'est **pas** une promesse de gain : variance
  élevée, cotes futures différentes des cotes de fermeture historiques,
  limits des bookmakers sur gagnants répétés.
* Les xG manquants (divisions mineures, débuts de saison) sont imputés à la
  moyenne — les features restent utilisables mais moins informatives.
* Le backtest suppose que la cote de fermeture était prenable : en pratique,
  placez vos paris plus tôt avec des cotes égales ou meilleures.

## Structure

```
prematch/
  sources/     football_data.py (CSV gratuits), odds_api.py, demo.py
  http.py      client poli : cache disque, délais, robots.txt, retries
  features.py  forme, Elo, xG glissants, repos — sans fuite temporelle
  model.py     Dixon-Coles + GBM calibré + ensemble, persistance joblib
  markets.py   Poisson bivarié, dévig (Shin/power), Kelly
  value.py     edge, sélecteur de picks, analyse par match
  backtest.py  walk-forward + sweep de seuils
  learning.py  ré-entraînement, promotion, dérive, risque adaptatif
  pipeline.py  orchestration unique
  telegram_bot.py  digest, alertes, commandes (/today /stats /bankroll)
  scheduler.py boucles 24/7 (rafraîchissement + cycle d'apprentissage)
  cli.py       interface en ligne de commande
```

## Avertissement

Outil d'analyse statistique fourni tel quel. Les paris sportifs comportent un
risque de perte. Aucune garantie de profit. Misez uniquement ce que vous
pouvez vous permettre de perdre. 18+.
