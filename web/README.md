# ERL Scout, l'application web

Le poste de travail du scout : une composition modifiable (cinq postes, trois places chacun), la liste des joueurs en percentiles, une fiche en tiroir et un suivi en étapes avec notes.

- **Données** : lues en direct sur l'API FastAPI (`/players`, `/players/{nom}/similar`, `/health`).
- **Compositions et suivi** : gardés dans le navigateur (`localStorage`), sans compte.
- **Stack** : React 19, TypeScript, Vite. Aucune bibliothèque d'interface : le glisser-déposer est celui du navigateur.

```bash
npm install
npm run dev        # http://localhost:5173, sur l'API publique
```

Pour travailler sur une API locale, créer `web/.env.development.local` avec `VITE_API_URL=http://localhost:8000`, puis lancer `uvicorn api.main:app` depuis la racine du dépôt.

Publication : `.github/workflows/web.yml` construit et publie sur GitHub Pages à chaque push sur `main` qui touche `web/`.
