import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Chemins relatifs : l'app est servie sous un sous-dossier (GitHub Pages,
// /Scouting/) et la navigation passe par le fragment d'URL, donc aucun
// serveur n'a besoin de connaître les routes.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
