import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// Deux sorties.
//  - `npm run build` : un site classique dans web/dist, pour un hébergement
//    statique éventuel.
//  - `npm run build:streamlit` : un seul fichier HTML, tout inclus, écrit dans
//    app/web/. L'app publique vit à l'adresse Streamlit historique, et
//    Streamlit Cloud ne lance pas npm : il sert ce fichier tel qu'il est commité.
export default defineConfig(({ mode }) => ({
  base: "./",
  plugins: mode === "streamlit" ? [react(), viteSingleFile()] : [react()],
  build: mode === "streamlit" ? { outDir: "../app/web", emptyOutDir: true, copyPublicDir: false } : {},
}));
