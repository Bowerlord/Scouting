import { createContext, useContext } from "react";
import type { Player, Role } from "./data";
import type { Comp, Stage, Tracked } from "./store";

export type View = "accueil" | "joueurs" | "composition" | "suivi";

export interface AppState {
  players: Player[];
  byId: Map<string, Player>;
  season: number;
  comps: Comp[];
  comp: Comp;
  track: Record<string, Tracked>;
  selected: string | null;
  open: (id: string | null) => void;
  go: (v: View) => void;
  toast: (msg: string) => void;
  /** Place un joueur sur une place précise de la composition ouverte. */
  place: (id: string, role: Role, index: number) => void;
  removeFromComp: (id: string) => void;
  addToComp: (id: string) => void;
  inComp: (id: string) => boolean;
  follow: (id: string, stage?: Stage) => void;
  unfollow: (id: string) => void;
  setNote: (id: string, note: string) => void;
  markSeen: (id: string) => void;
}

export const AppCtx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const v = useContext(AppCtx);
  if (!v) throw new Error("useApp hors du fournisseur");
  return v;
}

/** Le glisser-déposer transporte l'identifiant du joueur et rien d'autre. */
export const DRAG_TYPE = "text/plain";
