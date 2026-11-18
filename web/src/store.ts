// Ce que l'utilisateur construit : ses compositions et son suivi.
//
// Gardé dans le navigateur, sans compte. C'est le choix de la première
// version : rien à héberger, rien à sécuriser. Le prix à payer est qu'un autre
// navigateur ne voit pas les mêmes compositions.

import { useEffect, useState } from "react";
import type { Player, Role } from "./data";
import { ROLES } from "./data";

export const STAGES = ["Repéré", "À regarder", "Contacté", "Essai", "Signé"] as const;
export type Stage = (typeof STAGES)[number];

export type Lineup = Record<Role, (string | null)[]>;
export interface Comp {
  id: string;
  name: string;
  slots: Lineup;
}
export interface Tracked {
  stage: Stage;
  note: string;
  /** Date ISO de la dernière fois qu'on a regardé ce joueur. */
  seen: string;
}

export const emptyLineup = (): Lineup => ({ top: [null, null, null], jng: [null, null, null], mid: [null, null, null], bot: [null, null, null], sup: [null, null, null] });

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function useStored<T>(key: string, fallback: () => T) {
  const [value, setValue] = useState<T>(() => read(key, fallback()));
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* stockage indisponible : l'app marche, sans mémoire */
    }
  }, [key, value]);
  return [value, setValue] as const;
}

/** Première visite : une composition déjà remplie avec les trois meilleurs de chaque poste. */
export function starterComp(players: Player[]): Comp {
  const slots = emptyLineup();
  for (const { id } of ROLES) {
    const best = players.filter((p) => p.role === id && p.rank != null).slice(0, 3);
    slots[id] = [best[0]?.id ?? null, best[1]?.id ?? null, best[2]?.id ?? null];
  }
  return { id: "c1", name: "Compo LEC 2027", slots };
}

export const daysSince = (iso: string) => Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000));

/** Au-delà, un joueur suivi est signalé : on l'a oublié. */
export const STALE_DAYS = 21;
