// Données et calculs partagés par tous les écrans.
//
// L'API renvoie une ligne par joueur, saison et split. L'application raisonne
// en joueurs : on garde la meilleure ligne de chaque joueur à son poste, puis
// on situe chacun en percentile face aux autres joueurs du même poste.

export const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "")
  ?? "https://scouting-api-7750158787.europe-west1.run.app";

export const AGENT_URL = "https://scouting-lol-erl.streamlit.app/Agent";

export type Role = "top" | "jng" | "mid" | "bot" | "sup";

export const ROLES: { id: Role; label: string; plural: string }[] = [
  { id: "top", label: "Top", plural: "tops" },
  { id: "jng", label: "Jungle", plural: "junglers" },
  { id: "mid", label: "Mid", plural: "mids" },
  { id: "bot", label: "Bot", plural: "bots" },
  { id: "sup", label: "Support", plural: "supports" },
];

export const roleLabel = (r: Role) => ROLES.find((x) => x.id === r)!.label;
export const rolePlural = (r: Role) => ROLES.find((x) => x.id === r)!.plural;

/** En dessous, un classement n'a pas de valeur : trois parties ne disent rien. */
export const MIN_GAMES = 10;

interface ApiPlayer {
  playername: string;
  playername_original?: string | null;
  league: string;
  season?: number | null;
  split?: string | null;
  position: string;
  teamname?: string | null;
  talent_score?: number | null;
  games_played?: number | null;
  win_rate?: number | null;
  archetype?: string | null;
  champion_pool_size?: number | null;
  promoted_to_lec?: boolean | null;
  dpm_zscore?: number | null;
  cspm_zscore?: number | null;
  golddiffat15_zscore?: number | null;
  distance?: number;
}

export interface Player {
  id: string;
  key: string;
  name: string;
  role: Role;
  team: string;
  league: string;
  score: number;
  games: number;
  winRate: number;
  pool: number | null;
  archetype: string;
  promoted: boolean;
  /** Percentiles dans le poste, 0 à 100 : avance à 15 min, dégâts, farm. */
  pLane: number | null;
  pDmg: number | null;
  pFarm: number | null;
  rank: number | null;
  rankOf: number;
}

export const playerId = (key: string, role: string) => `${key}|${role}`;

// Servie par la coquille Streamlit, l'app reçoit les joueurs directement dans
// la page : pas d'aller-retour réseau, et pas de dépendance à la version
// déployée de l'API. Hors de Streamlit, elle les demande à l'API.
declare global {
  interface Window {
    __ERL__?: { rows: ApiPlayer[]; refreshed?: string | null };
  }
}
const embedded = () => (typeof window !== "undefined" ? window.__ERL__ : undefined);

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`L'API a répondu ${res.status} sur ${path}`);
  return res.json() as Promise<T>;
}

export async function loadHealth(): Promise<{ data_refreshed_at?: string | null }> {
  const e = embedded();
  if (e) return { data_refreshed_at: e.refreshed ?? null };
  return getJson("/health");
}

export async function loadSeason(season: number): Promise<Player[]> {
  const e = embedded();
  if (e) return buildPlayers(e.rows.filter((r) => r.season === season && (r.games_played ?? 0) >= 3));
  const rows: ApiPlayer[] = [];
  let offset = 0;
  for (;;) {
    const page = await getJson<{ total: number; items: ApiPlayer[] }>(
      `/players?season=${season}&min_games=3&limit=500&offset=${offset}`,
    );
    rows.push(...page.items);
    offset += page.items.length;
    if (offset >= page.total || page.items.length === 0) break;
  }
  return buildPlayers(rows);
}

export async function loadSimilar(key: string): Promise<{ key: string; name: string; role: Role; team: string; resemblance: number }[]> {
  const rows = await getJson<ApiPlayer[]>(`/players/${encodeURIComponent(key)}/similar?count=5`);
  return rows.map((r) => ({
    key: r.playername,
    name: r.playername_original || r.playername,
    role: r.position as Role,
    team: r.teamname || "",
    // La distance est euclidienne sur trois z-scores : au-delà de 3, les
    // profils n'ont plus rien en commun.
    resemblance: Math.round(100 * Math.max(0, 1 - (r.distance ?? 3) / 3)),
  }));
}

function pctRank(values: (number | null)[]): (number | null)[] {
  const known = values.filter((v): v is number => v != null).sort((a, b) => a - b);
  return values.map((v) => {
    if (v == null || known.length === 0) return null;
    let below = 0;
    while (below < known.length && known[below] <= v) below++;
    return Math.round((100 * below) / known.length);
  });
}

function buildPlayers(rows: ApiPlayer[]): Player[] {
  const best = new Map<string, ApiPlayer>();
  for (const r of rows) {
    const id = playerId(r.playername, r.position);
    const cur = best.get(id);
    if (!cur || (r.talent_score ?? -1) > (cur.talent_score ?? -1)) best.set(id, r);
  }
  const out: Player[] = [];
  for (const role of ROLES.map((x) => x.id)) {
    const group = [...best.values()].filter((r) => r.position === role);
    const eligible = group.filter((r) => (r.games_played ?? 0) >= MIN_GAMES);
    const lane = pctRank(eligible.map((r) => r.golddiffat15_zscore ?? null));
    const dmg = pctRank(eligible.map((r) => r.dpm_zscore ?? null));
    const farm = pctRank(eligible.map((r) => r.cspm_zscore ?? null));
    const sorted = [...eligible].sort((a, b) => (b.talent_score ?? 0) - (a.talent_score ?? 0));
    const rankOf = new Map(sorted.map((r, i) => [r.playername, i + 1]));
    const pct = new Map(eligible.map((r, i) => [r.playername, [lane[i], dmg[i], farm[i]] as const]));
    for (const r of group) {
      const p = pct.get(r.playername);
      out.push({
        id: playerId(r.playername, role),
        key: r.playername,
        name: r.playername_original || r.playername,
        role,
        team: r.teamname || "Équipe inconnue",
        league: r.league,
        score: Math.round(r.talent_score ?? 0),
        games: r.games_played ?? 0,
        winRate: Math.round(100 * (r.win_rate ?? 0)),
        pool: r.champion_pool_size ?? null,
        archetype: archetype(r.archetype),
        promoted: Boolean(r.promoted_to_lec),
        pLane: p ? p[0] : null,
        pDmg: p ? p[1] : null,
        pFarm: p ? p[2] : null,
        rank: rankOf.get(r.playername) ?? null,
        rankOf: eligible.length,
      });
    }
  }
  return out.sort((a, b) => (a.rank ?? 9999) - (b.rank ?? 9999));
}

/** Le clustering produit des étiquettes en anglais empilées : on en tire une seule, en français. */
function archetype(raw?: string | null): string {
  const a = raw ?? "";
  if (a.includes("Carry agressif")) return "Carry agressif";
  if (a.includes("Lane bully") && a.includes("High DPS")) return "Domine tôt, gros dégâts";
  if (a.includes("Lane bully")) return "Domine sa voie";
  if (a.includes("High DPS")) return "Gros dégâts";
  return "Profil équilibré";
}

export type Tier = "hi" | "mid" | "lo";
export const tier = (v: number | null): Tier => (v == null ? "mid" : v >= 60 ? "hi" : v >= 40 ? "mid" : "lo");

export function level(v: number | null): string {
  if (v == null) return "pas de donnée";
  if (v >= 80) return "parmi les meilleurs";
  if (v >= 60) return "au-dessus";
  if (v >= 40) return "dans la moyenne";
  if (v >= 20) return "en dessous";
  return "parmi les plus faibles";
}

export const stats = (p: Player): [string, number | null][] => [
  ["Avance à 15 min", p.pLane],
  ["Dégâts par minute", p.pDmg],
  ["Farm par minute", p.pFarm],
];

export function rankShort(p: Player): string {
  if (p.rank == null) return "–";
  return p.rank === 1 ? "1er" : `${p.rank}e`;
}

export function rankText(p: Player): string {
  if (p.rank == null) return `Moins de ${MIN_GAMES} matchs, pas classé`;
  return `${rankShort(p)} sur ${p.rankOf} ${rolePlural(p.role)}`;
}

export function sentence(p: Player): string {
  if (p.pLane == null) return "Pas assez de matchs pour le décrire.";
  const lane = p.pLane >= 80 ? "Écrase sa voie" : p.pLane >= 60 ? "Gagne sa voie" : p.pLane >= 40 ? "Tient sa voie" : "Perd souvent sa voie";
  const d = p.pDmg ?? 50;
  const dmg = d >= 80 ? "énormément de dégâts" : d >= 60 ? "beaucoup de dégâts" : d >= 40 ? "des dégâts normaux" : "peu de dégâts";
  if (p.role === "sup") return `${lane}, fait ${dmg}.`;
  const f = p.pFarm ?? 50;
  const farm = f >= 60 ? "farme bien" : f >= 40 ? "farm correct" : "farm faible";
  return `${lane}, fait ${dmg}, ${farm}.`;
}
