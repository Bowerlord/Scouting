import { useMemo, useState } from "react";
import { MIN_GAMES, ROLES, level, rankShort, rankText, roleLabel, stats, tier } from "./data";
import type { Player, Role } from "./data";
import { useApp } from "./app-context";
import { Pips, RoleIcon, useDraggable, useDrop } from "./ui";
import { STAGES, STALE_DAYS, daysSince } from "./store";
import type { Stage } from "./store";

/* ── Pièces communes ─────────────────────────────────────────────────────── */

function Chip({ p, showRole }: { p: Player; showRole?: boolean }) {
  return (
    <div className="chip" {...useDraggable(p.id)}>
      {showRole && <RoleIcon role={p.role} />}
      <span className="nm">{p.name}</span>
      <span className="rk">{rankShort(p)}</span>
      <Pips p={p} />
    </div>
  );
}

function BigCard({ p }: { p: Player }) {
  return (
    <div className="bigcard" {...useDraggable(p.id)}>
      <div className="nm">{p.name}</div>
      <div className="tm">{p.team}, {p.league}</div>
      <div className="rk">{rankText(p)}</div>
      <div className="mini">
        {stats(p).map(([l, v]) => (
          <div key={l} title={`${l} : ${level(v)}`}>
            <span>{l.split(" ")[0]}</span>
            <i><b className={tier(v)} style={{ width: `${v ?? 0}%` }} /></i>
          </div>
        ))}
      </div>
    </div>
  );
}

function RoleFilter({ value, onChange }: { value: Role | "all"; onChange: (r: Role | "all") => void }) {
  return (
    <div className="seg" role="group" aria-label="Filtrer par poste">
      {([["all", "Tous"], ...ROLES.map((r) => [r.id, r.label])] as [Role | "all", string][]).map(([id, l]) => (
        <button key={id} aria-pressed={value === id} onClick={() => onChange(id)}>{l}</button>
      ))}
    </div>
  );
}

function compSummary(app: ReturnType<typeof useApp>) {
  const titulars = ROLES.map((r) => app.comp.slots[r.id][0]).map((id) => (id ? app.byId.get(id) : undefined));
  const ranks = titulars.filter((p): p is Player => !!p && p.rank != null).map((p) => p.rank!);
  const avg = ranks.length ? ranks.reduce((a, b) => a + b, 0) / ranks.length : null;
  const holes = ROLES.map((r) => ({ r, n: app.comp.slots[r.id].filter(Boolean).length })).filter((x) => x.n < 3);
  return { avg, holes };
}

/* ── Composition ─────────────────────────────────────────────────────────── */

function Slot({ role, index }: { role: Role; index: number }) {
  const app = useApp();
  const id = app.comp.slots[role][index];
  const p = id ? app.byId.get(id) : undefined;
  const drop = useDrop((dragged) => app.place(dragged, role, index));
  return (
    <div className={`slot ${index === 0 ? "first" : ""}`} {...drop}>
      {p ? (index === 0 ? <BigCard p={p} /> : <Chip p={p} />) : id ? (
        <div className="slot-empty">Pas de données en {app.season}</div>
      ) : (
        <div className="slot-empty">{index === 0 ? `Titulaire ${roleLabel(role)} à trouver` : "Doublure à trouver"}</div>
      )}
    </div>
  );
}

export function CompositionView() {
  const app = useApp();
  const [filter, setFilter] = useState<Role | "all">("all");
  const [q, setQ] = useState("");
  const { avg, holes } = compSummary(app);
  const bench = app.players.filter(
    (p) => p.rank != null && !app.inComp(p.id) && (filter === "all" || p.role === filter) && (!q || p.name.toLowerCase().includes(q.toLowerCase()) || p.team.toLowerCase().includes(q.toLowerCase())),
  );
  const drop = useDrop((id) => app.removeFromComp(id));

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{app.comp.name}</h1>
          <p className="lede">Trois places par poste. Glisse un joueur d'une place à l'autre ou depuis les candidats, clique sur un nom pour sa fiche.</p>
        </div>
        <div className="sum">
          {avg != null && (
            <div className="sum-main"><b>{avg.toFixed(1).replace(".", ",")}e</b><span>rang moyen de tes titulaires dans leur poste</span></div>
          )}
          {holes.map(({ r, n }) => (
            <div key={r.id} className="sum-warn">{r.label} : {n} joueur{n > 1 ? "s" : ""}, il manque {3 - n} place{3 - n > 1 ? "s" : ""}</div>
          ))}
        </div>
      </header>

      <div className="depth">
        {ROLES.map((r) => (
          <section key={r.id} className="depth-col" aria-label={r.label}>
            <h2><RoleIcon role={r.id} /><span>{r.label}</span><em>{app.comp.slots[r.id].filter(Boolean).length}/3</em></h2>
            {[0, 1, 2].map((i) => <Slot key={i} role={r.id} index={i} />)}
          </section>
        ))}
      </div>

      <section className="bench" {...drop}>
        <div className="bench-h">
          <h2>Candidats</h2>
          <RoleFilter value={filter} onChange={setFilter} />
          <input className="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Nom ou équipe" aria-label="Filtrer les candidats" />
          <span className="muted">Dépose un joueur ici pour le retirer de la compo</span>
        </div>
        <div className="chips">
          {bench.slice(0, 60).map((p) => <Chip key={p.id} p={p} showRole />)}
        </div>
        {bench.length > 60 && <p className="muted more">{bench.length - 60} autres : affine avec le poste ou la recherche.</p>}
        {bench.length === 0 && <p className="muted more">Aucun candidat avec ces filtres.</p>}
      </section>
    </div>
  );
}

/* ── Joueurs ─────────────────────────────────────────────────────────────── */

type SortKey = "rank" | "name" | "pLane" | "pDmg" | "pFarm" | "winRate" | "games";

export function PlayersView() {
  const app = useApp();
  const [role, setRole] = useState<Role | "all">("all");
  const [league, setLeague] = useState("all");
  const [q, setQ] = useState("");
  const [minGames, setMinGames] = useState(MIN_GAMES);
  const [sort, setSort] = useState<{ k: SortKey; asc: boolean }>({ k: "rank", asc: true });
  const leagues = useMemo(() => [...new Set(app.players.map((p) => p.league))].sort(), [app.players]);

  const rows = useMemo(() => {
    const s = q.toLowerCase();
    const list = app.players.filter(
      (p) => (role === "all" || p.role === role) && (league === "all" || p.league === league) && p.games >= minGames && (!s || `${p.name} ${p.team}`.toLowerCase().includes(s)),
    );
    const val = (p: Player): number | string => (sort.k === "name" ? p.name.toLowerCase() : sort.k === "rank" ? p.rank ?? 9999 : (p[sort.k] as number | null) ?? -1);
    return list.sort((a, b) => {
      const x = val(a), y = val(b);
      const c = x < y ? -1 : x > y ? 1 : 0;
      return sort.asc ? c : -c;
    });
  }, [app.players, role, league, q, minGames, sort]);

  const th = (k: SortKey, label: string, defaultAsc = false) => (
    <th aria-sort={sort.k === k ? (sort.asc ? "ascending" : "descending") : "none"}>
      <button onClick={() => setSort((s) => ({ k, asc: s.k === k ? !s.asc : defaultAsc }))}>
        {label}{sort.k === k ? (sort.asc ? " ↑" : " ↓") : ""}
      </button>
    </th>
  );

  const pct = (v: number | null) => (
    <td><span className={`pct ${tier(v)}`} title={level(v)}>{v ?? "–"}</span></td>
  );

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Joueurs</h1>
          <p className="lede">Chaque stat est un percentile dans le poste : 80 veut dire mieux que 80 % des joueurs du même poste en {app.season}.</p>
        </div>
      </header>
      <div className="filters">
        <RoleFilter value={role} onChange={setRole} />
        <select value={league} onChange={(e) => setLeague(e.target.value)} aria-label="Ligue">
          <option value="all">Toutes les ligues</option>
          {leagues.map((l) => <option key={l}>{l}</option>)}
        </select>
        <label className="inline">Matchs minimum
          <input type="number" min={0} max={80} value={minGames} onChange={(e) => setMinGames(Number(e.target.value) || 0)} />
        </label>
        <input className="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Nom ou équipe" aria-label="Chercher" />
        <span className="muted">{rows.length} joueurs</span>
      </div>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              {th("rank", "Rang au poste", true)}
              {th("name", "Joueur", true)}
              <th>Équipe</th>
              {th("pLane", "Avance 15 min")}
              {th("pDmg", "Dégâts")}
              {th("pFarm", "Farm")}
              {th("winRate", "Victoires")}
              {th("games", "Matchs")}
              <th><span className="sr">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 200).map((p) => (
              <tr key={p.id} className={app.selected === p.id ? "sel" : ""} onClick={() => app.open(p.id)}>
                <td className="num">{p.rank ? rankShort(p) : "–"}</td>
                <td><span className="who"><RoleIcon role={p.role} /><b>{p.name}</b>{app.track[p.id] && <span className="tag small">{app.track[p.id].stage}</span>}</span></td>
                <td className="muted">{p.team}, {p.league}</td>
                {pct(p.pLane)}{pct(p.pDmg)}{pct(p.pFarm)}
                <td className="num">{p.winRate} %</td>
                <td className="num">{p.games}</td>
                <td className="row-act" onClick={(e) => e.stopPropagation()}>
                  <button className="link" onClick={() => (app.inComp(p.id) ? app.removeFromComp(p.id) : app.addToComp(p.id))}>{app.inComp(p.id) ? "Dans la compo" : "+ Compo"}</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <p className="muted more">Aucun joueur avec ces filtres. Baisse le nombre de matchs minimum ou change de ligue.</p>}
      </div>
    </div>
  );
}

/* ── Suivi ───────────────────────────────────────────────────────────────── */

const EMPTY: Record<Stage, string> = {
  "Repéré": "Glisse ici un joueur à surveiller",
  "À regarder": "Les joueurs à voir en vidéo",
  "Contacté": "Personne n'a encore été contacté",
  "Essai": "Aucun essai prévu",
  "Signé": "Glisse ici un joueur qui signe",
};

function TrackCard({ p }: { p: Player }) {
  const app = useApp();
  const t = app.track[p.id];
  const d = daysSince(t.seen);
  return (
    <div className="tcard" {...useDraggable(p.id)}>
      <div className="t1"><RoleIcon role={p.role} /><b>{p.name}</b><span>{roleLabel(p.role)}</span></div>
      <div className="tm">{p.team}, {p.league}</div>
      <div className="rk"><span>{rankText(p)}</span><Pips p={p} /></div>
      {t.note && <p>{t.note}</p>}
      <div className={`ft ${d > STALE_DAYS ? "stale" : ""}`}>{d === 0 ? "vu aujourd'hui" : d > STALE_DAYS ? `pas revu depuis ${d} jours` : `vu il y a ${d} j`}</div>
    </div>
  );
}

function StageCol({ stage }: { stage: Stage }) {
  const app = useApp();
  const ids = Object.keys(app.track).filter((id) => app.track[id].stage === stage && app.byId.has(id));
  const drop = useDrop((id) => app.follow(id, stage));
  return (
    <section className="kcol" {...drop} aria-label={stage}>
      <h2>{stage}<em>{ids.length}</em></h2>
      {ids.map((id) => <TrackCard key={id} p={app.byId.get(id)!} />)}
      {ids.length === 0 && <div className="kempty">{EMPTY[stage]}</div>}
    </section>
  );
}

export function PipelineView() {
  const app = useApp();
  const stale = Object.entries(app.track).filter(([id, t]) => app.byId.has(id) && daysSince(t.seen) > STALE_DAYS).length;
  const ideas = app.players
    .filter((p) => p.rank != null && !app.track[p.id] && !app.inComp(p.id) && (p.pLane ?? 0) + (p.pDmg ?? 0) >= 150)
    .slice(0, 6);
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{stale ? `${stale} joueur${stale > 1 ? "s" : ""} à revoir` : "Suivi"}</h1>
          <p className="lede">Glisse une carte d'une colonne à l'autre quand un joueur avance. Une carte devient orange quand personne ne l'a regardée depuis {STALE_DAYS} jours.</p>
        </div>
      </header>
      {ideas.length > 0 && (
        <div className="ideas">
          <span>Pas encore suivis, très forts en début de partie :</span>
          {ideas.map((p) => <Chip key={p.id} p={p} showRole />)}
        </div>
      )}
      <div className="kanban">{STAGES.map((s) => <StageCol key={s} stage={s} />)}</div>
    </div>
  );
}

/* ── Accueil ─────────────────────────────────────────────────────────────── */

export function HomeView({ refreshed }: { refreshed: string | null }) {
  const app = useApp();
  const { avg, holes } = compSummary(app);
  const stale = Object.entries(app.track).filter(([id, t]) => app.byId.has(id) && daysSince(t.seen) > STALE_DAYS);
  const counts = STAGES.map((s) => [s, Object.values(app.track).filter((t) => t.stage === s).length] as const);
  const headline = holes.length
    ? `${app.comp.name} : ${holes.map((h) => h.r.label.toLowerCase()).join(", ")} à compléter`
    : stale.length
      ? `${stale.length} joueur${stale.length > 1 ? "s" : ""} suivi${stale.length > 1 ? "s" : ""} à revoir`
      : `${app.comp.name} est complète`;

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{headline}</h1>
          <p className="lede">Saison {app.season}, {app.players.filter((p) => p.rank != null).length} joueurs classés{refreshed ? `, données jusqu'au ${new Date(refreshed).toLocaleDateString("fr-FR")}` : ""}.</p>
        </div>
      </header>

      <div className="home">
        <section className="panel">
          <div className="panel-h"><h2>Tes titulaires</h2><button className="link" onClick={() => app.go("composition")}>Ouvrir la composition</button></div>
          <div className="five">
            {ROLES.map((r) => {
              const id = app.comp.slots[r.id][0];
              const p = id ? app.byId.get(id) : undefined;
              return (
                <div key={r.id} className="five-slot">
                  <small><RoleIcon role={r.id} /> {r.label}</small>
                  {p ? <button className="link strong" onClick={() => app.open(p.id)}>{p.name}</button> : <b className="warn">à trouver</b>}
                  {p && <span className="muted">{rankText(p)}</span>}
                </div>
              );
            })}
          </div>
          {avg != null && <p className="muted">Rang moyen dans leur poste : {avg.toFixed(1).replace(".", ",")}e.</p>}
        </section>

        <section className="panel">
          <div className="panel-h"><h2>Ton suivi</h2><button className="link" onClick={() => app.go("suivi")}>Ouvrir le suivi</button></div>
          <div className="counts">{counts.map(([s, n]) => <div key={s}><b>{n}</b><span>{s}</span></div>)}</div>
          {stale.length > 0 ? (
            <div className="stale-list">
              {stale.map(([id, t]) => (
                <button key={id} className="stale-row" onClick={() => app.open(id)}>
                  <b>{app.byId.get(id)!.name}</b><span>pas revu depuis {daysSince(t.seen)} jours</span>
                </button>
              ))}
            </div>
          ) : (
            <p className="muted">Personne n'attend depuis plus de {STALE_DAYS} jours.</p>
          )}
        </section>

        <section className="panel wide">
          <div className="panel-h"><h2>Les trois meilleurs de chaque poste en {app.season}</h2><button className="link" onClick={() => app.go("joueurs")}>Voir tous les joueurs</button></div>
          <div className="tops">
            {ROLES.map((r) => (
              <div key={r.id}>
                <h3><RoleIcon role={r.id} /> {r.label}</h3>
                {app.players.filter((p) => p.role === r.id && p.rank != null).slice(0, 3).map((p) => <Chip key={p.id} p={p} />)}
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
