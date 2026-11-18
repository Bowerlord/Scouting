import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent, KeyboardEvent, ReactNode } from "react";
import { AGENT_URL, level, loadSimilar, rankText, roleLabel, rolePlural, sentence, stats, tier } from "./data";
import type { Player, Role } from "./data";
import { DRAG_TYPE, useApp } from "./app-context";
import { STAGES, daysSince } from "./store";

const ICONS: Record<Role, ReactNode> = {
  top: (<><path d="M5 19V5h14" /><path d="M9 15V9h6" opacity=".5" /></>),
  jng: (<><path d="M12 3c5 4 6 10 0 18c-6-8-5-14 0-18z" /><path d="M12 8v13" /></>),
  mid: (<><path d="M12 3l9 9-9 9-9-9z" /><path d="M8 16l8-8" /></>),
  bot: (<><path d="M5 19h14V5" /><path d="M9 15h6V9" opacity=".5" /></>),
  sup: (<><circle cx="12" cy="12" r="8" /><path d="M12 8v8M8 12h8" /></>),
};

export function RoleIcon({ role }: { role: Role }) {
  return (
    <svg className="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICONS[role]}
    </svg>
  );
}

export function Pips({ p }: { p: Player }) {
  return (
    <span className="pips" title="Avance à 15 min, dégâts, farm">
      {stats(p).map(([l, v]) => (
        <i key={l} className={tier(v)} title={`${l} : ${level(v)}`} />
      ))}
    </span>
  );
}

/** Rend un élément déplaçable et cliquable pour ouvrir la fiche. */
export function useDraggable(id: string) {
  const { open, selected } = useApp();
  return {
    draggable: true,
    tabIndex: 0,
    "data-sel": selected === id || undefined,
    onDragStart: (e: DragEvent) => {
      e.dataTransfer.setData(DRAG_TYPE, id);
      e.dataTransfer.effectAllowed = "move";
      (e.currentTarget as HTMLElement).classList.add("dragging");
    },
    onDragEnd: (e: DragEvent) => (e.currentTarget as HTMLElement).classList.remove("dragging"),
    onClick: () => open(id),
    onKeyDown: (e: KeyboardEvent) => {
      if (e.key === "Enter") open(id);
    },
  };
}

/** Une zone où l'on peut déposer un joueur. */
export function useDrop(onDrop: (id: string) => void) {
  const [over, setOver] = useState(false);
  return {
    "data-over": over || undefined,
    onDragOver: (e: DragEvent) => {
      e.preventDefault();
      setOver(true);
    },
    onDragLeave: () => setOver(false),
    onDrop: (e: DragEvent) => {
      e.preventDefault();
      setOver(false);
      const id = e.dataTransfer.getData(DRAG_TYPE);
      if (id) onDrop(id);
    },
  };
}

export function Drawer() {
  const app = useApp();
  const p = app.selected ? app.byId.get(app.selected) : undefined;
  const [similar, setSimilar] = useState<Awaited<ReturnType<typeof loadSimilar>> | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!p) return;
    let live = true;
    setSimilar(null);
    setFailed(false);
    loadSimilar(p.key)
      .then((s) => live && setSimilar(s.filter((x) => x.key !== p.key)))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [p]);

  if (!p) return null;
  const t = app.track[p.id];
  const inComp = app.inComp(p.id);

  return (
    <aside className="drawer" aria-label={`Fiche de ${p.name}`}>
      <button className="drawer-x" onClick={() => app.open(null)} aria-label="Fermer la fiche">×</button>
      <div className="dw-role"><RoleIcon role={p.role} /> {roleLabel(p.role)}, {p.team}, {p.league}</div>
      <h2 className="dw-name">{p.name}</h2>
      <div className="dw-rank"><b>{rankText(p)}</b> en {app.season}, potentiel LEC {p.score}/100</div>
      <div className="dw-tags">
        <span className="tag">{p.archetype}</span>
        {p.promoted && <span className="tag ok">Déjà monté en LEC</span>}
      </div>
      <p className="dw-phrase">{sentence(p)}</p>

      <div className="dw-stats">
        {stats(p).map(([l, v]) => (
          <div key={l} className={`dw-stat ${tier(v)}`}>
            <div className="h"><span>{l}</span><b>{level(v)}</b></div>
            <div className="bar"><i style={{ width: `${v ?? 0}%` }} /></div>
            {v != null && <small>mieux que {v} % des {rolePlural(p.role)} de {app.season}</small>}
          </div>
        ))}
      </div>

      <div className="dw-facts">
        <div><b>{p.winRate} %</b><span>victoires</span></div>
        <div><b>{p.games}</b><span>matchs</span></div>
        <div><b>{p.pool ?? "–"}</b><span>champions</span></div>
      </div>
      {p.games < 30 && <div className="dw-warn">Peu de matchs joués : chiffres à confirmer.</div>}

      <div className="dw-actions">
        {t ? (
          <label className="stage-pick">
            Suivi
            <select value={t.stage} onChange={(e) => app.follow(p.id, e.target.value as (typeof STAGES)[number])}>
              {STAGES.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
        ) : (
          <button className="btn primary" onClick={() => app.follow(p.id)}>Suivre ce joueur</button>
        )}
        <button className="btn" onClick={() => (inComp ? app.removeFromComp(p.id) : app.addToComp(p.id))}>
          {inComp ? "Retirer de la compo" : "Ajouter à la compo"}
        </button>
      </div>

      {t && (
        <div className="dw-note">
          <div className="h">
            <span>Notes</span>
            <span className="muted">vu il y a {daysSince(t.seen)} j</span>
          </div>
          <textarea value={t.note} onChange={(e) => app.setNote(p.id, e.target.value)} placeholder="Ce que tu as vu en le regardant jouer" />
          <div className="dw-note-act">
            <button className="link" onClick={() => app.markSeen(p.id)}>Je viens de le revoir</button>
            <button className="link danger" onClick={() => app.unfollow(p.id)}>Ne plus suivre</button>
          </div>
        </div>
      )}

      <div className="dw-sim">
        <h3>Profils proches</h3>
        {failed && <p className="muted">Les profils proches n'ont pas pu être chargés.</p>}
        {!failed && !similar && <p className="muted">Recherche des profils proches…</p>}
        {similar?.map((s) => {
          const id = `${s.key}|${s.role}`;
          const known = app.byId.has(id);
          return (
            <button key={id} className="sim-row" disabled={!known} onClick={() => app.open(id)}>
              <span>{app.byId.get(id)?.name ?? s.name}</span>
              <span className="muted">{s.resemblance} % de ressemblance</span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

export function Palette({ onClose }: { onClose: () => void }) {
  const app = useApp();
  const [q, setQ] = useState("");
  const [cur, setCur] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => input.current?.focus(), []);

  const results = useMemo(() => {
    const s = q.toLowerCase().trim();
    return app.players
      .filter((p) => !s || [p.name, p.team, p.league, roleLabel(p.role), p.archetype].join(" ").toLowerCase().includes(s))
      .slice(0, 8);
  }, [q, app.players]);

  const choose = (p?: Player) => {
    if (!p) return;
    app.open(p.id);
    onClose();
  };

  return (
    <div className="palette-back" onClick={onClose}>
      <div className="palette" role="dialog" aria-label="Chercher un joueur" onClick={(e) => e.stopPropagation()}>
        <input
          ref={input}
          value={q}
          placeholder="Nom, équipe, ligue, poste, style de jeu"
          aria-label="Chercher un joueur"
          onChange={(e) => {
            setQ(e.target.value);
            setCur(0);
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setCur((c) => Math.min(c + 1, results.length - 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setCur((c) => Math.max(c - 1, 0)); }
            if (e.key === "Enter") choose(results[cur]);
            if (e.key === "Escape") onClose();
          }}
        />
        <div className="palette-res">
          {results.length === 0 && <div className="palette-empty">Aucun joueur ne correspond. Essaie un nom d'équipe ou un poste.</div>}
          {results.map((p, i) => (
            <button key={p.id} className={`palette-row ${i === cur ? "cur" : ""}`} onMouseEnter={() => setCur(i)} onClick={() => choose(p)}>
              <RoleIcon role={p.role} />
              <b>{p.name}</b>
              <span>{p.team}, {p.league}, {p.archetype}</span>
              <span className="muted">{p.rank ? `#${p.rank}` : ""}</span>
            </button>
          ))}
        </div>
        <a className="palette-ask" href={AGENT_URL} target="_blank" rel="noreferrer">
          Poser une question en langage naturel à l'agent
        </a>
        <div className="palette-ft"><span>↑↓ pour choisir</span><span>Entrée pour ouvrir la fiche</span><span>Échap pour fermer</span></div>
      </div>
    </div>
  );
}
