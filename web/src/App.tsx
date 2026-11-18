import { useCallback, useEffect, useMemo, useState } from "react";
import { API_URL, ROLES, loadHealth, loadSeason, roleLabel } from "./data";
import type { Player } from "./data";
import { AppCtx } from "./app-context";
import type { AppState, View } from "./app-context";
import { STAGES, emptyLineup, starterComp, useStored } from "./store";
import type { Comp, Stage, Tracked } from "./store";
import { Drawer, Palette } from "./ui";
import { CompositionView, HomeView, PipelineView, PlayersView } from "./views";

const VIEWS: { id: View; label: string }[] = [
  { id: "accueil", label: "Accueil" },
  { id: "joueurs", label: "Joueurs" },
  { id: "composition", label: "Composition" },
  { id: "suivi", label: "Suivi" },
];
const SEASONS = [2026, 2025, 2024];

/** Dans l'iframe Streamlit, l'URL n'est pas toujours modifiable : la vue vit dans l'état, le fragment suit quand il peut. */
const setHash = (v: View) => {
  try {
    history.replaceState(null, "", `#${v}`);
  } catch {
    /* document sans URL propre (srcdoc) : sans conséquence */
  }
};

const viewFromHash = (): View => {
  const h = location.hash.replace("#", "") as View;
  return VIEWS.some((v) => v.id === h) ? h : "accueil";
};

export default function App() {
  const [view, setView] = useState<View>(viewFromHash);
  const [season, setSeason] = useStored("erl.season", () => 2026);
  const [players, setPlayers] = useState<Player[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshed, setRefreshed] = useState<string | null>(null);
  const [comps, setComps] = useStored<Comp[]>("erl.comps", () => []);
  const [compId, setCompId] = useStored("erl.comp", () => "c1");
  const [track, setTrack] = useStored<Record<string, Tracked>>("erl.track", () => ({}));
  const [selected, setSelected] = useState<string | null>(null);
  const [palette, setPalette] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    const on = () => setView(viewFromHash());
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    let live = true;
    setPlayers(null);
    setError(null);
    loadSeason(season)
      .then((p) => live && setPlayers(p))
      .catch((e: Error) => live && setError(e.message));
    loadHealth().then((h) => live && setRefreshed(h.data_refreshed_at ?? null)).catch(() => undefined);
    return () => {
      live = false;
    };
  }, [season]);

  // Première visite : une composition pré-remplie, pour que l'écran montre
  // tout de suite à quoi il sert.
  useEffect(() => {
    if (players && comps.length === 0) setComps([starterComp(players)]);
  }, [players, comps.length, setComps]);

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPalette(true);
      }
      if (e.key === "Escape" && !palette) setSelected(null);
    };
    addEventListener("keydown", on);
    return () => removeEventListener("keydown", on);
  }, [palette]);

  const toast = useCallback((m: string) => {
    setToastMsg(m);
    window.setTimeout(() => setToastMsg((cur) => (cur === m ? null : cur)), 2800);
  }, []);

  const comp = comps.find((c) => c.id === compId) ?? comps[0];

  const state = useMemo<AppState | null>(() => {
    if (!players || !comp) return null;
    const byId = new Map(players.map((p) => [p.id, p]));
    const name = (id: string) => byId.get(id)?.name ?? "Ce joueur";
    const updateComp = (fn: (c: Comp) => Comp) => setComps((all) => all.map((c) => (c.id === comp.id ? fn(c) : c)));
    const inComp = (id: string) => ROLES.some((r) => comp.slots[r.id].includes(id));
    const now = () => new Date().toISOString();

    return {
      players, byId, season, comps, comp, track, selected,
      open: setSelected,
      go: (v) => { setView(v); setHash(v); },
      toast,
      inComp,
      place: (id, role, index) => {
        const p = byId.get(id);
        if (!p) return;
        if (p.role !== role) {
          toast(`${p.name} joue ${roleLabel(p.role)}, il ne peut pas aller en ${roleLabel(role)}.`);
          return;
        }
        updateComp((c) => {
          const line = [...c.slots[role]];
          const from = line.indexOf(id);
          if (from === index) return c;
          const occupant = line[index];
          if (from >= 0) line[from] = occupant ?? null;
          line[index] = id;
          return { ...c, slots: { ...c.slots, [role]: line } };
        });
      },
      removeFromComp: (id) => {
        const p = byId.get(id);
        if (!p || !inComp(id)) return;
        updateComp((c) => ({ ...c, slots: { ...c.slots, [p.role]: c.slots[p.role].map((x) => (x === id ? null : x)) } }));
        toast(`${p.name} retiré de ${comp.name}.`);
      },
      addToComp: (id) => {
        const p = byId.get(id);
        if (!p || inComp(id)) return;
        const free = comp.slots[p.role].indexOf(null);
        if (free < 0) {
          toast(`Les 3 places en ${roleLabel(p.role)} sont prises : glisse ${p.name} sur une place pour remplacer.`);
          return;
        }
        updateComp((c) => ({ ...c, slots: { ...c.slots, [p.role]: c.slots[p.role].map((x, i) => (i === free ? id : x)) } }));
        toast(`${p.name} ajouté à ${comp.name}.`);
      },
      follow: (id, stage: Stage = STAGES[0]) => {
        setTrack((t) => ({ ...t, [id]: { note: t[id]?.note ?? "", stage, seen: now() } }));
        toast(`${name(id)} : étape ${stage}.`);
      },
      unfollow: (id) => {
        setTrack((t) => {
          const next = { ...t };
          delete next[id];
          return next;
        });
        toast(`${name(id)} n'est plus suivi.`);
      },
      setNote: (id, note) => setTrack((t) => (t[id] ? { ...t, [id]: { ...t[id], note } } : t)),
      markSeen: (id) => {
        setTrack((t) => (t[id] ? { ...t, [id]: { ...t[id], seen: now() } } : t));
        toast(`${name(id)} marqué comme revu aujourd'hui.`);
      },
    };
  }, [players, comp, comps, season, track, selected, toast, setComps, setTrack]);

  const newComp = () => {
    const id = `c${Date.now()}`;
    setComps((all) => [...all, { id, name: `Composition ${all.length + 1}`, slots: emptyLineup() }]);
    setCompId(id);
    setView("composition");
  };
  const renameComp = (name: string) => comp && setComps((all) => all.map((c) => (c.id === comp.id ? { ...c, name } : c)));
  const deleteComp = () => {
    if (!comp || comps.length < 2) return;
    setComps((all) => all.filter((c) => c.id !== comp.id));
    setCompId(comps.find((c) => c.id !== comp.id)!.id);
    setConfirmDelete(false);
  };

  return (
    <div className={`app ${selected ? "with-drawer" : ""}`}>
      <aside className="side">
        <div className="brand">ERL Scout</div>
        <nav className="nav" aria-label="Sections">
          {VIEWS.map((v) => (
            <button key={v.id} aria-current={view === v.id ? "page" : undefined} onClick={() => { setView(v.id); setHash(v.id); }}>
              <span>{v.label}</span>
              {v.id === "suivi" && Object.keys(track).length > 0 && <em>{Object.keys(track).length}</em>}
            </button>
          ))}
        </nav>
        <div className="side-group">
          <h2>Tes compositions</h2>
          {comps.map((c) => (
            <button key={c.id} aria-current={c.id === comp?.id || undefined} onClick={() => { setCompId(c.id); setView("composition"); }}>{c.name}</button>
          ))}
          <button className="new" onClick={newComp}>+ Nouvelle composition</button>
          {comp && view === "composition" && (
            <div className="comp-edit">
              <label htmlFor="comp-name">Nom</label>
              <input id="comp-name" value={comp.name} onChange={(e) => renameComp(e.target.value)} />
              {comps.length > 1 && (confirmDelete ? (
                <div className="confirm"><span>Supprimer {comp.name} ?</span><button className="link danger" onClick={deleteComp}>Supprimer</button><button className="link" onClick={() => setConfirmDelete(false)}>Annuler</button></div>
              ) : (
                <button className="link danger" onClick={() => setConfirmDelete(true)}>Supprimer cette composition</button>
              ))}
            </div>
          )}
        </div>
        <div className="side-foot">
          <label htmlFor="season">Saison</label>
          <select id="season" value={season} onChange={(e) => setSeason(Number(e.target.value))}>
            {SEASONS.map((s) => <option key={s}>{s}</option>)}
          </select>
          <button className="kbd-btn" onClick={() => setPalette(true)}>Chercher un joueur <kbd>Ctrl K</kbd></button>
        </div>
      </aside>

      <main className="main">
        {error && (
          <div className="state">
            <h1>Les données ne sont pas arrivées</h1>
            <p>{error}. L'API ({API_URL}) est peut-être en train de redémarrer : recharge la page dans une minute.</p>
          </div>
        )}
        {!error && !state && (
          <div className="state">
            <h1>Chargement de la saison {season}</h1>
            <p>Si l'API dormait, elle met jusqu'à trente secondes à se réveiller.</p>
          </div>
        )}
        {state && (
          <AppCtx.Provider value={state}>
            {view === "accueil" && <HomeView refreshed={refreshed} />}
            {view === "joueurs" && <PlayersView />}
            {view === "composition" && <CompositionView />}
            {view === "suivi" && <PipelineView />}
            {selected && <Drawer />}
            {palette && <Palette onClose={() => setPalette(false)} />}
          </AppCtx.Provider>
        )}
      </main>
      {toastMsg && <div className="toast" role="status">{toastMsg}</div>}
    </div>
  );
}

