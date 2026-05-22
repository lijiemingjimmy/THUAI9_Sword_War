from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OUR_KNOWN_SIGNATURES = {
    (3, 20, 4, 90),
    (1, 22, 4, 94),
    (2, 22, 4, 94),
    (3, 29, 0, 108),
}

OUR_KNOWN_POSITION_SETS = {
    frozenset({(7, 8), (10, 9), (13, 7)}),
    frozenset({(7, 12), (10, 11), (13, 13)}),
    frozenset({(5, 3), (10, 2), (15, 3)}),
    frozenset({(5, 16), (10, 17), (15, 16)}),
}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def soldier_signature(soldier: dict[str, Any]) -> tuple[int, int, int, int]:
    stats = soldier.get("stats", {})
    return (
        int(soldier.get("soldierType", -1)),
        int(stats.get("strength", -1)),
        int(stats.get("intelligence", -1)),
        int(stats.get("health", -1)),
    )


def infer_our_camp(soldiers: list[dict[str, Any]]) -> str | None:
    by_camp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for soldier in soldiers:
        by_camp[soldier.get("camp", "")].append(soldier)

    best_camp = None
    best_score = -1
    for camp, camp_soldiers in by_camp.items():
        score = sum(soldier_signature(s) in OUR_KNOWN_SIGNATURES for s in camp_soldiers)
        if score > best_score:
            best_camp = camp
            best_score = score
    if best_score <= 0:
        return None

    tied = [
        camp for camp, camp_soldiers in by_camp.items()
        if sum(soldier_signature(s) in OUR_KNOWN_SIGNATURES for s in camp_soldiers) == best_score
    ]
    return best_camp if len(tied) == 1 else None


def camp_position_set(soldiers: list[dict[str, Any]], camp: str) -> frozenset[tuple[int, int]]:
    points = []
    for soldier in soldiers:
        if soldier.get("camp") != camp:
            continue
        pos = soldier.get("position", {})
        points.append((int(pos.get("x", -1)), int(pos.get("z", -1))))
    return frozenset(points)


def infer_our_camp_by_positions(soldiers: list[dict[str, Any]]) -> str | None:
    candidates = []
    for camp in sorted({s.get("camp", "") for s in soldiers}):
        if camp_position_set(soldiers, camp) in OUR_KNOWN_POSITION_SETS:
            candidates.append(camp)
    return candidates[0] if len(candidates) == 1 else None


def final_alive_by_camp(data: dict[str, Any], id_to_camp: dict[int, str]) -> Counter:
    alive = Counter()
    if not data.get("gameRounds"):
        for sid, camp in id_to_camp.items():
            alive[camp] += 1
        return alive

    last_stats = data["gameRounds"][-1].get("stats", [])
    seen = set()
    for row in last_stats:
        sid = int(row.get("soldierId", -1))
        seen.add(sid)
        if as_bool(row.get("survived")):
            alive[id_to_camp.get(sid, "?")] += 1

    # Some replay frames only include the active soldier. Fall back to death actions if needed.
    if len(seen) < len(id_to_camp):
        dead = set()
        for round_data in data.get("gameRounds", []):
            for action in round_data.get("actions", []):
                if action.get("actionType") == "Death":
                    dead.add(int(action.get("soldierId", -1)))
        alive = Counter()
        for sid, camp in id_to_camp.items():
            if sid not in dead:
                alive[camp] += 1
    return alive


def compact_counter(counter: Counter) -> str:
    return ";".join(f"{key}x{count}" for key, count in sorted(counter.items(), key=lambda kv: str(kv[0])))


def camp_builds(soldiers: list[dict[str, Any]], camp: str) -> Counter:
    return Counter(soldier_signature(s) for s in soldiers if s.get("camp") == camp)


def initial_positions(soldiers: list[dict[str, Any]], camp: str) -> str:
    points = []
    for soldier in soldiers:
        if soldier.get("camp") != camp:
            continue
        pos = soldier.get("position", {})
        # Replay uses x/z as board coordinates and y as rendered height.
        points.append((int(soldier.get("ID", -1)), int(pos.get("x", -1)), int(pos.get("z", -1))))
    return ";".join(f"id{sid}@({x},{z})" for sid, x, z in sorted(points))


def analyze_file(path: Path, forced_our_camp: str | None = None, our_result: str | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    soldiers = data.get("soldiersData", [])
    id_to_camp = {int(s["ID"]): s.get("camp", "?") for s in soldiers}
    id_to_sig = {int(s["ID"]): soldier_signature(s) for s in soldiers}
    our_camp = forced_our_camp or infer_our_camp_by_positions(soldiers) or infer_our_camp(soldiers)
    enemy_camp = None

    raw_damage = Counter()
    attacks = Counter()
    target_counts = Counter()
    attacked_by_target = Counter()
    action_counts = Counter()
    movement_then_attack = Counter()
    deaths = Counter()
    death_rounds: dict[int, int] = {}
    first_attack_target: dict[str, int] = {}
    first_round_attacks = Counter()
    first_death_round = None
    first_death_camp = None
    first_death_id = None

    for round_data in data.get("gameRounds", []):
        round_no = int(round_data.get("roundNumber", 0))
        actions = round_data.get("actions", [])
        moved = set()
        attacked = set()
        for action in actions:
            kind = action.get("actionType", "")
            sid = int(action.get("soldierId", -1))
            camp = id_to_camp.get(sid, "?")
            action_counts[(camp, kind)] += 1

            if kind == "Movement":
                moved.add(sid)
            elif kind == "Attack":
                attacked.add(sid)
                attacks[camp] += 1
                tid = int(action.get("targetId", -1))
                target_counts[(camp, tid)] += 1
                attacked_by_target[(id_to_camp.get(tid, "?"), tid)] += 1
                if camp not in first_attack_target:
                    first_attack_target[camp] = tid
                if round_no <= 6:
                    first_round_attacks[(camp, tid)] += 1
                for damage in action.get("damageDealt", []):
                    raw_damage[camp] += int(damage.get("damage", 0))
            elif kind == "Death":
                deaths[camp] += 1
                death_rounds[sid] = round_no
                if first_death_round is None:
                    first_death_round = round_no
                    first_death_camp = camp
                    first_death_id = sid
        for sid in moved & attacked:
            movement_then_attack[id_to_camp.get(sid, "?")] += 1

    alive = final_alive_by_camp(data, id_to_camp)
    winner_camp = None
    camps = sorted({c for c in id_to_camp.values() if c != "?"})
    if len(camps) == 2 and alive[camps[0]] != alive[camps[1]]:
        winner_camp = camps[0] if alive[camps[0]] > alive[camps[1]] else camps[1]

    if our_camp is None and our_result in {"win", "loss"} and winner_camp in camps:
        if our_result == "win":
            our_camp = winner_camp
        else:
            our_camp = next((camp for camp in camps if camp != winner_camp), None)

    if our_camp:
        enemy_camps = sorted({c for c in id_to_camp.values() if c != our_camp})
        enemy_camp = enemy_camps[0] if enemy_camps else None

    camps_data = {}
    for camp in sorted({c for c in id_to_camp.values() if c != "?"}):
        camp_targets = Counter({tid: n for (c, tid), n in target_counts.items() if c == camp})
        camp_top_target = camp_targets.most_common(1)[0][0] if camp_targets else ""
        camps_data[camp] = {
            "builds": compact_counter(camp_builds(soldiers, camp)),
            "positions": initial_positions(soldiers, camp),
            "raw_damage": raw_damage.get(camp, 0),
            "attacks": attacks.get(camp, 0),
            "move_attack": movement_then_attack.get(camp, 0),
            "deaths": deaths.get(camp, 0),
            "first_attack_target": first_attack_target.get(camp, ""),
            "top_target": camp_top_target,
        }

    enemy_signatures = camp_builds(soldiers, enemy_camp) if enemy_camp is not None else Counter()
    our_signatures = camp_builds(soldiers, our_camp) if our_camp is not None else Counter()

    enemy_focus_target = None
    if enemy_camp is not None:
        enemy_targets = Counter({tid: n for (camp, tid), n in target_counts.items() if camp == enemy_camp})
        if enemy_targets:
            enemy_focus_target, _ = enemy_targets.most_common(1)[0]

    return {
        "file": str(path),
        "folder": path.parent.name,
        "rounds": len(data.get("gameRounds", [])),
        "our_camp": our_camp or "?",
        "enemy_camp": enemy_camp or "?",
        "winner_camp": winner_camp or "?",
        "our_win": bool(our_camp and winner_camp == our_camp),
        "first_death_round": first_death_round or "",
        "first_death_camp": first_death_camp or "?",
        "first_death_id": first_death_id if first_death_id is not None else "",
        "our_alive": alive.get(our_camp, 0) if our_camp else "",
        "enemy_alive": alive.get(enemy_camp, 0) if enemy_camp else "",
        "our_raw_damage": raw_damage.get(our_camp, 0) if our_camp else 0,
        "enemy_raw_damage": raw_damage.get(enemy_camp, 0) if enemy_camp else 0,
        "our_attacks": attacks.get(our_camp, 0) if our_camp else 0,
        "enemy_attacks": attacks.get(enemy_camp, 0) if enemy_camp else 0,
        "our_move_attack": movement_then_attack.get(our_camp, 0) if our_camp else 0,
        "enemy_move_attack": movement_then_attack.get(enemy_camp, 0) if enemy_camp else 0,
        "enemy_focus_target": enemy_focus_target if enemy_focus_target is not None else "",
        "red_builds": camps_data.get("Red", {}).get("builds", ""),
        "blue_builds": camps_data.get("Blue", {}).get("builds", ""),
        "red_positions": camps_data.get("Red", {}).get("positions", ""),
        "blue_positions": camps_data.get("Blue", {}).get("positions", ""),
        "red_raw_damage": camps_data.get("Red", {}).get("raw_damage", 0),
        "blue_raw_damage": camps_data.get("Blue", {}).get("raw_damage", 0),
        "red_attacks": camps_data.get("Red", {}).get("attacks", 0),
        "blue_attacks": camps_data.get("Blue", {}).get("attacks", 0),
        "red_move_attack": camps_data.get("Red", {}).get("move_attack", 0),
        "blue_move_attack": camps_data.get("Blue", {}).get("move_attack", 0),
        "red_deaths": camps_data.get("Red", {}).get("deaths", 0),
        "blue_deaths": camps_data.get("Blue", {}).get("deaths", 0),
        "red_first_attack_target": camps_data.get("Red", {}).get("first_attack_target", ""),
        "blue_first_attack_target": camps_data.get("Blue", {}).get("first_attack_target", ""),
        "red_top_target": camps_data.get("Red", {}).get("top_target", ""),
        "blue_top_target": camps_data.get("Blue", {}).get("top_target", ""),
        "first6_focus": compact_counter(first_round_attacks),
        "death_rounds": compact_counter(Counter({sid: rnd for sid, rnd in death_rounds.items()})),
        "our_signatures": compact_counter(our_signatures),
        "enemy_signatures": compact_counter(enemy_signatures),
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    by_folder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_folder[row["folder"]].append(row)

    print(f"Analyzed {len(rows)} replays")
    print()
    for folder, items in sorted(by_folder.items()):
        wins = sum(1 for r in items if r["our_win"])
        first_deaths = Counter(r["first_death_camp"] for r in items)
        enemy_sigs = Counter(r["enemy_signatures"] for r in items)
        avg_rounds = sum(int(r["rounds"]) for r in items) / max(1, len(items))
        avg_our_damage = sum(int(r["our_raw_damage"]) for r in items) / max(1, len(items))
        avg_enemy_damage = sum(int(r["enemy_raw_damage"]) for r in items) / max(1, len(items))
        avg_our_attacks = sum(int(r["our_attacks"]) for r in items) / max(1, len(items))
        avg_enemy_attacks = sum(int(r["enemy_attacks"]) for r in items) / max(1, len(items))
        red_wins = sum(1 for r in items if r["winner_camp"] == "Red")
        blue_wins = sum(1 for r in items if r["winner_camp"] == "Blue")
        avg_red_damage = sum(int(r["red_raw_damage"]) for r in items) / max(1, len(items))
        avg_blue_damage = sum(int(r["blue_raw_damage"]) for r in items) / max(1, len(items))
        avg_red_attacks = sum(int(r["red_attacks"]) for r in items) / max(1, len(items))
        avg_blue_attacks = sum(int(r["blue_attacks"]) for r in items) / max(1, len(items))
        red_first_targets = Counter(r["red_first_attack_target"] for r in items)
        blue_first_targets = Counter(r["blue_first_attack_target"] for r in items)
        print(f"[{folder}] games={len(items)} wins={wins} avg_rounds={avg_rounds:.1f}")
        print(f"  camp_wins Red/Blue={red_wins}/{blue_wins}")
        print(f"  first_death={dict(first_deaths)}")
        print(f"  avg_raw_damage ours/enemy={avg_our_damage:.1f}/{avg_enemy_damage:.1f}")
        print(f"  avg_attacks ours/enemy={avg_our_attacks:.1f}/{avg_enemy_attacks:.1f}")
        print(f"  avg_raw_damage Red/Blue={avg_red_damage:.1f}/{avg_blue_damage:.1f}")
        print(f"  avg_attacks Red/Blue={avg_red_attacks:.1f}/{avg_blue_attacks:.1f}")
        print(f"  first_attack_target Red={dict(red_first_targets)} Blue={dict(blue_first_targets)}")
        print("  common_enemy_builds:")
        for sig, count in enemy_sigs.most_common(5):
            print(f"    {count:>2}x {sig}")
        print("  common_red_builds:")
        for sig, count in Counter(r["red_builds"] for r in items).most_common(3):
            print(f"    {count:>2}x {sig}")
        print("  common_blue_builds:")
        for sig, count in Counter(r["blue_builds"] for r in items).most_common(3):
            print(f"    {count:>2}x {sig}")
        print("  outcome_splits:")
        for winner in ["Red", "Blue", "?"]:
            split = [r for r in items if r["winner_camp"] == winner]
            if not split:
                continue
            split_first_deaths = Counter(r["first_death_camp"] for r in split)
            split_red_first = Counter(r["red_first_attack_target"] for r in split)
            split_blue_first = Counter(r["blue_first_attack_target"] for r in split)
            split_red_attacks = sum(int(r["red_attacks"]) for r in split) / len(split)
            split_blue_attacks = sum(int(r["blue_attacks"]) for r in split) / len(split)
            split_red_move_attack = sum(int(r["red_move_attack"]) for r in split) / len(split)
            split_blue_move_attack = sum(int(r["blue_move_attack"]) for r in split) / len(split)
            print(
                f"    winner={winner} n={len(split)} "
                f"first_death={dict(split_first_deaths)} "
                f"avg_attacks R/B={split_red_attacks:.1f}/{split_blue_attacks:.1f} "
                f"move+attack R/B={split_red_move_attack:.1f}/{split_blue_move_attack:.1f}"
            )
            print(f"      first_target Red={dict(split_red_first)} Blue={dict(split_blue_first)}")
        print()

    all_enemy_sigs = Counter(r["enemy_signatures"] for r in rows)
    print("[overall] common enemy builds")
    for sig, count in all_enemy_sigs.most_common(10):
        print(f"  {count:>2}x {sig}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Skies-of-Tactics replay JSON files.")
    parser.add_argument("paths", nargs="+", help="Replay files or directories containing replay JSON.")
    parser.add_argument("--csv", dest="csv_path", help="Optional path for per-game CSV output.")
    parser.add_argument("--our-camp", choices=["Red", "Blue"], help="Force our camp when both sides have mirror builds.")
    parser.add_argument("--our-result", choices=["win", "loss"], help="Infer our camp from the final result.")
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)

    rows = [analyze_file(path, args.our_camp, args.our_result) for path in files]
    print_summary(rows)

    if args.csv_path:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(args.csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f"Wrote CSV: {args.csv_path}")


if __name__ == "__main__":
    main()
