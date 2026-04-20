"""Domain validation tests."""

from __future__ import annotations

import pytest

from gaffer.domain.constraints import FPL_RULES, FplRules
from gaffer.domain.enums import Formation, Position
from gaffer.domain.player import Player, PlayerProjection
from gaffer.domain.squad import XI, Bench, Squad, SquadSelection


class TestPosition:
    def test_from_fpl_handles_legacy_gk_alias(self):
        assert Position.from_fpl("GK") == Position.GKP

    def test_from_fpl_passes_through_canonical_values(self):
        assert Position.from_fpl("DEF") == Position.DEF
        assert Position.from_fpl("FWD") == Position.FWD

    def test_from_fpl_rejects_unknown(self):
        with pytest.raises(ValueError):
            Position.from_fpl("STRIKER")


class TestFormation:
    def test_from_counts_resolves_valid_formation(self):
        assert Formation.from_counts(4, 4, 2) == Formation.F_4_4_2

    def test_from_counts_rejects_illegal_triple(self):
        with pytest.raises(ValueError):
            Formation.from_counts(2, 5, 3)  # only 2 DEF — illegal

    def test_property_accessors(self):
        f = Formation.F_3_5_2
        assert f.defenders == 3
        assert f.midfielders == 5
        assert f.forwards == 2


class TestFplRules:
    def test_defaults_populated_post_init(self):
        rules = FplRules()
        assert rules.squad_quota[Position.GKP] == 2
        assert rules.squad_quota[Position.DEF] == 5
        assert rules.squad_quota[Position.MID] == 5
        assert rules.squad_quota[Position.FWD] == 3
        assert sum(rules.squad_quota.values()) == rules.squad_size

    def test_module_singleton_matches_default(self):
        assert FPL_RULES.budget == 100.0


class TestPlayer:
    def test_valid_player(self, player_factory):
        p = player_factory(1, Position.MID, price=8.5)
        assert p.id == 1
        assert p.position == Position.MID

    def test_price_must_be_positive(self):
        with pytest.raises(ValueError):
            Player(id=1, name="X", team="ARS", position=Position.MID, price=0.0)

    def test_chance_of_playing_bounded(self):
        with pytest.raises(ValueError):
            Player(
                id=1, name="X", team="ARS", position=Position.MID,
                price=5.0, chance_of_playing=120,
            )


class TestPlayerProjection:
    def test_valid_projection(self, player_factory):
        proj = PlayerProjection(
            player=player_factory(1, Position.MID),
            gameweek=5,
            expected_points=4.0,
            lower_80=2.0,
            upper_80=7.0,
        )
        assert proj.expected_points == 4.0

    def test_interval_must_bracket_point_estimate(self, player_factory):
        with pytest.raises(ValueError):
            PlayerProjection(
                player=player_factory(1, Position.MID),
                gameweek=5,
                expected_points=4.0,
                lower_80=5.0,  # above the point estimate
                upper_80=7.0,
            )

    def test_gameweek_bounded(self, player_factory):
        with pytest.raises(ValueError):
            PlayerProjection(
                player=player_factory(1, Position.MID),
                gameweek=39,
                expected_points=4.0,
                lower_80=2.0,
                upper_80=7.0,
            )


class TestSquad:
    def test_valid_squad(self, valid_squad_players):
        squad = Squad(players=valid_squad_players)
        assert len(squad.players) == 15
        assert squad.total_price == round(sum(p.price for p in valid_squad_players), 1)

    def test_wrong_size_rejected(self, valid_squad_players):
        with pytest.raises(ValueError, match="exactly 15"):
            Squad(players=valid_squad_players[:14])

    def test_wrong_quota_rejected(self, valid_squad_players, player_factory):
        # Drop one MID, add an extra FWD — keeps size 15 but breaks quotas.
        no_last_mid = [
            p for p in valid_squad_players
            if not (p.position == Position.MID and p == [
                q for q in valid_squad_players if q.position == Position.MID
            ][-1])
        ]
        broken = no_last_mid + [player_factory(99, Position.FWD, price=5.0, name="extra_fwd")]
        assert len(broken) == 15
        with pytest.raises(ValueError, match="MID|FWD"):
            Squad(players=broken)

    def test_over_budget_rejected(self, player_factory):
        teams = ["ARS", "MCI", "LIV", "CHE", "TOT", "NEW"]
        players = []
        pid = 1
        quota = {Position.GKP: 2, Position.DEF: 5, Position.MID: 5, Position.FWD: 3}
        for pos, count in quota.items():
            for _ in range(count):
                players.append(
                    player_factory(pid, pos, team=teams[pid % len(teams)], price=20.0)
                )
                pid += 1
        with pytest.raises(ValueError, match="exceeds budget"):
            Squad(players=players)

    def test_club_cap_enforced(self, player_factory):
        # 4 players from same club triggers the cap.
        players = []
        pid = 1
        quota = {Position.GKP: 2, Position.DEF: 5, Position.MID: 5, Position.FWD: 3}
        for pos, count in quota.items():
            for _ in range(count):
                team = "ARS" if pid <= 4 else f"T{pid}"
                players.append(player_factory(pid, pos, team=team, price=5.0))
                pid += 1
        with pytest.raises(ValueError, match="Club cap"):
            Squad(players=players)

    def test_duplicate_players_rejected(self, player_factory):
        # Custom squad: duplicate one MID, drop the other to keep quotas legal,
        # and spread teams so the club cap doesn't fire first.
        teams = ["ARS", "MCI", "LIV", "CHE", "TOT", "NEW", "WHU", "AVL"]
        positions = (
            [Position.GKP] * 2 + [Position.DEF] * 5
            + [Position.MID] * 5 + [Position.FWD] * 3
        )
        players = [
            player_factory(pid=i + 1, position=pos, team=teams[i % len(teams)], price=5.0)
            for i, pos in enumerate(positions)
        ]
        # Replace players[8] (a MID) with a duplicate of players[7] (also MID).
        dup = players[:8] + [players[7]] + players[9:]
        with pytest.raises(ValueError, match="duplicate"):
            Squad(players=dup)

    def test_by_position(self, valid_squad_players):
        squad = Squad(players=valid_squad_players)
        assert len(squad.by_position(Position.GKP)) == 2
        assert len(squad.by_position(Position.MID)) == 5


class TestXI:
    def _xi(self, players):
        return XI(players=players)

    def test_valid_4_4_2(self, valid_squad_players):
        gk = [p for p in valid_squad_players if p.position == Position.GKP][:1]
        defs = [p for p in valid_squad_players if p.position == Position.DEF][:4]
        mids = [p for p in valid_squad_players if p.position == Position.MID][:4]
        fwds = [p for p in valid_squad_players if p.position == Position.FWD][:2]
        xi = self._xi(gk + defs + mids + fwds)
        assert xi.formation == Formation.F_4_4_2

    def test_wrong_size(self, valid_squad_players):
        with pytest.raises(ValueError, match="exactly 11"):
            self._xi(valid_squad_players[:10])

    def test_two_keepers_rejected(self, valid_squad_players):
        gks = [p for p in valid_squad_players if p.position == Position.GKP]
        defs = [p for p in valid_squad_players if p.position == Position.DEF][:4]
        mids = [p for p in valid_squad_players if p.position == Position.MID][:3]
        fwds = [p for p in valid_squad_players if p.position == Position.FWD][:2]
        with pytest.raises(ValueError):
            self._xi(gks + defs + mids + fwds)  # 2 GK + 4 + 3 + 2 = 11

    def test_invalid_formation_rejected(self, valid_squad_players):
        # 1 GK + 2 DEF + 5 MID + 3 FWD — DEF below the legal min of 3.
        gk = [p for p in valid_squad_players if p.position == Position.GKP][:1]
        defs = [p for p in valid_squad_players if p.position == Position.DEF][:2]
        mids = [p for p in valid_squad_players if p.position == Position.MID][:5]
        fwds = [p for p in valid_squad_players if p.position == Position.FWD][:3]
        with pytest.raises(ValueError):
            self._xi(gk + defs + mids + fwds)


class TestBench:
    def test_valid_bench(self, valid_squad_players):
        gk = [p for p in valid_squad_players if p.position == Position.GKP][1:2]
        defs = [p for p in valid_squad_players if p.position == Position.DEF][4:5]
        mids = [p for p in valid_squad_players if p.position == Position.MID][4:5]
        fwds = [p for p in valid_squad_players if p.position == Position.FWD][2:3]
        bench = Bench(players=gk + defs + mids + fwds)
        assert bench.players[0].position == Position.GKP

    def test_first_slot_must_be_gk(self, valid_squad_players):
        defs = [p for p in valid_squad_players if p.position == Position.DEF][:4]
        with pytest.raises(ValueError, match="goalkeeper"):
            Bench(players=defs)

    def test_no_extra_keeper_outfield(self, valid_squad_players):
        gks = [p for p in valid_squad_players if p.position == Position.GKP]
        # First two slots both GK — illegal.
        with pytest.raises(ValueError):
            Bench(players=gks + [gks[0], gks[0]])  # malformed but exercises the check


class TestSquadSelection:
    def _build(self, players):
        squad = Squad(players=players)
        gk_in_xi = [p for p in players if p.position == Position.GKP][0]
        defs = [p for p in players if p.position == Position.DEF][:4]
        mids = [p for p in players if p.position == Position.MID][:4]
        fwds = [p for p in players if p.position == Position.FWD][:2]
        xi_players = [gk_in_xi] + defs + mids + fwds
        xi = XI(players=xi_players)
        bench_gk = [p for p in players if p.position == Position.GKP][1]
        bench_outfield = [
            p for p in players
            if p not in xi_players and p.position != Position.GKP
        ][:3]
        bench = Bench(players=[bench_gk] + bench_outfield)
        return squad, xi, bench, xi_players

    def test_valid_selection(self, valid_squad_players):
        squad, xi, bench, xi_players = self._build(valid_squad_players)
        sel = SquadSelection(
            squad=squad, xi=xi, bench=bench,
            captain=xi_players[1],
            vice_captain=xi_players[2],
        )
        assert sel.captain.id != sel.vice_captain.id

    def test_captain_must_be_in_xi(self, valid_squad_players):
        squad, xi, bench, xi_players = self._build(valid_squad_players)
        not_in_xi = bench.players[1]
        with pytest.raises(ValueError, match="Captain"):
            SquadSelection(
                squad=squad, xi=xi, bench=bench,
                captain=not_in_xi,
                vice_captain=xi_players[2],
            )

    def test_captain_and_vice_must_differ(self, valid_squad_players):
        squad, xi, bench, xi_players = self._build(valid_squad_players)
        with pytest.raises(ValueError, match="different"):
            SquadSelection(
                squad=squad, xi=xi, bench=bench,
                captain=xi_players[1], vice_captain=xi_players[1],
            )
