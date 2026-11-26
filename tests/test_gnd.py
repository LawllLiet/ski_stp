"""Regression tests for GnD. Run with ``python3 -m unittest discover -s stp_skinny/tests``.

The local test enumerates all assignments directly, so it does not require STP.
"""

from itertools import product
from decimal import Decimal
from pathlib import Path
from random import Random
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import CPMITM_LIN as rules
import model_stp
import solve_stp
import stp_backend as gp


class GuessColorTest(unittest.TestCase):
    def test_exhaustive_color_transition(self):
        model = gp.Model("local_gnd")
        before = rules.Cell(model, "before")
        after, (gb, gr, gbr) = rules.add_gnd(model, before, "guess")
        cells = (before, after)
        variables = [cell_var for cell in cells for cell_var in (cell.b, cell.w, cell.red)]
        variables.extend((gb, gr, gbr))
        # These are all the variables in this isolated relation; every assignment
        # can thus be evaluated directly without an external solver.
        self.assertEqual({v.name for v in variables}, set(model.vars))
        allowed = set()
        for values in product((0, 1), repeat=len(variables)):
            for var, value in zip(variables, values):
                var.X = value
            try:
                model.validate()
            except ValueError:
                continue
            allowed.add((values[:3], values[3:6], values[6:]))
        W, B, R, G, A = (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 0, 0)
        expected = {
            (W, W, (0, 0, 0)), (W, B, (1, 0, 0)),
            (W, R, (0, 1, 0)), (W, G, (0, 0, 1)),
            *((color, color, (0, 0, 0)) for color in (B, R, G, A)),
        }
        self.assertEqual(allowed, expected)


class GuessIntegrationTest(unittest.TestCase):
    def test_inputs_are_inserted_in_both_mc_directions_only(self):
        calls, forward_calls, backward_calls = [], [], []
        orig_gnd, orig_forward, orig_backward = (
            model_stp.add_gnd, model_stp.add_mc_forward, model_stp.add_mc_backward
        )

        def capture_guess(model, before, name):
            after, guesses = orig_gnd(model, before, name)
            calls.append((before, after, guesses))
            return after, guesses

        def capture_forward(model, zin, xout, name):
            forward_calls.append((zin, xout))
            return orig_forward(model, zin, xout, name)

        def capture_backward(model, zin, xout, name):
            backward_calls.append((zin, xout))
            return orig_backward(model, zin, xout, name)

        with patch.object(model_stp, "add_gnd", side_effect=capture_guess), \
             patch.object(model_stp, "add_mc_forward", side_effect=capture_forward), \
             patch.object(model_stp, "add_mc_backward", side_effect=capture_backward):
            _, index = model_stp.build_model(4, 0, 2, enable_gnd=True)

        self.assertEqual(len(calls), 16 * (len(index.forward) + len(index.backward)))
        self.assertEqual(len(index.gnd), len(calls))
        for entry, (before, after, guesses) in zip(index.gnd, calls):
            self.assertIs(entry["before"], before)
            self.assertIs(entry["after"], after)
            for key, flag in zip(("gb", "gr", "gbr"), guesses):
                self.assertIs(entry[key], flag)
            rnd, j = entry["round"], entry["cell"]
            if entry["direction"] == "F":
                self.assertIn(rnd, index.forward)
                self.assertEqual(entry["state"], "Z")
                self.assertIs(before, index.Z[rnd][j])
            else:
                self.assertEqual(entry["direction"], "B")
                self.assertIn(rnd, index.backward)
                self.assertEqual(entry["state"], "X")
                self.assertIs(before, index.X[(rnd + 1) % index.tr][j])
        for pos, guesses in enumerate((index.guess_b, index.guess_r, index.guess_br)):
            self.assertEqual({v.name for v in guesses}, {c[2][pos].name for c in calls})
        by_before = {id(before): after for before, after, _ in calls}
        self.assertEqual(len(by_before), len(calls))
        for rnd, (zin, xout) in zip(index.forward, forward_calls):
            self.assertTrue(all(zin[j] is by_before[id(index.Z[rnd][j])] for j in range(16)))
            self.assertTrue(all(xout[j] is index.X[(rnd + 1) % index.tr][j] for j in range(16)))
        for rnd, (zin, xout) in zip(index.backward, backward_calls):
            self.assertTrue(all(zin[j] is index.Z[rnd][j] for j in range(16)))
            self.assertTrue(all(xout[j] is by_before[id(index.X[(rnd + 1) % index.tr][j])] for j in range(16)))

    def test_disabled_keeps_original_mc_inputs(self):
        forward_calls, backward_calls = [], []
        orig_forward, orig_backward = model_stp.add_mc_forward, model_stp.add_mc_backward

        def capture_forward(model, zin, xout, name):
            forward_calls.append((zin, xout))
            return orig_forward(model, zin, xout, name)

        def capture_backward(model, zin, xout, name):
            backward_calls.append((zin, xout))
            return orig_backward(model, zin, xout, name)

        with patch.object(model_stp, "add_gnd", wraps=model_stp.add_gnd) as guess, \
             patch.object(model_stp, "add_mc_forward", side_effect=capture_forward), \
             patch.object(model_stp, "add_mc_backward", side_effect=capture_backward):
            _, index = model_stp.build_model(4, 0, 2, enable_gnd=False)
        guess.assert_not_called()
        self.assertEqual(len(index.gnd), 0)
        self.assertEqual(len(index.guess_b) + len(index.guess_r) + len(index.guess_br), 0)
        for rnd, (zin, xout) in zip(index.forward, forward_calls):
            self.assertIs(zin, index.Z[rnd])
            self.assertIs(xout, index.X[(rnd + 1) % index.tr])
        for rnd, (zin, xout) in zip(index.backward, backward_calls):
            self.assertIs(zin, index.Z[rnd])
            self.assertIs(xout, index.X[(rnd + 1) % index.tr])


class GuessObjectiveTest(unittest.TestCase):
    def test_positive_gain_constraints_and_boundaries(self):
        model, index = model_stp.build_model(4, 0, 2, enable_gnd=True)
        expected = (
            index.GDeg1 - index.n_gr - index.n_gbr >= 1,
            index.GDeg2 - index.n_gb - index.n_gbr >= 1,
            index.GMat - index.n_gb - index.n_gr - index.n_gbr >= 1,
        )

        # Locate each exact constraint in the base model; no solver is needed to
        # evaluate its boundary independently of the other model constraints.
        selected = []
        for op, expr in expected:
            matches = [c for c in model.constraints
                       if len(c) == 2 and c[0] == op
                       and c[1].terms == expr.terms and c[1].constant == expr.constant]
            self.assertEqual(len(matches), 1)
            selected.append(matches[0])
        for guesses in (index.guess_b, index.guess_r, index.guess_br):
            self.assertTrue(guesses)
            guesses[0].X = 1
        for op, expr in selected:
            self.assertEqual(op, 'bvsge')
            degree = next(model.vars[name] for name in expr.terms
                          if name in ('GDeg1', 'GDeg2', 'GMat'))
            deduction = -sum(coef * model.vars[name].X
                             for name, coef in expr.terms.items() if name != degree.name)
            for remaining, valid in ((0, False), (1, True), (2, True)):
                degree.X = deduction + remaining
                value = expr.constant + sum(coef * model.vars[name].X
                                            for name, coef in expr.terms.items())
                self.assertEqual(value >= 0, valid)
        self.assertLess(2 * solve_stp.A, 1)

    def test_integer_thresholds_match_requested_objective(self):
        model = gp.Model("thresholds")
        fields = {}
        for name, limit in (("nlin", 2), ("GDeg1", 48), ("GDeg2", 48),
                            ("GMat", 16), ("n_gb", 32), ("n_gr", 32), ("n_gbr", 32)):
            fields[name] = model.addVar(lb=0, ub=limit, vtype=gp.GRB.INTEGER, name=name)
        index = SimpleNamespace(**fields)
        random = Random(1294)
        for case, (n, n0, _) in enumerate(product(range(3), range(3), range(400))):
            db, dr, dm = random.randrange(49), random.randrange(49), random.randrange(17)
            gb, gr, gbr = random.randrange(12), random.randrange(12), random.randrange(12)
            k = random.randrange(-10, 17)
            for name, value in zip(fields, (n, db, dr, dm, gb, gr, gbr)):
                fields[name].X = value
            actual = True
            for op, expr in solve_stp.conditions(index, n, (k, n0)):
                value = expr.constant + sum(coef * model.vars[name].X for name, coef in expr.terms.items())
                actual &= value == 0 if op == "=" else value <= 0 if op == "bvsle" else value >= 0
            alpha = solve_stp.A
            score = min(Decimal(db - gr - gbr) - alpha * n,
                        Decimal(dr - gb - gbr) - alpha * n,
                        Decimal(dm - gb - gr - gbr))
            self.assertEqual(actual, score >= Decimal(k) - alpha * n0, msg=f"case={case}")

    def test_candidate_boundaries_are_positive(self):
        vals = solve_stp.candidates()
        self.assertEqual(set(vals), {(k, n0) for k in range(1, 17) for n0 in range(3)})
        self.assertEqual(vals[0], (1, 2))
        self.assertEqual(vals[-1], (16, 0))
        numeric = [Decimal(k) - solve_stp.A * n for k, n in vals]
        self.assertGreater(numeric[0], 0)
        self.assertEqual(numeric, sorted(numeric))


if __name__ == "__main__":
    unittest.main()
