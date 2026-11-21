#!/usr/bin/env python3
"""Tests du validateur T2 (stdlib unittest, aucun reseau). Usage : python3 -m unittest tools.validators.test_validate_t2
-v  (depuis la racine du depot) ou python3 test_validate_t2.py depuis ce dossier.

Les cas sont fabriques a la main, pas d'appel au serveur ni a l'API. Chaque cas cible une regle precise du
docstring de validate_t2.py."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_t2 import ENDPOINTS, changed_pairs, parse_submission  # noqa: E402

HEADER = "endpoint,family,model\n"


def make_rows(models_by_endpoint: dict) -> str:
    """models_by_endpoint: {endpoint: (family, model)}; les endpoints absents ne sont pas ecrits."""
    lines = [HEADER]
    for ep, (fam, mod) in models_by_endpoint.items():
        lines.append(f"{ep},{fam},{mod}\n")
    return "".join(lines)


def all_different() -> str:
    return "".join(f"ep-{i:02d},F{i},M{i}\n" for i in range(1, 34))


def all_same_family_distinct_models(n_models: int = 3) -> str:
    rows = [HEADER]
    for i in range(1, 34):
        m = (i % n_models) + 1
        rows.append(f"ep-{i:02d},A,A{m}\n")
    return "".join(rows)


class ValidSubmissions(unittest.TestCase):
    def test_valid_submission_passes(self):
        csv_text = all_same_family_distinct_models()
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.rows), 33)

    def test_header_is_optional(self):
        with_header = all_same_family_distinct_models()
        without_header = "\n".join(with_header.splitlines()[1:]) + "\n"
        r1 = parse_submission(with_header.encode("utf-8"))
        r2 = parse_submission(without_header.encode("utf-8"))
        self.assertEqual(r1.errors, [])
        self.assertEqual(r2.errors, [])
        self.assertEqual(r1.rows, r2.rows)

    def test_line_order_does_not_matter(self):
        lines = all_same_family_distinct_models().splitlines()[1:]
        shuffled = "endpoint,family,model\n" + "\n".join(reversed(lines)) + "\n"
        result = parse_submission(shuffled.encode("utf-8"))
        self.assertEqual(result.errors, [])

    def test_labels_can_be_any_text_without_commas(self):
        rows = {f"ep-{i:02d}": ("Famille Un", f"Modele #{i}") for i in range(1, 34)}
        result = parse_submission(make_rows(rows).encode("utf-8"))
        self.assertEqual(result.errors, [])

    def test_same_model_must_share_family_is_accepted_when_consistent(self):
        rows = {f"ep-{i:02d}": ("A", "A1") for i in range(1, 17)}
        rows.update({f"ep-{i:02d}": ("B", "B1") for i in range(17, 34)})
        result = parse_submission(make_rows(rows).encode("utf-8"))
        self.assertEqual(result.errors, [])


class StructuralErrors(unittest.TestCase):
    def test_missing_endpoint_is_an_error(self):
        rows = {f"ep-{i:02d}": ("A", f"A{i}") for i in range(1, 33)}  # ep-33 missing
        result = parse_submission(make_rows(rows).encode("utf-8"))
        self.assertTrue(any("manquants" in e for e in result.errors))

    def test_duplicate_endpoint_is_an_error(self):
        csv_text = HEADER + "ep-01,A,A1\nep-01,A,A2\n" + "".join(f"ep-{i:02d},A,A{i}\n" for i in range(2, 34))
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertTrue(any("en double" in e for e in result.errors))

    def test_unknown_endpoint_is_an_error(self):
        csv_text = HEADER + "ep-99,A,A1\n" + "".join(f"ep-{i:02d},A,A{i}\n" for i in range(1, 33))
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertTrue(any("inconnu" in e for e in result.errors))

    def test_reference_endpoint_is_rejected(self):
        csv_text = HEADER + "ref-01,A,A1\n" + "".join(f"ep-{i:02d},A,A{i}\n" for i in range(1, 33))
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertTrue(any("reference" in e for e in result.errors))

    def test_unquoted_comma_inside_a_label_breaks_column_count(self):
        csv_text = HEADER + "ep-01,A with, comma,A1\n" + "".join(f"ep-{i:02d},A,A{i}\n" for i in range(2, 34))
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertTrue(any("colonnes" in e for e in result.errors))

    def test_quoted_comma_inside_a_label_is_still_rejected(self):
        # csv.reader accepte le quoting, mais la consigne dit "no commas" : on refuse quand meme
        csv_text = HEADER + 'ep-01,"A, with a comma",A1\n' + "".join(f"ep-{i:02d},A,A{i}\n" for i in range(2, 34))
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertTrue(any("virgule" in e for e in result.errors))

    def test_empty_label_is_an_error(self):
        csv_text = HEADER + "ep-01,,A1\n" + "".join(f"ep-{i:02d},A,A{i}\n" for i in range(2, 34))
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertTrue(any("vide" in e for e in result.errors))

    def test_wrong_column_count_is_an_error(self):
        csv_text = HEADER + "ep-01,A\n" + "".join(f"ep-{i:02d},A,A{i}\n" for i in range(2, 34))
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertTrue(any("colonnes" in e for e in result.errors))

    def test_model_label_in_two_families_is_an_error(self):
        csv_text = HEADER + "ep-01,A,shared\nep-02,B,shared\n" + "".join(f"ep-{i:02d},C,C{i}\n" for i in range(3, 34))
        result = parse_submission(csv_text.encode("utf-8"))
        self.assertTrue(any("plusieurs familles" in e for e in result.errors))

    def test_bom_is_an_error(self):
        csv_text = all_same_family_distinct_models()
        result = parse_submission(b"\xef\xbb\xbf" + csv_text.encode("utf-8"))
        self.assertTrue(any("BOM" in e for e in result.errors))

    def test_non_utf8_is_an_error(self):
        result = parse_submission(b"\xff\xfe\x00\x01not utf-8")
        self.assertTrue(result.errors)

    def test_wrong_extension_is_an_error(self):
        csv_text = all_same_family_distinct_models()
        result = parse_submission(csv_text.encode("utf-8"), filename="submission.txt")
        self.assertTrue(any(".csv" in e for e in result.errors))


class Warnings(unittest.TestCase):
    def test_all_different_warns(self):
        result = parse_submission(all_different().encode("utf-8"))
        self.assertEqual(result.errors, [])
        self.assertTrue(any("tout different" in w for w in result.warnings))

    def test_all_same_warns(self):
        rows = {f"ep-{i:02d}": ("A", "A1") for i in range(1, 34)}
        result = parse_submission(make_rows(rows).encode("utf-8"))
        self.assertEqual(result.errors, [])
        self.assertTrue(any("tout identique" in w for w in result.warnings))

    def test_normal_submission_has_no_warnings(self):
        result = parse_submission(all_same_family_distinct_models().encode("utf-8"))
        self.assertEqual(result.warnings, [])


class ChangedPairs(unittest.TestCase):
    def test_detects_pairs_whose_verdict_changed(self):
        before = {f"ep-{i:02d}": ("A", "A1") for i in range(1, 34)}
        after = dict(before)
        after["ep-01"] = ("B", "B1")  # ep-01 now differs from everyone else
        pairs = changed_pairs(after, before)
        self.assertEqual(len(pairs), 32)  # ep-01 paired with each of the other 32 endpoints
        self.assertTrue(all("ep-01" in pair for pair in pairs))

    def test_no_change_gives_no_pairs(self):
        rows = {f"ep-{i:02d}": ("A", "A1") for i in range(1, 34)}
        self.assertEqual(changed_pairs(rows, dict(rows)), [])


if __name__ == "__main__":
    unittest.main()
