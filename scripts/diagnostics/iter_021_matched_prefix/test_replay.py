from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from replay import (
    binding_join,
    binding_projection_query,
    fetch_binding_projection,
    attach_binding_projection,
    BoundaryCapture,
    build_pair,
    capture_cor_method_outputs,
    is_complete_prefix,
    resume_jsonl,
    seal_prefix,
)


ROOT = Path(__file__).resolve().parents[3]
HASH_MANIFEST = ROOT / "outputs/experiments/fresh50_continuous_agent/RESULTS/iter_021_matched_prefix_replay/PROTECTED_SOURCE_HASHES.json"


def fixture(**overrides):
    record = {
        "question_id": "fixture-q1",
        "repeat": 1,
        "model_settings": {"model": "saved-fixture", "temperature": 0, "max_completion_tokens": 128},
        "boundary_depth": 3,
        "boundary_action": "Forward",
        "route": "cor",
        "decisions": [{"stage": "reasoning", "prompt": "fixture prompt", "response": "Forward"}],
        "kg_cache": [{"query_hash": "prefix-query-sha", "status": "ok", "result_ids": ["m.a", "m.b", "m.c"]}],
        "method_events": [
            {"stage": "relation_search", "relation_chain": [], "head_relation_ids": ["r.one"], "tail_relation_ids": []},
            {"stage": "relation_prune", "relation_chain": [], "candidate_relation_ids_before_prune": ["r.one"], "selected_relations": [{"id": "r.one", "direction": "OUTGOING"}]},
            {"stage": "entity_search", "relation_chain": [{"id": "r.one"}], "candidate_entity_ids": ["m.c"]},
            {"stage": "reasoning", "relation_chain": [{"id": "r.one"}], "action": "Forward", "response": "Forward"},
        ],
        "sparql_history": [{"sparql": "SELECT ?targetEntity WHERE {}", "status": "ok"}],
        "dfs_stack": [
            {"path": [{"kind": "entity", "id": "m.root"}], "depth": 0},
            {"path": [{"kind": "entity", "id": "m.root"}, {"kind": "relation", "id": "r.one", "direction": "OUTGOING"}], "depth": 1},
        ],
        "current_path": [
            {"kind": "entity", "id": "m.root"},
            {"kind": "relation", "id": "r.one", "direction": "OUTGOING"},
            {"kind": "relation", "id": "r.two", "direction": "INCOMING"},
            {"kind": "relation", "id": "r.three", "direction": "OUTGOING"},
        ],
        "active_binding_rows": [{"?e1": "m.a", "?e2": "m.b", "?e3": "m.c"}],
        "frontier_ids": ["m.c"],
        "visited_node_ids": ["m.root", "m.a", "m.b", "m.c"],
        "candidate_relation_ids_before_prune": ["r.next", "r.alt"],
        "candidate_relation_ids_after_prune": ["r.next"],
        "answer_source": "GRAPH_FILTER",
        "filter_input_ids": ["m.c"],
        "filter_output_ids": ["m.c"],
        "final_answer_ids": ["m.c"],
        "evidence_pointer": "fixture://prefix-3",
    }
    record.update(overrides)
    return record


class MatchedPrefixReplayTests(unittest.TestCase):
    def test_identical_prefixes_replay_bit_identically(self):
        one = build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                         candidate_relation_ids_before_prune=["r.next", "r.alt"],
                         candidate_relation_ids_after_prune=["r.next"], kg_query="SELECT ?e3 ?e4 WHERE {}",
                         kg_rows=[{"?e3": "m.c", "?e4": "m.d"}])
        two = build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                         candidate_relation_ids_before_prune=["r.next", "r.alt"],
                         candidate_relation_ids_after_prune=["r.next"], kg_query="SELECT ?e3 ?e4 WHERE {}",
                         kg_rows=[{"?e3": "m.c", "?e4": "m.d"}])
        self.assertEqual(json.dumps(one, sort_keys=True), json.dumps(two, sort_keys=True))
        self.assertEqual(one["STOP_AT_3"]["prefix_hash"], one["CONTINUE_ONE_HOP"]["prefix_hash"])
        self.assertTrue(one["STOP_AT_3"]["prefix_replay_exact"])

    def test_per_hop_binding_and_stack_serialization_preserves_order_and_depth(self):
        record = fixture()
        record["dfs_stack"] = [
            {"path": [{"kind": "entity", "id": "m.root"}, {"kind": "relation", "id": "r.a", "direction": "OUTGOING"}], "depth": 1},
            {"path": [{"kind": "entity", "id": "m.root"}, {"kind": "relation", "id": "r.b", "direction": "INCOMING"}], "depth": 1},
        ]
        record["active_binding_rows"] = [{"?e1": "m.a", "?e2": "m.b", "?e3": "m.c"}]
        sealed = seal_prefix(record)
        self.assertTrue(sealed["prefix_replay_exact"])
        self.assertEqual([1, 1], [x["depth"] for x in sealed["dfs_stack"]])
        self.assertEqual("m.c", sealed["active_binding_rows"][0]["?e3"])

    def test_loop_is_logged_and_one_hop_cap_is_enforced(self):
        pair = build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                          candidate_relation_ids_before_prune=["r.next"], candidate_relation_ids_after_prune=["r.next"],
                          kg_query="Q", kg_rows=[{"?e3": "m.c", "?e4": "m.root"}])
        arm = pair["CONTINUE_ONE_HOP"]
        self.assertEqual(1, arm["new_edge_count"])
        self.assertEqual(["m.root"], arm["treatment"]["loop_target_ids"])
        self.assertEqual(4, arm["boundary_depth"] + arm["new_edge_count"])

    def test_intermediate_binding_join_and_merge_uses_ids(self):
        merged = binding_join(
            [{"?e1": "m.a", "?e2": "m.shared"}, {"?e1": "m.b", "?e2": "m.other"}],
            [{"?e2": "m.shared", "?e3": "m.answer"}],
        )
        self.assertEqual([{"?e1": "m.a", "?e2": "m.shared", "?e3": "m.answer"}], merged)
        self.assertEqual([], binding_join([{"?e2": "m.shared"}], [{"?e2": "m.other", "?e3": "m.x"}]))

    def test_empty_frontier_is_valid_zero_edge_continuation(self):
        pair = build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                          candidate_relation_ids_before_prune=["r.next"], candidate_relation_ids_after_prune=["r.next"],
                          kg_query="Q", kg_rows=[])
        self.assertEqual(0, pair["CONTINUE_ONE_HOP"]["new_edge_count"])
        self.assertEqual([], pair["CONTINUE_ONE_HOP"]["treatment"]["new_binding_rows"])

    def test_failed_kg_query_is_not_misclassified_as_empty_frontier(self):
        with self.assertRaisesRegex(ValueError, "status=ok"):
            build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                       candidate_relation_ids_before_prune=["r.next"], candidate_relation_ids_after_prune=["r.next"],
                       kg_query="Q", kg_rows=[], kg_status="timeout")

    def test_more_than_one_new_hop_variable_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one hop"):
            build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                       candidate_relation_ids_before_prune=["r.next"], candidate_relation_ids_after_prune=["r.next"],
                       kg_query="Q", kg_rows=[{"?e3": "m.c", "?e4": "m.d", "?e5": "m.e"}])

    def test_unsupported_oracle_and_malformed_binding_state_are_rejected(self):
        self.assertFalse(is_complete_prefix(fixture(oracle_relation_id="gold.r")))
        self.assertFalse(is_complete_prefix(fixture(active_binding_rows=[{"?e1": "Hawaiian hibiscus"}])))
        self.assertFalse(is_complete_prefix(fixture(active_binding_rows=[{"?e1": None}])))

    def test_exact_replay_rejects_missing_settings_prompt_cache_or_current_path(self):
        self.assertFalse(is_complete_prefix(fixture(model_settings={"model": "m", "temperature": 0})))
        self.assertFalse(is_complete_prefix(fixture(decisions=[{"stage": "reasoning"}])))
        self.assertFalse(is_complete_prefix(fixture(kg_cache=[{"query_hash": "q", "status": "error", "result_ids": []}])))
        self.assertFalse(is_complete_prefix(fixture(method_events=[])))
        self.assertFalse(is_complete_prefix(fixture(sparql_history=[])))
        self.assertFalse(is_complete_prefix(fixture(current_path=[])))

    def test_missing_or_unselected_candidate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "post-prune"):
            build_pair(fixture(), selected_relation_id="r.missing", relation_direction="OUTGOING",
                       candidate_relation_ids_before_prune=["r.next"], candidate_relation_ids_after_prune=["r.next"],
                       kg_query="Q", kg_rows=[])
        with self.assertRaisesRegex(ValueError, "present before"):
            build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                       candidate_relation_ids_before_prune=["r.alt"], candidate_relation_ids_after_prune=["r.next"],
                       kg_query="Q", kg_rows=[])

    def test_labels_can_change_without_changing_id_prefix_hash(self):
        first = seal_prefix(fixture())
        second = seal_prefix(fixture())
        self.assertEqual(first["prefix_hash"], second["prefix_hash"])
        p1 = build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                        candidate_relation_ids_before_prune=["r.next"], candidate_relation_ids_after_prune=["r.next"],
                        kg_query="Q", kg_rows=[{"?e3": "m.c", "?e4": "m.d"}], labels={"m.d": "Hawaiian hibiscus"})
        p2 = build_pair(fixture(), selected_relation_id="r.next", relation_direction="OUTGOING",
                        candidate_relation_ids_before_prune=["r.next"], candidate_relation_ids_after_prune=["r.next"],
                        kg_query="Q", kg_rows=[{"?e3": "m.c", "?e4": "m.d"}], labels={"m.d": "yellow hibiscus"})
        self.assertEqual(p1["CONTINUE_ONE_HOP"]["prefix_hash"], p2["CONTINUE_ONE_HOP"]["prefix_hash"])
        self.assertEqual(p1["CONTINUE_ONE_HOP"]["treatment"]["KG_result_ids"], p2["CONTINUE_ONE_HOP"]["treatment"]["KG_result_ids"])

    def test_answer_source_keeps_direct_and_graph_paths_distinct(self):
        direct = seal_prefix(fixture(answer_source="DIRECT_LLM", filter_input_ids=None, filter_output_ids=None,
                                     final_answer_ids=["m.outside_graph"]))
        filtered = seal_prefix(fixture(answer_source="GRAPH_FILTER", filter_input_ids=["m.b"],
                                       filter_output_ids=["m.b"], final_answer_ids=["m.b"]))
        self.assertTrue(direct["prefix_replay_exact"])
        self.assertTrue(filtered["prefix_replay_exact"])
        self.assertNotEqual(direct["final_answer_ids"], filtered["final_answer_ids"])

    def test_non_forward_short_path_is_not_extended(self):
        with self.assertRaisesRegex(ValueError, "ineligible"):
            build_pair(fixture(boundary_action="Finish"), selected_relation_id="r.next", relation_direction="OUTGOING",
                       candidate_relation_ids_before_prune=["r.next"], candidate_relation_ids_after_prune=["r.next"],
                       kg_query="Q", kg_rows=[])

    def test_resume_is_idempotent_and_refuses_partial_or_conflicting_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            row = {"record_id": "r1", "value": 1}
            self.assertEqual(1, resume_jsonl(str(path), [row]))
            self.assertEqual(0, resume_jsonl(str(path), [row]))
            self.assertEqual(1, resume_jsonl(str(path), [{"record_id": "r2", "value": 2}]))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                resume_jsonl(str(path), [{"record_id": "r1", "value": 9}])
            path.write_bytes(path.read_bytes() + b'{"record_id":"partial"')
            with self.assertRaisesRegex(ValueError, "partial"):
                resume_jsonl(str(path), [])

    def test_binding_query_sidecar_preserves_where_and_modifiers(self):
        query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?targetEntity WHERE { ns:m.a ns:r.one ?e1 . ?e1 ns:r.two ?targetEntity . } LIMIT 1000"
        probe = binding_projection_query(query, ["?e1", "?targetEntity"])
        self.assertIn("SELECT DISTINCT ?e1 ?targetEntity WHERE", probe)
        self.assertEqual(query[query.index("WHERE"):], probe[probe.index("WHERE"):])
        self.assertEqual("PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?targetEntity WHERE { ns:m.a ns:r.one ?e1 . ?e1 ns:r.two ?targetEntity . } LIMIT 1000", query)

    def test_sidecar_executor_normalizes_freebase_ids_and_rejects_literals(self):
        query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?targetEntity WHERE { ns:m.a ns:r.one ?targetEntity . } LIMIT 1000"
        seen = []
        def fake_executor(text):
            seen.append(text)
            return {"status": "ok", "rows": [{"targetEntity": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.answer"}}]}
        result = fetch_binding_projection(query, ["?targetEntity"], fake_executor)
        self.assertEqual([{"?targetEntity": "m.answer"}], result["rows"])
        self.assertEqual(result["query"], seen[0])
        with self.assertRaisesRegex(ValueError, "unsupported or missing RDF binding"):
            fetch_binding_projection(query, ["?targetEntity"], lambda _: {"status": "ok", "rows": [{"targetEntity": {"type": "literal", "value": "label"}}]})

    def test_binding_projection_attaches_complete_id_rows_to_prefix(self):
        snapshot = fixture(frontier_ids=["m.c"])
        projection_query = "SELECT DISTINCT ?first ?second ?endpoint WHERE {}"
        projection = {
            "status": "ok", "query": projection_query,
            "query_hash": hashlib.sha256(projection_query.encode("utf-8")).hexdigest(), "limit": 100,
            "rows": [{"?first": "m.a", "?second": "m.b", "?endpoint": "m.c"}],
        }
        result = attach_binding_projection(snapshot, projection, ["?first", "?second", "?endpoint"])
        self.assertEqual([{"?e1": "m.a", "?e2": "m.b", "?e3": "m.c"}], result["active_binding_rows"])
        self.assertEqual(["m.a", "m.b", "m.c"], result["active_binding_ids"])
        self.assertEqual(projection_query, result["binding_projection_query"])
        self.assertEqual(hashlib.sha256(projection_query.encode("utf-8")).hexdigest(),
                         result["binding_projection_query_hash"])
        self.assertTrue(result["prefix_replay_exact"])

    def test_binding_projection_rejects_truncation_and_frontier_mismatch(self):
        snapshot = fixture()
        rows = [{"?a": "m.a", "?b": "m.b", "?c": f"m.{i}"} for i in range(3)]
        with self.assertRaisesRegex(ValueError, "reached LIMIT"):
            attach_binding_projection(snapshot, {"status": "ok", "query_hash": "h", "limit": 3, "rows": rows},
                                      ["?a", "?b", "?c"])
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            attach_binding_projection(snapshot, {"status": "ok", "query_hash": "h", "limit": 100,
                                                 "rows": [{"?a": "m.a", "?b": "m.b", "?c": "m.other"}]},
                                      ["?a", "?b", "?c"])

    def test_cor_method_capture_preserves_returns_and_restores_instance(self):
        class Relation:
            def __init__(self, relation_id, left=True):
                self.id, self.left = relation_id, left
        class FakeCoR:
            def relation_search(self, topic_entity, relation_chain, sparql_history):
                sparql_history.extend([
                    {"search_type": "relation_search_head", "sparql": "QH", "status": "ok", "result_count": 1},
                    {"search_type": "relation_search_tail", "sparql": "QT", "status": "ok", "result_count": 0},
                ])
                return ["r.next"], []
            def relation_prune(self, question, topic_entity, relation_chain, head_relations, tail_relations, prompt_history):
                return [Relation("r.next")]
            def entity_search(self, topic_entity, relation_chain, sparql_history):
                sparql_history.append({"search_type": "entity_search", "sparql": "QE", "status": "ok", "result_count": 1})
                return ["m.answer"], ["Answer label"], []
            def reasoning(self, question, topic_entity, relation_chain, target_entity_candidates, prompt_history):
                return "Forward", "saved response", []
            def validate(self, question, topic_entity, relation_chain, target_entity_names, prompt_history):
                return ["Answer label"]
            def generate_directly(self, question, prompt_history):
                return "Direct answer"

        agent = FakeCoR()
        original_func = FakeCoR.relation_search
        sparql, events = [], []
        with capture_cor_method_outputs(agent, events):
            self.assertEqual((["r.next"], []), agent.relation_search(None, [], sparql))
            selected = agent.relation_prune("q", None, [], ["r.next"], [], [])
            self.assertEqual("r.next", selected[0].id)
            self.assertEqual((["m.answer"], ["Answer label"], []), agent.entity_search(None, [], sparql))
            self.assertEqual(("Forward", "saved response", []), agent.reasoning("q", None, [], ["Answer label"], []))
            self.assertEqual(["Answer label"], agent.validate("q", None, [], ["Answer label"], []))
            self.assertEqual("Direct answer", agent.generate_directly("q", []))
        self.assertNotIn("relation_search", agent.__dict__)
        self.assertIs(original_func, agent.relation_search.__func__)
        rel_search = events[0]
        self.assertEqual(["r.next"], rel_search["head_relation_ids"])
        self.assertEqual(2, len(rel_search["kg_cache"]))
        self.assertEqual(hashlib.sha256(b"QH").hexdigest(), rel_search["kg_cache"][0]["query_hash"])
        self.assertEqual("Direct answer", events[-1]["direct_llm_answer"])

    def test_cor_method_capture_restores_instance_after_exception(self):
        class RaisingAgent:
            def entity_search(self, topic_entity, relation_chain, sparql_history):
                raise RuntimeError("fixture method failure")

        agent = RaisingAgent()
        original = RaisingAgent.entity_search
        with self.assertRaisesRegex(RuntimeError, "fixture method failure"):
            with capture_cor_method_outputs(agent, []):
                agent.entity_search(None, [], [])
        self.assertNotIn("entity_search", agent.__dict__)
        self.assertIs(original, agent.entity_search.__func__)

    def test_protected_production_sources_remain_unchanged(self):
        manifest = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            content = (ROOT / item["path"]).read_bytes()
            self.assertEqual(item["sha256_at_iteration_start"], hashlib.sha256(content).hexdigest(), item["path"])

    def test_live_trace_hook_captures_stack_and_marks_missing_bindings(self):
        source = ROOT / "chain_of_relations/methods/cor/agent.py"
        lines = source.read_text(encoding="utf-8").splitlines()
        line_number = next(i for i, line in enumerate(lines, 1) if line.strip() == "if current_depth >= self.depth:")
        Entity = type("Entity", (), {})
        Relation = type("Relation", (), {})
        root = Entity(); root.id = "m.root"; root.name = "Root"
        rel = Relation(); rel.id = "r.one"; rel.left = True
        state = SimpleNamespace(path=[root, rel], depth=3)
        pending = SimpleNamespace(path=[root, rel], depth=2)
        frame = SimpleNamespace(
            f_code=SimpleNamespace(co_name="answer", co_filename=str(source)),
            f_lineno=line_number,
            f_locals={"action": "Forward", "current_depth": 3, "self": SimpleNamespace(depth=3),
                      "current_state": state, "stack": SimpleNamespace(items=[pending]), "entity_ids": ["m.answer"]},
        )
        capture = BoundaryCapture("q-live", 1, 3, "test://live", [], {"model": "fixture"})
        capture.trace(frame, "line", None)
        self.assertEqual(1, len(capture.snapshots))
        snap = capture.snapshots[0]
        self.assertEqual("m.answer", snap["frontier_ids"][0])
        self.assertEqual("r.one", snap["current_path"][1]["id"])
        self.assertEqual({"model": "fixture"}, snap["model_settings"])
        self.assertEqual(["m.answer", "m.root"], snap["visited_node_ids"])
        self.assertIsNone(snap["active_binding_rows"])
        self.assertFalse(snap["prefix_replay_exact"])

    def test_trace_wrapper_restores_hook_after_success_and_exception(self):
        from replay import run_with_boundary_capture
        import sys

        initial = sys.gettrace()
        result, snapshots = run_with_boundary_capture(lambda: 7, question_id="q", repeat=1, depth=3,
                                                       evidence_pointer="test://trace")
        self.assertEqual(7, result)
        self.assertEqual([], snapshots)
        self.assertIs(initial, sys.gettrace())
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            run_with_boundary_capture(lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")),
                                      question_id="q", repeat=1, depth=3, evidence_pointer="test://trace")
        self.assertIs(initial, sys.gettrace())


if __name__ == "__main__":
    unittest.main()
