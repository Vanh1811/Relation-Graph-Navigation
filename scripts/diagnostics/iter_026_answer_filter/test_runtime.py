from __future__ import annotations

import unittest

from runtime import GatewayFreebaseBackend, GatewayLLMAPI, make_tgar_agent


class StubGateway:
    def __init__(self):
        self.model_calls = []
        self.queries = []

    def call_model_once(self, messages, *, max_completion_tokens, reasoning_effort):
        self.model_calls.append((messages, max_completion_tokens, reasoning_effort))
        return {
            "content": "OK",
            "usage": {"input_tokens": 8, "output_tokens": 1, "total_tokens": 9},
        }

    def select_once(self, query):
        self.queries.append(query)
        if "VALUES ?entity" in query:
            return {"bindings": [{
                "entity": {"value": "http://rdf.freebase.com/ns/m.alpha"},
                "targetEntity": {"value": "Alpha", "xml:lang": "en"},
            }]}
        if "m.alpha" in query:
            return {"bindings": [{"targetEntity": {"value": "Alpha", "xml:lang": "en"}}]}
        return {"bindings": []}


class RuntimeAdapterTests(unittest.TestCase):
    def test_llm_proxy_uses_one_gateway_request_and_omits_temperature(self):
        gateway = StubGateway()
        adapter = GatewayLLMAPI(gateway)
        text, usage = adapter.generate(
            "user text", temperature=0.1, max_tokens=2048, system_prompt="system text",
        )
        self.assertEqual(text, "OK")
        self.assertEqual(usage["total_tokens"], 9)
        messages, max_completion_tokens, effort = gateway.model_calls[0]
        self.assertEqual(messages, [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ])
        self.assertEqual(max_completion_tokens, 1024)
        self.assertEqual(effort, "low")
        self.assertEqual(len(gateway.model_calls), 1)

    def test_freebase_batch_labels_use_only_metered_gateway_queries(self):
        gateway = StubGateway()
        backend = GatewayFreebaseBackend(gateway)
        self.assertEqual(backend.id2entity_names(["m.alpha"]), {"m.alpha": "Alpha"})
        self.assertEqual(len(gateway.queries), 1)
        self.assertTrue(gateway.queries[0].lstrip().startswith("PREFIX"))

    def test_frozen_tgar_agent_constructs_with_safe_adapters_without_reading_credentials(self):
        import chain_of_relations.methods.cor.agent as cor_agent_module
        original_llm_api = cor_agent_module.LLMAPI
        gateway = StubGateway()
        agent = make_tgar_agent(gateway)
        self.assertIsInstance(agent.cor_agent.llm_api, GatewayLLMAPI)
        self.assertIsInstance(agent.cor_agent.kg_backend, GatewayFreebaseBackend)
        self.assertIs(cor_agent_module.LLMAPI, original_llm_api)
        self.assertEqual(gateway.model_calls, [])
        self.assertEqual(gateway.queries, [])


if __name__ == "__main__":
    unittest.main()
