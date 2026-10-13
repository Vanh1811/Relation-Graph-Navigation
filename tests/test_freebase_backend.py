from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend


def test_freebase_relation_formatter_handles_prefixed_and_absolute_uris():
	backend = FreebaseBackend()

	assert backend.format_relation_node("film.film.country") == "ns:film.film.country"
	assert (
		backend.format_relation_node("http://www.w3.org/2000/01/rdf-schema#label")
		== "<http://www.w3.org/2000/01/rdf-schema#label>"
	)
