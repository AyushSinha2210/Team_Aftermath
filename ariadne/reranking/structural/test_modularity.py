from ariadne.reranking.structural.modularity import detect_code_communities


def test_detect_code_communities_two_clusters():
	# Two disjoint tightly-coupled clusters: (auth module) and (billing module)
	graph = {
		# Cluster 1: Auth
		"auth:login": ["auth:verify", "auth:createSession"],
		"auth:verify": ["auth:login"],
		"auth:createSession": ["auth:login"],
		# Cluster 2: Billing
		"billing:charge": ["billing:invoice", "billing:gateway"],
		"billing:invoice": ["billing:charge"],
		"billing:gateway": ["billing:charge"],
	}

	communities = detect_code_communities(graph)

	assert len(communities) == 6
	# Auth nodes share the same community ID
	auth_comm = communities["auth:login"]
	assert communities["auth:verify"] == auth_comm
	assert communities["auth:createSession"] == auth_comm

	# Billing nodes share a different community ID
	bill_comm = communities["billing:charge"]
	assert communities["billing:invoice"] == bill_comm
	assert communities["billing:gateway"] == bill_comm

	assert auth_comm != bill_comm


def test_detect_code_communities_empty():
	assert detect_code_communities({}) == {}
