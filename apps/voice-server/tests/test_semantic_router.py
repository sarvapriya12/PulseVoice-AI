import pytest
from services.rag.semantic_router import route_query, init_semantic_router, ROUTE_PROTOTYPES

def test_semantic_router_initialization():
    init_semantic_router()
    from services.rag.semantic_router import _ROUTER_INITIALIZED, _ROUTE_MATRICES
    assert _ROUTER_INITIALIZED is True
    assert len(_ROUTE_MATRICES) == len(ROUTE_PROTOTYPES)


def test_route_core_vitals():
    route, conf = route_query("what are your clinic hours on monday?")
    assert route == "core_vitals"
    assert conf > 0.50

    route_loc, conf_loc = route_query("where is the clinic located, what is the address?")
    assert route_loc == "core_vitals"
    assert conf_loc > 0.50


def test_route_chitchat():
    route, conf = route_query("hello good morning how are you?")
    assert route == "chitchat"
    assert conf > 0.50


def test_route_booking():
    route, conf = route_query("i need to book an appointment for next tuesday")
    assert route == "booking_tool"
    assert conf > 0.50


def test_route_rag_knowledge():
    route, conf = route_query("how should i prepare for my blood test fasting?")
    assert route == "rag_knowledge"
    assert conf > 0.40
