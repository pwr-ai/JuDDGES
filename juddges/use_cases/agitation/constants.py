"""
Constants for the agitation use case.
"""

# Search Constants
BATCH_SIZE = 100
MAX_OBJECTS = 10_000
QUERY_COLUMN = "query"

# Extraction Constants
LLM_BATCH_SIZE = 10
MAX_TEXT_LENGTH = 150000

# Search Queries
AGITATION_QUERIES = {
    "bm25_queries": [
        "art. 111 kodeksu wyborczego",
        "fałszywe informacje w materiałach wyborczych",
        "postępowanie wyborcze internetowe",
        "kampania wyborcza media społecznościowe",
        "pozew wyborczy fałszywe informacje",
        "proces wyborczy dezinformacja",
        "orzeczenie sądu kampania wyborcza online",
        "orzeczenie fałszywe informacje wybory",
        "orzeczenia dotyczące agitacji wyborczej",
        "orzeczenie sądu art. 111 Kodeks wyborczy",
    ],
    "vector_queries": [
        "Orzeczenia dotyczące fałszywych informacji wyborczych rozpowszechnianych online podczas kampanii",
        "Polskie orzeczenia sądowe w sporach dotyczących prawa wyborczego w mediach społecznościowych",
        "Wyroki dotyczące wprowadzających w błąd lub fałszywych treści kampanii",
        "Sprawy, w których kandydat zakwestionował treści wyborcze z powodu dezinformacji",
        "Środki prawne stosowane do przeciwdziałania dezinformacji online w wyborach",
        "Postępowania sądowe dotyczące cyfrowej propagandy wyborczej",
        "Reakcja sądownictwa na kampanie algorytmiczne lub działania VLOP",
        "Jak polskie sądy interpretują artykuł 111 w sprawach cyfrowych",
        "Spory dotyczące kampanii prowadzonych przez osoby trzecie bez zgody komitetu",
        "Praktyki sądowe dotyczące fałszywych materiałów wyborczych online",
    ],
}
