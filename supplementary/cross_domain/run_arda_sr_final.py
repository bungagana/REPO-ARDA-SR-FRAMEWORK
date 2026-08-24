import argparse
import json
import re
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import GEMINI_API_KEY  # noqa: E402
# Import order matters: utils.kb_builder (torch/faiss/sentence-transformers)
# MUST be imported before utils.llm_client (google-generativeai/grpc) -- the
# reverse order segfaults on Windows (native-library load-order conflict,
# found 2026-08-18).
from utils.kb_builder import KnowledgeBase  # noqa: E402
from utils.llm_client import GeminiClient  # noqa: E402
from arda_sr.pipeline import ARDASRPipeline  # noqa: E402
from arda_sr.aqr import AQR, MODES  # noqa: E402
from arda_sr.retrieval import HybridRetriever  # noqa: E402
from arda_sr.dda import DDA  # noqa: E402
from config import HYBRID_ALPHA  # noqa: E402

OUT_PATH = THIS_DIR / "results" / "arda_sr_final_results.json"

# ── Fix 1: AQR routing ──────────────────────────────────────────────────
FIXED_AQR_PROMPT = """\
You are an AQR (Adaptive Query Router) for a government transmigration QA system.

Query: "{query}"

Classify this query into exactly one dominant response mode, then assign confidence scores.

Mode definitions and indicators:
- m1 (Direct Knowledge): query asks about a general concept, definition, or policy goal that does NOT require looking up a specific document (e.g. "what is X", "what does Y mean"). No named region/area/number is referenced. >>> Does NOT apply to domain-specific technical acronyms, classification codes, or defined regulatory terms (e.g. "K-1/K-2/K-3", "IPKT", "IDM", "RTSP", "RKT", "HPL", "SKP", or similar short capitalized codes/acronyms specific to Indonesian transmigration regulation) -- these have precise definitions fixed by regulation/corpus documents that general knowledge cannot reliably reconstruct, so a question asking what one of these terms/codes means must route to m2, EVEN IF no named region/area/number is referenced. <<<
- m2 (Factual Retrieval): query asks for a specific fact, figure, or attribute about a named area/region/regulation (e.g. area size, population, IPKT score, status) that requires looking up the corpus. >>> Also use m2 for "what does/is [acronym/code]" questions about domain-specific technical terms as described in the m1 exclusion above. <<<
- m3 (Hybrid/Ambiguous): query asks for information that is NOT anchored to a specific named area/document (no specific region name, no specific figure/number referenced) and instead asks about general patterns, comparisons, "typical"/"average"/"how does X compare" judgments, or impacts that would require synthesizing across multiple documents or making an interpretive judgment call rather than looking up one clear fact. Also use m3 when the query could plausibly be answered by more than one mode with similar likelihood.
- m4 (Policy Scenario): query explicitly presents policy options/interventions/strategies and asks for a recommendation, comparison, or decision among them (look for words like "opsi", "strategi", "intervensi", "sebaiknya", "prioritas", "rekomendasi").

Only spread probability mass roughly evenly across modes when the query is GENUINELY ambiguous per the m3 indicators above. If the query clearly matches one mode's indicators, you MUST assign it a DOMINANT probability of 0.80-0.92, and split the remaining mass thinly across the other three (each should be small, e.g. 0.02-0.08). Being decisive when the signal is clear is CORRECT behavior, not overconfidence -- do not hedge or spread mass out of caution when indicators clearly point to one mode.

Feature scores (floats 0.0-1.0): entity_signal, domain_specificity, temporal_ref, multihop_signal, context_dep.
Probabilities must sum to 1.0.

Respond with ONLY valid JSON (no markdown, no explanation). Example for a CLEARLY policy-scenario query (decisive, not spread):
{{"features":{{"entity_signal":0.8,"domain_specificity":0.9,"temporal_ref":0.1,"multihop_signal":0.6,"context_dep":0.3}},"mode_probs":{{"m1":0.02,"m2":0.05,"m3":0.05,"m4":0.88}},"reasoning":"brief reason"}}

Now classify:
Query: "{query}"
JSON:"""


class FixedAQR(AQR):
    def classify(self, query: str) -> dict:
        prompt = FIXED_AQR_PROMPT.format(query=query)
        try:
            result = self.client.generate_json(prompt)
        except Exception:
            return self._default_result()
        features = result.get("features", {})
        mode_probs = result.get("mode_probs", {"m1": 0.25, "m2": 0.25, "m3": 0.25, "m4": 0.25})
        total = sum(mode_probs.get(m, 0.0) for m in MODES)
        if total < 1e-9:
            total = 1.0
        mode_probs = {m: mode_probs.get(m, 0.0) / total for m in MODES}
        entropy = self._entropy(mode_probs)
        hybrid = entropy > self.tau_h
        dominant = max(mode_probs, key=mode_probs.get)
        mode = "m3" if hybrid else dominant
        return {"mode": mode, "mode_probs": mode_probs, "entropy": round(entropy, 4),
                "features": features, "hybrid_path": hybrid, "reasoning": result.get("reasoning", "")}


# ── Fixes 2-4: retrieval (no year filter + entity boost + 3.docx relabel) ─
# Fix 10 (found 2026-08-19, re-auditing why "Kerang" queries kept failing
# retrieval across the whole session despite fixes 3/5/9): the original
# pattern only matched the ALL-CAPS header style used by most profile
# files ("KAWASAN TRANSMIGRASI MUTIARA – MUNA, SULTRA"). kerang1.txt (and
# several others) instead open with a Title-Case variant ("Kerang – Paser,
# Kalimantan Timur\nKawasan Transmigrasi Kerang memiliki luas..."), which
# the case-sensitive "KAWASAN TRANSMIGRASI" trigger never matched -- so
# "KERANG" never made it into entity_names at all, and no boost/fix 5/fix 9
# mechanism downstream could ever engage for it, regardless of how much
# those mechanisms were tuned. Naively adding re.IGNORECASE to the whole
# pattern was tried and rejected: it also matches "kawasan transmigrasi X"
# giving mid-sentence trigger phrase (frequent in regulation PDFs/prose
# docs), and un-anchored from case the greedy name-capture group swallows
# whole clauses ("SELAUT MEMILIKI AKSES INTERNET..."). Fix: keep the NAME
# capture case-sensitive (so it still stops at the first lowercase-led
# word, bounding it to real Title-Case name tokens) and only add
# "Kawasan Transmigrasi" as a second, explicit trigger-phrase alternative
# -- covers the one real casing variant seen in this corpus without
# opening up full case-insensitivity. Verified: single-kawasan file count
# (_build_file_entities' size==1 cohort) went 12 -> 41, correctly picking
# up kerang1.txt -> KERANG and ~28 other previously-invisible profile
# files, while known multi-kawasan compendium files (3.docx,
# EBOOK_SIPUKAT_Profil_Kawasan.pdf, pengantar_kriteria_kawasan.docx, ...)
# still correctly resolve to size > 1 and stay excluded from the boost.
HEADER_RE = re.compile(
    r"(?:KAWASAN TRANSMIGRASI|Kawasan Transmigrasi) "
    r"([A-Z][A-Za-z]*(?:[ –—-][A-Z][A-Za-z]*){0,3})"
)
_JUNK_NAMES = {
    "DAN", "DI", "KE", "YANG", "DENGAN", "PADA", "ATAU", "SERTA",
    # fix 10 fallout: the Title-Case trigger variant also fires on generic
    # words/short truncated fragments that happen to follow "Kawasan
    # Transmigrasi" in incidental prose (regulation PDFs, cross-references)
    # rather than a real kawasan name. Each of these, left in, would match
    # as a substring inside nearly every file's text (e.g. "TAHUN" appears
    # in "IPKT Tahun 2023" everywhere) and pollute _build_file_entities'
    # per-file entity sets from size==1 to size>1, silently EXCLUDING the
    # genuine single-kawasan files (muna.txt, kerang1.txt, ...) that fix 9
    # depends on -- caught by re-testing qa_0206/qa_0216 immediately after
    # landing fix 10 and finding muna.txt had regressed out of file_entities
    # entirely. "LAMUNTI"/"KETUNGAU H"/"MUARA TA"/"SALI" are truncated
    # duplicates of a longer valid name (LAMUNTI DADAHUP, KETUNGAU HULU,
    # MUARA TAKUNG-KAMANG BARU, SALIM BATU) picked up from a differently
    # line-wrapped occurrence of the same header elsewhere in the corpus;
    # dropping the short fragment leaves the full name intact.
    "TAHUN", "PASAL", "DELINIASI", "RPJMN", "PRIORITAS NASIONAL",
    "LAMUNTI", "KETUNGAU H", "MUARA TA", "SALI",
}
ENTITY_BOOST = 0.5
LIST_HEADER_RE = re.compile(r"\d+\)\s*[A-Z][\w –—-]{2,40}?\s*[–—-]")


def _build_entity_names(kb: KnowledgeBase) -> set:
    names = set()
    for c in kb.chunks:
        for m in HEADER_RE.finditer(c.get("text", "")):  # fix 10: all matches, not just first
            name = m.group(1).strip().upper()
            if name in _JUNK_NAMES or len(name) < 4:
                continue
            names.add(name)
    return names


def _build_file_entities(kb: KnowledgeBase, entity_names: set) -> dict:
    """Map filename -> the single entity (kawasan) name that file is about,
    for files that are genuinely about exactly one kawasan (fix 5, found
    2026-08-19 while re-judging the 35 queries de-ambiguated in
    data/qa_dataset.json).

    Why "exactly one": a single-kawasan source file's "KAWASAN TRANSMIGRASI
    X" header -- the only text fix 3's entity boost can match on -- appears
    in chunk 0 ONLY, so continuation chunks (chunk 1, 2, ...) holding the
    actual numeric facts a query asks about don't repeat the name and lose
    the top-5 slot to wrong-kawasan chunks, even when chunk 0 of the RIGHT
    file already ranked #1 (qa_0206/0209/0212 all showed this: right doc
    found, right chunk missing, DDA fell back to a hallucinated "direct"
    draft or an honest-but-wrong refusal).

    A first version of this fix collected ALL entity names found anywhere
    in a file's chunks with no cap. That over-fired on multi-kawasan
    reference/compendium files that name many kawasan in passing as
    examples (EBOOK_SIPUKAT_Profil_Kawasan.pdf mentions 24 kawasan across
    its ~200 chunks, Selaut.docx mentions 2) -- every one of THEIR chunks
    then got boosted for ANY query naming ANY of those kawasan, flooding
    top-5 with irrelevant general-reference chunks and pushing the actual
    per-kawasan profile file out entirely (regression observed on qa_0209:
    mas perkasa.docx dropped out of top-5 completely, replaced by 5x
    EBOOK_SIPUKAT_Profil_Kawasan.pdf chunks). Capping to files whose
    aggregate entity set has size == 1 fixes this: every genuine
    single-kawasan profile file (muna.txt, mambi.txt, daduhub.txt, ... --
    verified 12/12 of the files behind the 35 de-ambiguated queries) maps
    to exactly one name, while compendium/reference files (size 2-38) are
    excluded from the file-wide boost and only get it via fix 3's original
    per-chunk-text check.

    Fix 10 addendum: this used to detect a mention via bare substring
    (`n in text.upper()`) against the full entity_names list. That was safe
    while every entity name came from the ALL-CAPS-only header pattern
    (multi-word names like "MUTIARA", "GERBANG MAS PERKASA" essentially
    never collide with ordinary prose). Once fix 10 widened entity
    extraction to also catch Title-Case headers, short single-word names
    that are also ordinary Indonesian words entered the set -- "KERANG"
    (kawasan name) is also the word for "shellfish" -- and the bare
    substring check started matching it inside unrelated PDFs (a cacao
    cultivation guide, a regional-cooperation regulation, ...) that happen
    to use the word "kerang" once, wrongly making those files claim
    KERANG as their entity and breaking the single-file guarantee (fix 9)
    for the real kerang1.txt. Fix: require the SAME "Kawasan Transmigrasi
    X" context HEADER_RE already demands for extraction in the first
    place, rather than a bare word-anywhere check -- a mention only counts
    if it actually reads as a kawasan reference, not just contains the
    word."""
    file_entity_sets: dict = {}
    for c in kb.chunks:
        found = {m.group(1).strip().upper() for m in HEADER_RE.finditer(c.get("text", ""))}
        found &= entity_names  # still respect the same junk/length filtering as entity_names
        if found:
            file_entity_sets.setdefault(c.get("filename", ""), set()).update(found)
    return {f: next(iter(ents)) for f, ents in file_entity_sets.items() if len(ents) == 1}


def _build_3docx_label_map(kb: KnowledgeBase) -> dict:
    docx3 = sorted((c for c in kb.chunks if c.get("filename") == "3.docx"),
                   key=lambda c: c.get("chunk_index", 0))
    label_map = {}
    current_header = None
    for c in docx3:
        text = c.get("text", "")
        m = LIST_HEADER_RE.search(text[:60])
        if m:
            header_line = text[m.start():text.find(".", m.start()) if "." in text[m.start():] else len(text)]
            current_header = header_line.strip()
        if current_header and not text.lstrip().startswith(tuple(str(d) + ")" for d in range(10))):
            label_map[c["chunk_id"]] = f"[{current_header}]\n{text}"
        else:
            label_map[c["chunk_id"]] = text
    return label_map


class FinalRetriever(HybridRetriever):
    """HybridRetriever with fixes 2-5 applied: no year hard-filter,
    named-entity boost (all entities mentioned, not just the longest),
    3.docx chunk relabeling, and whole-file entity association so every
    chunk of the correct kawasan's file gets boosted, not just its header
    chunk."""

    def __init__(self, kb: KnowledgeBase):
        super().__init__(kb)
        self.entity_names = _build_entity_names(kb)
        self.label_map = _build_3docx_label_map(kb)
        self.file_entities = _build_file_entities(kb, self.entity_names)  # fix 5

    @staticmethod
    def extract_metadata_from_query(query: str) -> dict:
        filt = HybridRetriever.extract_metadata_from_query(query)
        filt.pop("year", None)  # fix 2
        return filt

    def _query_entities(self, query: str):
        qu = query.upper()
        candidates = [n for n in self.entity_names if n in qu]
        return [c for c in candidates if not any(c != o and c in o for o in candidates)]

    def retrieve(self, query: str, k: int = 5, metadata_filter=None):
        chunks = self.kb.chunks
        entities = self._query_entities(query)  # fix 3 (moved earlier for fix 11, see below)
        candidate_ids = list(range(len(chunks)))
        if metadata_filter:
            candidate_ids = [i for i, c in enumerate(chunks) if self._matches(c, metadata_filter)] or candidate_ids
            # Fix 11: metadata_filter (e.g. {"commodities": "kelapa sawit"},
            # extracted from the query's own wording) hard-EXCLUDES any
            # chunk whose stored commodities list doesn't contain that
            # word -- found 2026-08-19 chasing why qa_0216 (Kerang palm-oil
            # production) still failed even after fix 9/10 fixed retrieval
            # for every other Kerang query: kerang1.txt chunk 2 literally
            # contains "sawit 11 ton/ha (2.039.959 ton/tahun)" -- the exact
            # reference answer -- but that chunk's commodities tag is only
            # ["jagung", "sapi"] (an incomplete/wrong tag from however the
            # KB was built), so the hard filter dropped all of kerang1.txt
            # from candidate_ids before entity boost or fix 9's guarantee
            # ever got a chance to run. Same root shape as fix 2 (which
            # already had to remove an unreliable year hard-filter) -- a
            # chunk-level metadata tag that doesn't reliably reflect the
            # chunk's actual content shouldn't be allowed to silently veto
            # a chunk before ranking. Rather than drop the commodity filter
            # entirely (it likely still helps in the common case with no
            # confident entity match), only override it when there IS a
            # single confidently-matched kawasan file: re-admit that file's
            # own chunk ids even if the metadata filter had excluded them,
            # since a confirmed single-kawasan match is strictly more
            # reliable evidence of relevance than a possibly-mistagged
            # commodity field.
            single_match = {f for f, e in self.file_entities.items() if e in entities} if entities else set()
            if len(single_match) == 1:
                target_file = next(iter(single_match))
                own_ids = {i for i, c in enumerate(chunks) if c.get("filename") == target_file}
                candidate_ids = list(set(candidate_ids) | own_ids)

        q_vec = self.kb.embed_query(query)
        n_faiss = min(len(chunks), max(k * 5, 50))
        faiss_scores_arr, top_indices = self.kb._faiss.search(q_vec, n_faiss)
        faiss_scores = {}
        for rank, idx in enumerate(top_indices[0]):
            if idx >= 0:
                faiss_scores[idx] = float(faiss_scores_arr[0][rank])

        bm25_all = self.kb.bm25_scores(query)
        max_bm25 = bm25_all.max() or 1.0
        bm25_norm = bm25_all / max_bm25

        scored = []
        for i in candidate_ids:
            dense = faiss_scores.get(i, 0.0)
            bm25 = float(bm25_norm[i])
            hybrid = HYBRID_ALPHA * dense + (1 - HYBRID_ALPHA) * bm25
            if entities:
                chunk = chunks[i]
                haystack = (chunk.get("filename", "") + " " + chunk.get("text", "")[:150]).upper()
                own_hit = any(e in haystack for e in entities)
                file_entity = self.file_entities.get(chunk.get("filename", ""))  # fix 5
                file_hit = file_entity is not None and file_entity in entities
                if own_hit or file_hit:
                    hybrid += ENTITY_BOOST
            scored.append((i, hybrid))

        scored.sort(key=lambda x: -x[1])

        # Fix 9: a flat per-chunk boost still makes each chunk of the
        # matched file compete individually against the whole corpus for a
        # k=5 slot -- found 2026-08-19 re-auditing why qa_0206 (Kawasan
        # Mutiara klinik count) still failed post-fix-5: chunk 0 (header)
        # scored 1.21 and took the #1 slot, but chunk 2 (the one actually
        # containing "31 klinik") only reached 0.556 after the +0.5 boost,
        # missing the #5 cutoff of 0.594 by a mere 0.038 -- its BASE score
        # was low because that chunk's text is a dense mid-paragraph list
        # of health-facility figures that doesn't itself repeat "Mutiara"
        # or read as a clear semantic match for a "how many klinik"
        # question. Rather than keep raising ENTITY_BOOST as a blunt lever
        # (which risks flooding results the way the uncapped fix-5 version
        # did), when the query resolves to EXACTLY ONE confidently-matched
        # single-kawasan file, guarantee that file's own chunks fill as
        # many of the k slots as it has (small files: 3-10 chunks per the
        # corpus, so this rarely crowds out everything else) -- since a
        # single unambiguous file match means every one of its chunks is
        # relevant by construction, there's no reason to let organic score
        # gate which of ITS OWN chunks the caller gets to see.
        matched_files = {f for f, e in self.file_entities.items() if e in entities} if entities else set()
        if len(matched_files) == 1:
            target_file = next(iter(matched_files))
            score_by_id = dict(scored)
            own_ids = [i for i in candidate_ids if chunks[i].get("filename") == target_file]
            if 0 < len(own_ids) <= k:
                own_sorted = sorted(own_ids, key=lambda i: -score_by_id[i])
                rest_sorted = [(i, s) for i, s in scored if i not in own_ids]
                scored = [(i, score_by_id[i]) for i in own_sorted] + rest_sorted

        results = []
        for idx, score in scored[:k]:
            c = dict(chunks[idx])
            if c.get("filename") == "3.docx" and c.get("chunk_id") in self.label_map:  # fix 4
                c["text"] = self.label_map[c["chunk_id"]]
            c["hybrid_score"] = round(score, 4)
            results.append(c)
        return results


# ── Fix 6: DDA prompt hardening (retrieval-extraction miss + parametric
#    over-confidence) ────────────────────────────────────────────────────
# Found 2026-08-19 investigating qa_0245 (Sumalata -> ibu kota kecamatan
# distance) and qa_0243 (village "Salubanua" sub-district). Root cause is
# NOT retrieval in either case -- in qa_0245 the correct chunk (sumlatala.txt
# chunk 1, containing "...ke ibu kota kecamatan 4,6 kilometer") was already
# in the top-5 evidence, but the retrieval-grounded draft still answered
# "informasi tidak mencukupi" (failed to extract the right one of three
# co-located distance figures in the same passage -- kecamatan/kabupaten/
# provinsi all listed together). DDA's arbitrator then had to choose between
# an honest-but-empty retrieval draft and a confident parametric draft that
# guessed a specific wrong number ("1 kilometer") -- and picked the
# confident wrong guess (score_dir 0.35 > score_ret 0.25, margin 0.05).
# qa_0243 is the same failure shape one step further: retrieval found
# nothing relevant at all (query never named a kawasan), so the parametric
# draft filled in with a real Indonesian village that happens to share a
# name, stated as fact.
#
# This is a generation-quality problem, not a lookup problem, so the fix is
# to change what the two drafts are ASKED to do, not to add keyword/regex
# patching on their output:
#   - RETRIEVAL_PROMPT: explicitly instruct the model to scan every
#     evidence passage for ALL numbers/labels before concluding a fact is
#     absent -- co-located figures for a different attribute (e.g. distance
#     to kabupaten sitting next to distance to kecamatan) are the single
#     most common extraction miss observed, so call that pattern out by
#     name as something to check for, not just "read carefully."
#   - DIRECT_PROMPT: explicitly instruct the model that any SPECIFIC,
#     checkable claim about this domain (a distance, a headcount, which
#     sub-district a named village sits in, ...) that it is not certain of
#     from training data must be flagged as uncertain rather than stated as
#     fact -- closing the exact gap that let a real-but-unrelated
#     "Salubanua" surface as a confident answer.
# Both changes operate on the model's own reasoning about what it knows and
# doesn't -- no string-matching over the output is added.
HARDENED_DIRECT_PROMPT = """\
You are a knowledgeable assistant for the Indonesian transmigration domain.
Answer the following question using your general knowledge. Be concise.

IMPORTANT: this question may be about a SPECIFIC named village, kawasan
transmigrasi, or dataset figure that exists only in a private document
corpus you have not been shown. If the question asks for a specific,
checkable fact (an exact distance, a headcount, a percentage, which
sub-district/kecamatan a named place belongs to, etc.) and you are not
genuinely certain your answer refers to the SAME place/document the
question means -- rather than a same-named or similar-sounding place you
happen to know from general knowledge -- say so explicitly (e.g. "I don't
have verified information on this specific location/figure") instead of
stating a specific number or place as if confirmed. A plausible-sounding
specific answer that turns out to reference the wrong real-world place is
worse than an honest "I'm not certain."

EXCEPTION -- this caution does NOT apply when the question already states
the specific numeric values you need as part of the question itself (e.g.
"If total production is 32,685.17 tons and the population is 13,677, what
is the per-capita rate?"). That is self-contained arithmetic/logic on
numbers the question handed you, not an unverified external fact -- solve
it directly and confidently using exactly the given values, showing the
calculation. Only hedge on facts you'd have to supply FROM your own
knowledge, never on a computation over values already stated in the
question.

Question: {query}

Answer:"""

HARDENED_RETRIEVAL_PROMPT = """\
You are a precise assistant for the Indonesian transmigration domain.
Answer the question strictly based on the provided evidence. Do not add information not in the evidence.

Before answering, scan EVERY evidence passage for every number, name, or
label that could be relevant -- profile passages routinely list several
co-located figures for related-but-different attributes in the same
sentence (e.g. distance to kecamatan, kabupaten, AND provinsi all listed
together; or counts for several facility types in one list). Match the
question's exact attribute/unit to the correct one among them before
concluding the evidence lacks it -- do not stop at the first number you see
if it isn't labeled for what the question actually asks.

If the question has multiple parts (e.g. it compares two named areas, or
asks for several linked facts), treat each part separately: answer every
part the evidence DOES support with its specific figures, and only flag as
missing the specific part(s) the evidence genuinely doesn't cover. Do not
refuse the whole question just because one sub-part is missing -- a partial
answer that clearly states what is and isn't covered is far more useful
than a blanket "insufficient evidence," and matches what a careful human
analyst would do with the same partial evidence.

Only if the evidence addresses NONE of the question's parts should you say
so briefly, with no partial answer to give.

Question: {query}

Evidence:
{evidence}

Answer:"""


class HardenedDDA(DDA):
    """DDA with fix 6 applied: hardened DIRECT/RETRIEVAL prompts (see
    module docstring fix 6) plus an evidence-truncation fix, to reduce
    parametric over-confidence and retrieval-extraction misses. Scoring
    (_score_utility/_weighted_utility) and the arbitration rule are
    untouched -- the fix targets what each draft is asked to do (and what
    text it's actually shown), not how they're judged or picked.

    The truncation fix matters more than it looks: tracing WHY qa_0209's
    retrieval draft said "not mentioned" despite mas perkasa.docx chunk 2
    being in evidence found that DDA._format_evidence() cuts every evidence
    chunk to `text[:600]` -- and chunk 2 is 820 chars, with its "F.
    Fasilitas Kesehatan / Di kawasan terdapat 4 unit puskesmas" line sitting
    right after the 600-char cutoff, so the model was never shown the
    answer at all. Checked corpus-wide: 1092 of 1175 chunks (93%) exceed
    600 chars, so this wasn't a one-off -- it was silently truncating the
    answer out of evidence on the majority of retrievals. 5 evidence items
    at up to ~820 chars each is ~4-5K chars total, trivial for the model's
    context window, so there is no real reason to truncate this
    aggressively; the fix raises the per-chunk cap to 2000 (comfortably
    above the corpus's observed max of 820, leaving headroom if a longer
    chunk ever appears in a rebuilt KB) rather than removing the cap
    outright, keeping a safety bound against a single pathological chunk
    blowing up the prompt."""

    EVIDENCE_CHAR_CAP = 2000

    def _generate_direct(self, query: str) -> str:
        prompt = HARDENED_DIRECT_PROMPT.format(query=query)
        return self.client.generate(prompt, max_tokens=512)

    def _generate_retrieval(self, query: str, evidence_text: str) -> str:
        prompt = HARDENED_RETRIEVAL_PROMPT.format(query=query, evidence=evidence_text)
        return self.client.generate(prompt, max_tokens=768)

    @staticmethod
    def _format_evidence(evidence):
        parts = []
        for i, e in enumerate(evidence, 1):
            text = e.get("text", "")[:HardenedDDA.EVIDENCE_CHAR_CAP]
            parts.append(f"[Evidence {i}] (source: {e.get('filename','?')})\n{text}")
        return "\n\n".join(parts)


def build_final_pipeline(kb: KnowledgeBase, client: GeminiClient) -> ARDASRPipeline:
    pipeline = ARDASRPipeline(kb, client=client)
    pipeline.aqr = FixedAQR(client)
    pipeline.retriever = FinalRetriever(kb)
    pipeline.dda = HardenedDDA(client=pipeline.dda.client, betas=pipeline.dda.betas)  # fix 6
    return pipeline


def load_existing() -> dict:
    if OUT_PATH.exists():
        try:
            recs = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            return {r["query_id"]: r for r in recs}
        except Exception:
            return {}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Limit to first N queries (default: all)")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set in .env", flush=True); sys.exit(1)

    qa_data = json.loads((ROOT_DIR / "data" / "qa_dataset.json").read_text(encoding="utf-8"))
    if args.n:
        qa_data = qa_data[:args.n]

    done = load_existing()
    print(f"Resuming: {len(done)} queries already done (of {len(qa_data)} total).", flush=True)

    print("Loading KB...", flush=True)
    kb = KnowledgeBase().load()
    client = GeminiClient()
    pipeline = build_final_pipeline(kb, client)
    print(f"Pipeline ready (all 4 fixes applied). Running on {len(qa_data)} queries...\n", flush=True)

    all_results = list(done.values())
    n_new = n_failed = 0
    t0 = time.time()

    for i, q in enumerate(qa_data, 1):
        qid = q["query_id"]
        if qid in done:
            continue
        try:
            r = pipeline.run(q["question"], reference_answer=q.get("reference_answer", ""))
        except Exception as exc:
            n_failed += 1
            print(f"  [FAIL] {qid}: {str(exc)[:150]}", flush=True)
            continue
        r["query_id"] = qid
        r["category"] = q["category"]
        all_results.append(r)
        done[qid] = r
        n_new += 1

        if i % 10 == 0 or i == len(qa_data):
            elapsed = time.time() - t0
            rate = n_new / max(elapsed, 1e-6)
            eta_s = (len(qa_data) - len(done)) / max(rate, 1e-6)
            print(f"  [{i}/{len(qa_data)}] new={n_new} failed={n_failed} "
                  f"({elapsed:.0f}s elapsed, {rate:.2f} q/s, ETA {eta_s/60:.1f}min)", flush=True)

        if n_new % args.checkpoint_every == 0:
            OUT_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ...checkpoint saved ({len(all_results)} total records)", flush=True)

    OUT_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. total={len(all_results)} new_this_run={n_new} failed_this_run={n_failed}", flush=True)
    print(f"Written -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
