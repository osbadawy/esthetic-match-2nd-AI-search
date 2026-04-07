# #!/usr/bin/env python3
# from __future__ import annotations

# import os
# import pickle
# import time
# import difflib
# import re

# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# from llm_client import LocalLLMClient


# DEFAULT_EMBEDDING_MODEL = "sentence-transformers/static-similarity-mrl-multilingual-v1"


# # ---------------------------- helpers ----------------------------

# def _norm(x: str) -> str:
#     return " ".join(str(x or "").strip().lower().split())


# def _norm_type_value(x: str) -> str:
#     """
#     Normalize DB type to {surgical, non-surgical, ""}.
#     Handles many variants: Non surgical, non-surg, non-surgical, etc.
#     """
#     t = _norm(x).replace("_", "-").replace("–", "-").replace("—", "-")
#     if ("non" in t and "surg" in t) or ("nonsurg" in t):
#         return "non-surgical"
#     if "non" in t:
#         return "non-surgical"
#     if "surg" in t:
#         return "surgical"
#     return ""


# def _norm_type_choice(choice: str) -> str:
#     c = _norm(choice)
#     if "both" in c:
#         return "both"
#     if ("non" in c and "surg" in c) or ("non" in c):
#         return "non-surgical"
#     if "surg" in c:
#         return "surgical"
#     return "both"


# def _to_proc_type(db_type: str) -> str:
#     t = _norm(db_type)
#     if ("non" in t and "surg" in t) or ("non" in t):
#         return "Non-Surgical"
#     if "surg" in t:
#         return "Surgical"
#     return "Not found in database."


# def _db_str(v) -> str:
#     if v is None:
#         return ""
#     if isinstance(v, float) and np.isnan(v):
#         return ""
#     s = str(v).strip()
#     if s.lower() == "nan":
#         return ""
#     return s


# def _first_present(row: pd.Series, keys: List[str]) -> str:
#     for k in keys:
#         if k in row.index:
#             val = _db_str(row.get(k, ""))
#             if val:
#                 return val
#     return ""


# def _na_db(v: str) -> str:
#     return v if v else "Not found in database."


# # ---------------------------- data model ----------------------------

# @dataclass
# class RetrievedCandidate:
#     row_index: int
#     similarity: float

#     procedure: str
#     region: str
#     sub_zone: str
#     type: str

#     short_description: str
#     concerns: str
#     techniques: str

#     expected_results: str
#     procedure_duration_hours: str
#     downtime_days: str
#     results_visible_timeline: str
#     result_duration: str
#     potential_side_effects: str
#     anesthesia_type: str
#     hospital_stay: str
#     protocol_type: str
#     session_frequency: str

#     average_cost_min_eur: str
#     average_cost_max_eur: str
#     average_cost_min_chf: str
#     average_cost_max_chf: str


# # ---------------------------- app ----------------------------

# class RAGTreatmentSearchApp:
#     """
#     HF-ready local structured RAG (DB-based details).

#     DB: database.xlsx (NEW schema)
#       - Uses sheet_name default: "Procedures"
#       - Reads procedure details from DB columns (no web calls)

#     API is kept compatible with your existing gradio_new_rag_app.py:
#       RAGTreatmentSearchApp(excel_path=..., embeddings_cache_path=...)
#     """

#     def __init__(
#         self,
#         excel_path: str = "database.xlsx",
#         sheet_name: str = "Procedures",
#         embeddings_cache_path: str = "treatment_embeddings.pkl",
#         embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
#         llm: Optional[LocalLLMClient] = None,
#     ):
#         try:
#             torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
#         except Exception:
#             pass

#         self.excel_path = excel_path
#         self.sheet_name = sheet_name
#         self.embeddings_cache_path = embeddings_cache_path

#         self.df = self._load_db()
#         self._normalize_columns()

#         self.model = SentenceTransformer(embedding_model_name, device="cpu")
#         self.embeddings, self.texts = self._load_or_build_embeddings()

#         self.llm = llm or LocalLLMClient()

#         # hard gate: avoid returning output when issue is empty
#         self.min_issue_chars = int(os.getenv("MIN_ISSUE_CHARS", "5"))

#         # mismatch sensitivity (tuned)
#         self.local_issue_min_sim = float(os.getenv("LOCAL_ISSUE_MIN_SIM", "0.42"))
#         self.global_issue_min_sim = float(os.getenv("GLOBAL_ISSUE_MIN_SIM", "0.52"))
#         self.global_local_delta = float(os.getenv("GLOBAL_LOCAL_DELTA", "0.10"))

#     # ---------------- DB ----------------

#     def _load_db(self) -> pd.DataFrame:
#         xl = pd.ExcelFile(self.excel_path)
#         if self.sheet_name not in xl.sheet_names:
#             raise ValueError(f"Sheet '{self.sheet_name}' not found. Found: {xl.sheet_names}")
#         return pd.read_excel(self.excel_path, sheet_name=self.sheet_name)

#     def _normalize_columns(self) -> None:
#         """
#         Supports the NEW schema you described.
#         We also create UI-friendly aliases: Region, Sub-Zone, Procedure, Type.
#         """
#         # Required minimal new schema keys (based on your DB update)
#         required_any = [
#             "procedure_title",
#             "main_zone",
#             "treatment_type",
#         ]
#         missing_any = [c for c in required_any if c not in self.df.columns]
#         if missing_any:
#             raise ValueError(f"Database missing required columns: {missing_any}")

#         # Build unified Region/Sub-Zone fields
#         # Region -> main_zone
#         self.df["Region"] = self.df["main_zone"].fillna("").astype(str).str.strip()

#         # Sub-Zone: prefer face_subzone else body_subzone else any existing fallback
#         if "face_subzone" in self.df.columns or "body_subzone" in self.df.columns:
#             face = self.df["face_subzone"].fillna("").astype(str).str.strip() if "face_subzone" in self.df.columns else ""
#             body = self.df["body_subzone"].fillna("").astype(str).str.strip() if "body_subzone" in self.df.columns else ""
#             sub = face
#             if isinstance(sub, str):
#                 # shouldn't happen, but keep safe
#                 sub = ""
#             self.df["Sub-Zone"] = face
#             mask_empty = self.df["Sub-Zone"].eq("") | self.df["Sub-Zone"].str.lower().eq("nan")
#             if not isinstance(body, str):
#                 self.df.loc[mask_empty, "Sub-Zone"] = body.loc[mask_empty]
#         else:
#             # last fallback if DB already has something named Sub-Zone
#             self.df["Sub-Zone"] = self.df.get("Sub-Zone", "").fillna("").astype(str).str.strip()

#         # Procedure/Type
#         self.df["Procedure"] = self.df["procedure_title"].fillna("").astype(str).str.strip()
#         self.df["Type"] = self.df["treatment_type"].fillna("").astype(str).str.strip()

#         # Normalize core columns
#         for col in ["Type", "Region", "Sub-Zone", "Procedure"]:
#             self.df[col] = self.df[col].astype(str).fillna("").str.strip()

#         self.df["_region_norm"] = self.df["Region"].apply(_norm)
#         self.df["_subzone_norm"] = self.df["Sub-Zone"].apply(_norm)
#         self.df["_type_norm"] = self.df["Type"].apply(_norm_type_value)
#         # Build searchable DB sub-zone vocabulary for fuzzy matching
#         self._db_subzone_terms = self._build_db_subzone_terms()


#     def get_regions(self) -> List[str]:
#         regions = [r for r in self.df["Region"].dropna().unique().tolist() if str(r).strip() and str(r).strip().lower() != "nan"]
#         return sorted(regions)

#     def get_sub_zones(self, region: str) -> List[str]:
#         r = _norm(region)
#         sub = self.df[self.df["_region_norm"].eq(r)]["Sub-Zone"].dropna().unique().tolist()
#         out = []
#         for s in sub:
#             ss = str(s).strip()
#             if ss and ss.lower() != "nan":
#                 out.append(ss)
#         return sorted(out)
#     # ---------------- Sub-zone fuzzy matching ----------------

#     def _split_db_subzone(self, s: str) -> List[str]:
#         """
#         DB often has composite strings like:
#           "Cheeks, Cheekbones" or "Jawline / Jowls"
#         Split into individual normalized tokens.
#         """
#         s = _norm(s)
#         if not s:
#             return []
#         parts = re.split(r"[,\|/]|(?:\band\b)", s)
#         parts = [_norm(p) for p in parts]
#         return [p for p in parts if p]

#     def _build_db_subzone_terms(self) -> List[str]:
#         """
#         Build a unique list of normalized sub-zone terms present in DB.
#         Includes both full strings and split components.
#         """
#         terms = set()

#         for raw in self.df["Sub-Zone"].fillna("").astype(str).tolist():
#             raw = raw.strip()
#             if not raw or raw.lower() == "nan":
#                 continue

#             terms.add(_norm(raw))
#             for p in self._split_db_subzone(raw):
#                 terms.add(p)

#         return sorted([t for t in terms if t])

#     def _resolve_db_subzones(self, requested_sub_zone: str, max_matches: int = 5) -> List[str]:
#         """
#         Generalized closest-match: map requested sub_zone (from hardcoded list)
#         to the closest DB sub-zone term(s).

#         Examples:
#           "teartrough" -> ["tear trough"]
#           "eyes"       -> returns closest DB terms (e.g. "eyelids", etc.) if present
#         """
#         q = _norm(requested_sub_zone)
#         if not q:
#             return []

#         # 1) Exact match
#         if q in self._db_subzone_terms:
#             return [q]

#         # 2) Substring-based matches (handles "teartrough" vs "tear trough")
#         substring_hits = [t for t in self._db_subzone_terms if (q in t or t in q)]
#         if substring_hits:
#             substring_hits = sorted(substring_hits, key=len)
#             return substring_hits[:max_matches]

#         # 3) Fuzzy match
#         return difflib.get_close_matches(q, self._db_subzone_terms, n=max_matches, cutoff=0.62)

#     def _subzone_mask(self, sub_zone: str) -> "pd.Series":
#         """
#         Boolean mask of rows whose DB sub-zone matches the requested sub_zone
#         using resolved closest DB terms.
#         """
#         keys = self._resolve_db_subzones(sub_zone)
#         if not keys:
#             return pd.Series([False] * len(self.df), index=self.df.index)

#         m = pd.Series([False] * len(self.df), index=self.df.index)
#         for k in keys:
#             # Match whole-ish word boundaries within composite DB strings
#             pat = r"(^|[\s,;/\-|])" + re.escape(k) + r"($|[\s,;/\-|])"
#             m = m | self.df["_subzone_norm"].str.contains(pat, na=False, regex=True)

#         return m

#     def get_concerns_for_subzone(self, sub_zone: str) -> List[str]:
#         """
#         Get all unique concerns from database for a given sub-zone.
#         Extracts concerns from the 'concerns' column.
#         """
#         # sz = _norm(sub_zone)
#         # filtered_df = self.df[self.df["_subzone_norm"].eq(sz) | self.df["_subzone_norm"].str.contains(sz, na=False)]
#         filtered_df = self.df[self._subzone_mask(sub_zone)]

#         all_concerns = set()
#         for _, row in filtered_df.iterrows():
#             concerns_text = _first_present(row, ["concerns", "aesthetic_concerns", "Aesthetic Concerns"])
#             if concerns_text and concerns_text != "Not found in database.":
#                 # Split by common separators: comma, semicolon, pipe, newline
#                 items = concerns_text.replace(";", ",").replace("|", ",").replace("\n", ",").split(",")
#                 for item in items:
#                     cleaned = item.strip()
#                     if cleaned and len(cleaned) > 2:  # Filter out very short items
#                         all_concerns.add(cleaned)
        
#         return sorted(list(all_concerns))

#     def get_region_from_subzone(self, sub_zone: str) -> str:
#         """
#         Find the region that contains the given sub-zone.
#         Returns empty string if not found.
#         """
#         # sz = _norm(sub_zone)
#         # matches = self.df[self.df["_subzone_norm"].eq(sz) | self.df["_subzone_norm"].str.contains(sz, na=False)]
#         matches = self.df[self._subzone_mask(sub_zone)]

#         if len(matches) > 0:
#             region = matches.iloc[0]["Region"]
#             return str(region).strip()
        
#         return ""

#     def search_by_concerns(
#         self,
#         region: str,
#         sub_zone: str,
#         type_choice: str,
#         concerns: List[str],
#         retrieval_k: int = 12,
#         final_k: int = 5,
#     ) -> dict:
#         """
#         Search for procedures based on array of concerns.
#         Returns only procedure names (not full details).
        
#         Args:
#             region: Selected region
#             sub_zone: Selected sub-zone
#             type_choice: Treatment preference (Surgical/Non-Surgical/Both)
#             concerns: List of concern strings
#             retrieval_k: Number of candidates to retrieve
#             final_k: Number of final recommendations
            
#         Returns:
#             dict with:
#             - mismatch: bool
#             - notice: str (if mismatch)
#             - recommended_procedures: List[str] (procedure names only)
#             - suggested_region_subzones: List[dict] (if mismatch)
#         """
#         region = (region or "").strip()
#         sub_zone = (sub_zone or "").strip()
        
#         if not region or not sub_zone:
#             return {
#                 "mismatch": False,
#                 "notice": "Region and sub-zone are required",
#                 "recommended_procedures": [],
#                 "suggested_region_subzones": []
#             }
        
#         if not concerns:
#             return {
#                 "mismatch": False,
#                 "notice": "At least one concern is required",
#                 "recommended_procedures": [],
#                 "suggested_region_subzones": []
#             }
        
#         # Combine concerns into issue text for semantic search
#         issue_text = ", ".join(concerns)
        
#         # Use the existing recommend method
#         result = self.recommend(
#             region=region,
#             sub_zone=sub_zone,
#             type_choice=type_choice,
#             issue_text=issue_text,
#             retrieval_k=retrieval_k,
#             final_k=final_k,
#         )
        
#         # Transform result to return only procedure names
#         if result["status"] == "mismatch":
#             return {
#                 "mismatch": True,
#                 "notice": "Your selected sub-zone does not match your concerns.",
#                 "recommended_procedures": [],
#                 "suggested_region_subzones": result.get("suggested_region_subzones", [])
#             }
        
#         # Extract only procedure names from recommended_procedures
#         procedure_names = []
#         for proc in result.get("recommended_procedures", []):
#             if isinstance(proc, dict) and "procedure_name" in proc:
#                 procedure_names.append(proc["procedure_name"])
        
#         return {
#             "mismatch": False,
#             "notice": "",
#             "recommended_procedures": procedure_names,
#             "suggested_region_subzones": []
#         }

#     # ---------------- Embeddings ----------------

#     def _row_to_text(self, row: pd.Series) -> str:
#         """
#         Build semantic text from DB fields (for embeddings).
#         Keep it compact but informative so issue-only similarity works.
#         """
#         proc = _db_str(row.get("procedure_title", ""))
#         reg = _db_str(row.get("main_zone", ""))
#         sub = _db_str(row.get("Sub-Zone", ""))
#         typ = _db_str(row.get("treatment_type", ""))

#         short_desc = _first_present(row, ["short_description", "procedure_description", "description"])
#         concerns = _first_present(row, ["concerns", "aesthetic_concerns", "Aesthetic Concerns"])
#         techniques = _first_present(row, ["techniques_brands_variants", "Technique / Technology / Brand", "techniques"])

#         expected = _first_present(row, ["expected_results", "expected_result"])
#         sidefx = _first_present(row, ["potential_side_effects", "side_effects", "risks"])

#         parts = [
#             f"Procedure: {proc}",
#             f"Type: {typ}",
#             f"Region: {reg}",
#             f"Sub-Zone: {sub}",
#             f"Description: {short_desc}",
#             f"Concerns: {concerns}",
#             f"Techniques: {techniques}",
#             f"Expected: {expected}",
#             f"Side effects: {sidefx}",
#         ]
#         return " | ".join([p for p in parts if p and not p.endswith(": ")])

#     def _load_or_build_embeddings(self) -> Tuple[np.ndarray, List[str]]:
#         if os.path.exists(self.embeddings_cache_path):
#             try:
#                 with open(self.embeddings_cache_path, "rb") as f:
#                     data = pickle.load(f)
#                 emb = np.array(data["embeddings"], dtype=np.float32)
#                 txt = list(data["texts"])
#                 if emb.shape[0] == len(self.df) and len(txt) == len(self.df):
#                     return emb, txt
#             except Exception:
#                 pass
#         return self._build_embeddings()

#     def _build_embeddings(self) -> Tuple[np.ndarray, List[str]]:
#         texts = [self._row_to_text(self.df.iloc[i]) for i in range(len(self.df))]
#         embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
#         payload = {"created_at": time.time(), "texts": texts, "embeddings": embeddings.tolist()}
#         with open(self.embeddings_cache_path, "wb") as f:
#             pickle.dump(payload, f)
#         return embeddings, texts

#     # ---------------- Retrieval ----------------

#     # def _candidate_indices(self, region: str, sub_zone: str, type_norm: str) -> np.ndarray:
#     #     r = _norm(region)
#     #     sz = _norm(sub_zone)

#     #     m = self.df["_region_norm"].eq(r)
#     #     if sz:
#     #         m = m & (self.df["_subzone_norm"].eq(sz) | self.df["_subzone_norm"].str.contains(sz, na=False))
#     #     if type_norm in {"surgical", "non-surgical"}:
#     #         m = m & self.df["_type_norm"].eq(type_norm)

#     #     return np.where(m.values)[0]
#     def _candidate_indices(self, region: str, sub_zone: str, type_norm: str) -> np.ndarray:
#         r = _norm(region)

#         m = self.df["_region_norm"].eq(r)
#         if sub_zone and sub_zone.strip():
#             m = m & self._subzone_mask(sub_zone)

#         if type_norm in {"surgical", "non-surgical"}:
#             m = m & self.df["_type_norm"].eq(type_norm)

#         return np.where(m.values)[0]

#     def _semantic_over(self, idxs: np.ndarray, query: str, top_k: int) -> List[RetrievedCandidate]:
#         if idxs.size == 0:
#             return []
#         q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
#         sims = cosine_similarity(q_emb, self.embeddings[idxs])[0]
#         order = sims.argsort()[::-1]

#         out: List[RetrievedCandidate] = []
#         for pos in order[: max(top_k, 1) * 10]:
#             row_index = int(idxs[pos])
#             row = self.df.iloc[row_index]

#             proc = _db_str(row.get("procedure_title", "")) or _db_str(row.get("Procedure", ""))
#             reg = _db_str(row.get("main_zone", "")) or _db_str(row.get("Region", ""))
#             sub = _db_str(row.get("Sub-Zone", "")) or _db_str(row.get("face_subzone", "")) or _db_str(row.get("body_subzone", ""))
#             typ = _db_str(row.get("treatment_type", "")) or _db_str(row.get("Type", ""))

#             out.append(
#                 RetrievedCandidate(
#                     row_index=row_index,
#                     similarity=float(sims[pos]),
#                     procedure=_na_db(proc),
#                     region=_na_db(reg),
#                     sub_zone=_na_db(sub),
#                     type=_na_db(typ),

#                     short_description=_na_db(_first_present(row, ["short_description", "procedure_description", "description"])),
#                     concerns=_na_db(_first_present(row, ["concerns", "aesthetic_concerns", "Aesthetic Concerns"])),
#                     techniques=_na_db(_first_present(row, ["techniques_brands_variants", "Technique / Technology / Brand", "techniques"])),

#                     expected_results=_na_db(_first_present(row, ["expected_results"])),
#                     procedure_duration_hours=_na_db(_first_present(row, ["procedure_duration_hours"])),
#                     downtime_days=_na_db(_first_present(row, ["downtime_days"])),
#                     results_visible_timeline=_na_db(_first_present(row, ["results_visible_timeline"])),
#                     result_duration=_na_db(_first_present(row, ["result_duration"])),
#                     potential_side_effects=_na_db(_first_present(row, ["potential_side_effects"])),
#                     anesthesia_type=_na_db(_first_present(row, ["anesthesia_type"])),
#                     hospital_stay=_na_db(_first_present(row, ["hospital_stay"])),
#                     protocol_type=_na_db(_first_present(row, ["protocol_type"])),
#                     session_frequency=_na_db(_first_present(row, ["session_frequency"])),

#                     average_cost_min_eur=_na_db(_first_present(row, ["average_cost_min_eur"])),
#                     average_cost_max_eur=_na_db(_first_present(row, ["average_cost_max_eur"])),
#                     average_cost_min_chf=_na_db(_first_present(row, ["average_cost_min_chf"])),
#                     average_cost_max_chf=_na_db(_first_present(row, ["average_cost_max_chf"])),
#                 )
#             )
#             if len(out) >= top_k:
#                 break
#         return out

#     def _global_semantic(self, issue_text: str, top_k: int = 15) -> List[RetrievedCandidate]:
#         if not issue_text.strip():
#             return []
#         q_emb = self.model.encode([issue_text], convert_to_numpy=True).astype(np.float32)
#         sims = cosine_similarity(q_emb, self.embeddings)[0]
#         order = sims.argsort()[::-1]

#         out: List[RetrievedCandidate] = []
#         for idx in order[: max(top_k, 1) * 20]:
#             row = self.df.iloc[int(idx)]
#             # Build minimal candidate (details not required for mismatch suggestion list)
#             proc = _db_str(row.get("procedure_title", "")) or _db_str(row.get("Procedure", ""))
#             reg = _db_str(row.get("main_zone", "")) or _db_str(row.get("Region", ""))
#             sub = _db_str(row.get("Sub-Zone", "")) or _db_str(row.get("face_subzone", "")) or _db_str(row.get("body_subzone", ""))
#             typ = _db_str(row.get("treatment_type", "")) or _db_str(row.get("Type", ""))

#             out.append(
#                 RetrievedCandidate(
#                     row_index=int(idx),
#                     similarity=float(sims[idx]),
#                     procedure=_na_db(proc),
#                     region=_na_db(reg),
#                     sub_zone=_na_db(sub),
#                     type=_na_db(typ),

#                     short_description="Not found in database.",
#                     concerns="Not found in database.",
#                     techniques="Not found in database.",

#                     expected_results="Not found in database.",
#                     procedure_duration_hours="Not found in database.",
#                     downtime_days="Not found in database.",
#                     results_visible_timeline="Not found in database.",
#                     result_duration="Not found in database.",
#                     potential_side_effects="Not found in database.",
#                     anesthesia_type="Not found in database.",
#                     hospital_stay="Not found in database.",
#                     protocol_type="Not found in database.",
#                     session_frequency="Not found in database.",

#                     average_cost_min_eur="Not found in database.",
#                     average_cost_max_eur="Not found in database.",
#                     average_cost_min_chf="Not found in database.",
#                     average_cost_max_chf="Not found in database.",
#                 )
#             )
#             if len(out) >= top_k:
#                 break
#         return out

#     def _local_issue_only_best_sim(self, region: str, sub_zone: str, type_choice: str, issue_text: str) -> float:
#         """
#         Compute issue-only similarity inside selected region/sub-zone to detect irrelevance.
#         """
#         issue_text = (issue_text or "").strip()
#         if not issue_text:
#             return 0.0

#         t = _norm_type_choice(type_choice)
#         if t == "both":
#             idx_s = self._candidate_indices(region, sub_zone, "surgical")
#             idx_n = self._candidate_indices(region, sub_zone, "non-surgical")
#             idxs = np.unique(np.concatenate([idx_s, idx_n])) if (idx_s.size or idx_n.size) else np.array([], dtype=int)
#         else:
#             idxs = self._candidate_indices(region, sub_zone, t)

#         if idxs.size == 0:
#             # region only
#             if t == "both":
#                 idx_s = self._candidate_indices(region, "", "surgical")
#                 idx_n = self._candidate_indices(region, "", "non-surgical")
#                 idxs = np.unique(np.concatenate([idx_s, idx_n])) if (idx_s.size or idx_n.size) else np.array([], dtype=int)
#             else:
#                 idxs = self._candidate_indices(region, "", t)

#         if idxs.size == 0:
#             return 0.0

#         q_emb = self.model.encode([issue_text], convert_to_numpy=True).astype(np.float32)
#         sims = cosine_similarity(q_emb, self.embeddings[idxs])[0]
#         return float(np.max(sims)) if sims.size else 0.0

#     def semantic_search(
#         self,
#         region: str,
#         sub_zone: str,
#         type_choice: str,
#         issue_text: str,
#         top_k: int = 12,
#     ) -> List[RetrievedCandidate]:
#         type_norm = _norm_type_choice(type_choice)

#         query = f"Region: {region} | Sub-Zone: {sub_zone} | Preference: {type_choice} | Issue: {issue_text}"

#         if type_norm == "both":
#             idx_s = self._candidate_indices(region, sub_zone, "surgical")
#             idx_n = self._candidate_indices(region, sub_zone, "non-surgical")
#             per = max(3, top_k // 2)
#             res = self._semantic_over(idx_s, query, per) + self._semantic_over(idx_n, query, per)
#             res.sort(key=lambda x: x.similarity, reverse=True)
#             # de-dupe by row index
#             seen = set()
#             out = []
#             for c in res:
#                 if c.row_index in seen:
#                     continue
#                 seen.add(c.row_index)
#                 out.append(c)
#                 if len(out) >= top_k:
#                     break
#             return out

#         idx = self._candidate_indices(region, sub_zone, type_norm)
#         if idx.size == 0:
#             idx = self._candidate_indices(region, "", type_norm)
#         return self._semantic_over(idx, query, top_k)

#     # ---------------- LLM rerank ----------------

#     def _llm_rerank(self, issue_text: str, candidates: List[RetrievedCandidate], top_k: int) -> List[RetrievedCandidate]:
#         if not candidates:
#             return []

#         cand_block = "\n".join([f"- {c.procedure} (Type: {c.type}, Sub-Zone: {c.sub_zone})" for c in candidates])
#         prompt = f"""
# Pick the best {top_k} procedure names for:
# "{issue_text}"

# Candidates:
# {cand_block}

# Return ONLY a comma-separated list of procedure names (exactly as written).
# """.strip()

#         raw = (self.llm.generate(prompt, temperature=0.1, max_tokens=90) or "").strip()
#         names = [n.strip() for n in raw.split(",") if n.strip()]
#         if not names:
#             return candidates[:top_k]

#         lookup = {c.procedure.lower(): c for c in candidates}
#         out: List[RetrievedCandidate] = []
#         for n in names:
#             c = lookup.get(n.lower())
#             if c and c not in out:
#                 out.append(c)
#             if len(out) >= top_k:
#                 break

#         # fill remainder
#         for c in candidates:
#             if len(out) >= top_k:
#                 break
#             if c not in out:
#                 out.append(c)

#         return out

#     # ---------------- Formatting (DB details) ----------------

#     def _format_cost(self, mn: str, mx: str, unit: str) -> str:
#         if mn == "Not found in database." and mx == "Not found in database.":
#             return "Not found in database."
#         if mn == "Not found in database." and mx != "Not found in database.":
#             return f"Up to {mx} {unit}"
#         if mx == "Not found in database." and mn != "Not found in database.":
#             return f"From {mn} {unit}"
#         return f"{mn} – {mx} {unit}"

#     def _format_procedure_block(self, c: RetrievedCandidate, idx: int) -> str:
#         proc_type = _to_proc_type(c.type)

#         desc = c.short_description
#         if desc == "Not found in database." and c.concerns != "Not found in database.":
#             desc = c.concerns

#         cost_eur = self._format_cost(c.average_cost_min_eur, c.average_cost_max_eur, "EUR")
#         cost_chf = self._format_cost(c.average_cost_min_chf, c.average_cost_max_chf, "CHF")

#         lines = [
#             f"## Procedure {idx}: {c.procedure}",
#             f"- **Procedure type:** {proc_type}",
#             f"- **Region / Sub-Zone (DB):** {c.region} / {c.sub_zone}",
#             f"- **Description (DB):** {desc}",
#             f"- **Concerns (DB):** {c.concerns}",
#             f"- **Techniques / brands (DB):** {c.techniques}",
#             f"- **Expected results (DB):** {c.expected_results}",
#             f"- **Protocol type (DB):** {c.protocol_type}",
#             f"- **Session frequency (DB):** {c.session_frequency}",
#             f"- **Anesthesia (DB):** {c.anesthesia_type}",
#             f"- **Hospital stay (DB):** {c.hospital_stay}",
#             f"- **Procedure duration (hours) (DB):** {c.procedure_duration_hours}",
#             f"- **Downtime (days) (DB):** {c.downtime_days}",
#             f"- **Results visible timeline (DB):** {c.results_visible_timeline}",
#             f"- **Result duration / longevity (DB):** {c.result_duration}",
#             f"- **Potential side effects (DB):** {c.potential_side_effects}",
#             f"- **Average cost (EUR) (DB):** {cost_eur}",
#             f"- **Average cost (CHF) (DB):** {cost_chf}",
#         ]
#         return "\n".join(lines).strip()

#     def _format_final_answer(self, best: List[RetrievedCandidate]) -> str:
#         blocks = [self._format_procedure_block(c, i + 1) for i, c in enumerate(best)]
#         blocks.append("\n---\n**Safety disclaimer:** This tool is for informational purposes only and is not medical advice. Please consult a licensed clinician.")
#         return "\n\n".join(blocks).strip()

#     # ---------------- Main API ----------------

#     def _candidate_to_result_obj(self, c: RetrievedCandidate) -> Dict[str, object]:
#         """Convert a RetrievedCandidate into a JSON-serializable object for API responses."""
#         proc_type = _to_proc_type(c.type)

#         desc = c.short_description
#         if desc == "Not found in database." and c.concerns != "Not found in database.":
#             desc = c.concerns

#         cost_eur = self._format_cost(c.average_cost_min_eur, c.average_cost_max_eur, "EUR")
#         cost_chf = self._format_cost(c.average_cost_min_chf, c.average_cost_max_chf, "CHF")

#         return {
#             "procedure_name": c.procedure,
#             "procedure_type": proc_type,
#             "db_region": c.region,
#             "db_sub_zone": c.sub_zone,
#             "db_description": desc,
#             "db_concerns": c.concerns,
#             "db_techniques_brands": c.techniques,
#             "expected_results": c.expected_results,
#             "protocol_type": c.protocol_type,
#             "session_frequency": c.session_frequency,
#             "anesthesia_type": c.anesthesia_type,
#             "hospital_stay": c.hospital_stay,
#             "procedure_duration_hours": c.procedure_duration_hours,
#             "downtime_days": c.downtime_days,
#             "results_visible_timeline": c.results_visible_timeline,
#             "result_duration": c.result_duration,
#             "potential_side_effects": c.potential_side_effects,
#             "average_cost": {
#                 "eur": cost_eur,
#                 "chf": cost_chf,
#             },
#             "similarity": round(float(c.similarity), 4),
#         }

#     def recommend(
#         self,
#         region: str,
#         sub_zone: str,
#         type_choice: str,
#         issue_text: str,
#         retrieval_k: int = 12,
#         final_k: int = 5,
#     ) -> Dict[str, object]:
#         """
#         Structured API response for backend integration (DigitalOcean etc.).

#         Returns:
#           - status: blocked | mismatch | ok
#           - message: human-readable note
#           - suggested_region_subzones: only when mismatch
#           - recommended_procedures: list[dict] only when ok
#           - debug: internal signals
#         """
#         region = (region or "").strip()
#         sub_zone = (sub_zone or "").strip()
#         issue_text = (issue_text or "").strip()

#         if not region or not sub_zone:
#             return {
#                 "status": "blocked",
#                 "message": "Please provide both region and sub_zone.",
#                 "recommended_procedures": [],
#                 "suggested_region_subzones": [],
#                 "debug": {"blocked": True, "reason": "missing_region_or_subzone"},
#             }

#         if len(issue_text) < self.min_issue_chars:
#             return {
#                 "status": "blocked",
#                 "message": f"Please provide a problem statement (minimum {self.min_issue_chars} characters).",
#                 "recommended_procedures": [],
#                 "suggested_region_subzones": [],
#                 "debug": {"blocked": True, "reason": "missing_issue_text"},
#             }

#         candidates = self.semantic_search(region, sub_zone, type_choice, issue_text, top_k=int(retrieval_k))
#         if not candidates:
#             return {
#                 "status": "blocked",
#                 "message": "No matching procedures found for the selected region/sub-zone and problem statement. Please revise inputs.",
#                 "recommended_procedures": [],
#                 "suggested_region_subzones": [],
#                 "debug": {"blocked": True, "reason": "no_candidates"},
#             }

#         # mismatch detection (same logic as answer())
#         global_cands = self._global_semantic(issue_text, top_k=15)
#         global_best = global_cands[0].similarity if global_cands else 0.0
#         local_best = candidates[0].similarity if candidates else 0.0

#         selected_region_norm = _norm(region)
#         selected_sub_norm = _norm(sub_zone)

#         selected_in_global = any(
#             _norm(c.region) == selected_region_norm and (
#                 selected_sub_norm in _norm(c.sub_zone) or _norm(c.sub_zone) in selected_sub_norm
#             )
#             for c in global_cands[:10]
#         )

#         local_issue_best = self._local_issue_only_best_sim(region, sub_zone, type_choice, issue_text)

#         mismatch_strict = (
#             (local_issue_best > 0.0 and local_issue_best < self.local_issue_min_sim)
#             and (global_best >= self.global_issue_min_sim)
#             and (not selected_in_global)
#         )

#         mismatch_delta = (
#             (global_best >= self.global_issue_min_sim)
#             and ((global_best - local_best) >= self.global_local_delta)
#             and (not selected_in_global)
#         )

#         if mismatch_strict or mismatch_delta:
#             suggestions = []
#             seen = set()
#             for c in global_cands:
#                 key = (c.region, c.sub_zone)
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 suggestions.append({"region": c.region, "sub_zone": c.sub_zone})
#                 if len(suggestions) >= 8:
#                     break

#             return {
#                 "status": "mismatch",
#                 "message": "Selected region/sub-zone does not match the problem statement. Please choose a more appropriate region/sub-zone.",
#                 "recommended_procedures": [],
#                 "suggested_region_subzones": suggestions,
#                 "debug": {
#                     "mismatch": True,
#                     "global_best": round(float(global_best), 4),
#                     "local_best": round(float(local_best), 4),
#                     "local_issue_best": round(float(local_issue_best), 4),
#                     "candidate_count": len(candidates),
#                 },
#             }

#         best = self._llm_rerank(issue_text, candidates, top_k=int(final_k))
#         if len(best) < int(final_k):
#             for c in candidates:
#                 if c not in best:
#                     best.append(c)
#                 if len(best) >= int(final_k):
#                     break
#         best = best[: int(final_k)]

#         results = [self._candidate_to_result_obj(c) for c in best]

#         return {
#             "status": "ok",
#             "message": "Success",
#             "recommended_procedures": results,
#             "suggested_region_subzones": [],
#             "debug": {
#                 "mismatch": False,
#                 "candidate_count": len(candidates),
#                 "final_count": len(best),
#                 "candidates": [
#                     {
#                         "procedure": c.procedure,
#                         "similarity": round(float(c.similarity), 4),
#                         "type": c.type,
#                         "region": c.region,
#                         "sub_zone": c.sub_zone,
#                     }
#                     for c in candidates[:20]
#                 ],
#             },
#         }



#     def answer(
#         self,
#         region: str,
#         sub_zone: str,
#         type_choice: str,
#         issue_text: str,
#         retrieval_k: int = 12,
#         final_k: int = 5,
#     ) -> Dict[str, object]:

#         region = (region or "").strip()
#         sub_zone = (sub_zone or "").strip()
#         issue_text = (issue_text or "").strip()

#         # Hard gate: must provide issue text
#         if not region or not sub_zone:
#             return {
#                 "answer_md": "Please select **Region** and **Sub-Zone** before running the search.",
#                 "sources": [],
#                 "_debug": {"mismatch": False, "blocked": True, "reason": "missing_region_or_subzone"},
#             }

#         if len(issue_text) < self.min_issue_chars:
#             return {
#                 "answer_md": f"Please describe your issue/problem (minimum {self.min_issue_chars} characters) to get recommendations.",
#                 "sources": [],
#                 "_debug": {"mismatch": False, "blocked": True, "reason": "missing_issue_text"},
#             }

#         candidates = self.semantic_search(region, sub_zone, type_choice, issue_text, top_k=int(retrieval_k))
#         if not candidates:
#             return {
#                 "answer_md": "No matching procedures found for your selected Region/Sub-Zone and issue. Please revise your inputs.",
#                 "sources": [],
#                 "_debug": {"mismatch": False, "candidate_count": 0, "final_count": 0},
#             }

#         # ---------- mismatch detection ----------
#         global_cands = self._global_semantic(issue_text, top_k=15)
#         global_best = global_cands[0].similarity if global_cands else 0.0
#         local_best = candidates[0].similarity if candidates else 0.0

#         selected_region_norm = _norm(region)
#         selected_sub_norm = _norm(sub_zone)

#         selected_in_global = any(
#             _norm(c.region) == selected_region_norm and (
#                 selected_sub_norm in _norm(c.sub_zone) or _norm(c.sub_zone) in selected_sub_norm
#             )
#             for c in global_cands[:10]
#         )

#         local_issue_best = self._local_issue_only_best_sim(region, sub_zone, type_choice, issue_text)

#         mismatch_strict = (
#             (local_issue_best > 0.0 and local_issue_best < self.local_issue_min_sim)
#             and (global_best >= self.global_issue_min_sim)
#             and (not selected_in_global)
#         )

#         mismatch_delta = (
#             (global_best >= self.global_issue_min_sim)
#             and ((global_best - local_best) >= self.global_local_delta)
#             and (not selected_in_global)
#         )

#         if mismatch_strict or mismatch_delta:
#             # suggest correct region/sub-zones based on issue text
#             suggestions = []
#             seen = set()
#             for c in global_cands:
#                 key = (c.region, c.sub_zone)
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 suggestions.append(key)
#                 if len(suggestions) >= 8:
#                     break

#             sug_lines = "\n".join([f"- **{r} → {sz}**" for (r, sz) in suggestions]) or "- (No suggestions found)"

#             answer_md = f"""## Notice (Mismatch Detected)
# Sorry for inconvenience. Your selected **Region/Sub-Zone** does not match your problem description.

# ### Your input
# - **Selected Region:** {region}
# - **Selected Sub-Zone:** {sub_zone}
# - **Issue text:** {issue_text}

# ### Suggested Region/Sub-Zones (from your DB)
# {sug_lines}

# ### Next step
# Please select one of the suggested **Region/Sub-Zones** and run the search again.
# """.strip()

#             return {
#                 "answer_md": answer_md,
#                 "sources": [],
#                 "_debug": {
#                     "mismatch": True,
#                     "global_best": round(global_best, 4),
#                     "local_best": round(local_best, 4),
#                     "local_issue_best": round(local_issue_best, 4),
#                     "candidate_count": len(candidates),
#                 },
#             }
#         # ---------------------------------------

#         best = self._llm_rerank(issue_text, candidates, top_k=int(final_k))

#         # Ensure exactly final_k if possible
#         if len(best) < int(final_k):
#             for c in candidates:
#                 if c not in best:
#                     best.append(c)
#                 if len(best) >= int(final_k):
#                     break
#         best = best[: int(final_k)]

#         answer_md = self._format_final_answer(best)

#         return {
#             "answer_md": answer_md,
#             "sources": [],  # DB-only mode
#             "_debug": {
#                 "mismatch": False,
#                 "candidate_count": len(candidates),
#                 "final_count": len(best),
#                 "candidates": [
#                     {
#                         "procedure": c.procedure,
#                         "similarity": round(c.similarity, 4),
#                         "type": c.type,
#                         "region": c.region,
#                         "sub_zone": c.sub_zone,
#                     }
#                     for c in candidates[:20]
#                 ],
#             },
#         }


#============= Updated code to handle duplication - 3 april 2026 - 11:10 AM ======================


#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
import time
import difflib
import re

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import LocalLLMClient


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/static-similarity-mrl-multilingual-v1"


# ---------------------------- helpers ----------------------------

def _norm(x: str) -> str:
    return " ".join(str(x or "").strip().lower().split())


def _norm_type_value(x: str) -> str:
    """
    Normalize DB type to {surgical, non-surgical, ""}.
    Handles many variants: Non surgical, non-surg, non-surgical, etc.
    """
    t = _norm(x).replace("_", "-").replace("–", "-").replace("—", "-")
    if ("non" in t and "surg" in t) or ("nonsurg" in t):
        return "non-surgical"
    if "non" in t:
        return "non-surgical"
    if "surg" in t:
        return "surgical"
    return ""


def _norm_type_choice(choice: str) -> str:
    c = _norm(choice)
    if "both" in c:
        return "both"
    if ("non" in c and "surg" in c) or ("non" in c):
        return "non-surgical"
    if "surg" in c:
        return "surgical"
    return "both"


def _to_proc_type(db_type: str) -> str:
    t = _norm(db_type)
    if ("non" in t and "surg" in t) or ("non" in t):
        return "Non-Surgical"
    if "surg" in t:
        return "Surgical"
    return "Not found in database."


def _db_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def _first_present(row: pd.Series, keys: List[str]) -> str:
    for k in keys:
        if k in row.index:
            val = _db_str(row.get(k, ""))
            if val:
                return val
    return ""


def _na_db(v: str) -> str:
    return v if v else "Not found in database."


# ---------------------------- data model ----------------------------

@dataclass
class RetrievedCandidate:
    row_index: int
    similarity: float

    procedure: str
    region: str
    sub_zone: str
    type: str

    short_description: str
    concerns: str
    techniques: str

    expected_results: str
    procedure_duration_hours: str
    downtime_days: str
    results_visible_timeline: str
    result_duration: str
    potential_side_effects: str
    anesthesia_type: str
    hospital_stay: str
    protocol_type: str
    session_frequency: str

    average_cost_min_eur: str
    average_cost_max_eur: str
    average_cost_min_chf: str
    average_cost_max_chf: str


# ---------------------------- app ----------------------------

class RAGTreatmentSearchApp:
    """
    HF-ready local structured RAG (DB-based details).

    DB: database.xlsx (NEW schema)
      - Uses sheet_name default: "Procedures"
      - Reads procedure details from DB columns (no web calls)

    API is kept compatible with your existing gradio_new_rag_app.py:
      RAGTreatmentSearchApp(excel_path=..., embeddings_cache_path=...)
    """

    def __init__(
        self,
        excel_path: str = "database.xlsx",
        sheet_name: str = "Procedures",
        embeddings_cache_path: str = "treatment_embeddings.pkl",
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm: Optional[LocalLLMClient] = None,
    ):
        try:
            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        except Exception:
            pass

        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.embeddings_cache_path = embeddings_cache_path

        self.df = self._load_db()
        self._normalize_columns()

        self.model = SentenceTransformer(embedding_model_name, device="cpu")
        self.embeddings, self.texts = self._load_or_build_embeddings()

        self.llm = llm or LocalLLMClient()

        # hard gate: avoid returning output when issue is empty
        self.min_issue_chars = int(os.getenv("MIN_ISSUE_CHARS", "5"))

        # mismatch sensitivity (tuned)
        self.local_issue_min_sim = float(os.getenv("LOCAL_ISSUE_MIN_SIM", "0.42"))
        self.global_issue_min_sim = float(os.getenv("GLOBAL_ISSUE_MIN_SIM", "0.52"))
        self.global_local_delta = float(os.getenv("GLOBAL_LOCAL_DELTA", "0.10"))

    # ---------------- DB ----------------

    def _load_db(self) -> pd.DataFrame:
        xl = pd.ExcelFile(self.excel_path)
        if self.sheet_name not in xl.sheet_names:
            raise ValueError(f"Sheet '{self.sheet_name}' not found. Found: {xl.sheet_names}")
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)

        # Drop rows with no procedure_title (incomplete/placeholder rows)
        before = len(df)
        df = df[df["procedure_title"].notna() & (df["procedure_title"].astype(str).str.strip() != "")]
        dropped_nan = before - len(df)

        # Deduplicate using all 4 identity fields:
        # procedure_title + main_zone + face_subzone + body_subzone
        # A row is only a duplicate if ALL 4 match exactly.
        # If even one differs (e.g. same procedure but different sub-zone), it is kept as unique.
        dedup_cols = [c for c in ["procedure_title", "main_zone", "face_subzone", "body_subzone"] if c in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        dropped_dupes = before - len(df)

        if dropped_nan or dropped_dupes:
            import logging
            logging.getLogger(__name__).info(
                f"DB cleanup: dropped {dropped_nan} rows with no procedure_title, "
                f"{dropped_dupes} exact duplicate rows (matched on {dedup_cols}). "
                f"Final: {len(df)} rows."
            )

        return df

    def _normalize_columns(self) -> None:
        """
        Supports the NEW schema you described.
        We also create UI-friendly aliases: Region, Sub-Zone, Procedure, Type.
        """
        # Required minimal new schema keys (based on your DB update)
        required_any = [
            "procedure_title",
            "main_zone",
            "treatment_type",
        ]
        missing_any = [c for c in required_any if c not in self.df.columns]
        if missing_any:
            raise ValueError(f"Database missing required columns: {missing_any}")

        # Build unified Region/Sub-Zone fields
        # Region -> main_zone
        self.df["Region"] = self.df["main_zone"].fillna("").astype(str).str.strip()

        # Sub-Zone: prefer face_subzone else body_subzone else any existing fallback
        if "face_subzone" in self.df.columns or "body_subzone" in self.df.columns:
            face = self.df["face_subzone"].fillna("").astype(str).str.strip() if "face_subzone" in self.df.columns else ""
            body = self.df["body_subzone"].fillna("").astype(str).str.strip() if "body_subzone" in self.df.columns else ""
            sub = face
            if isinstance(sub, str):
                # shouldn't happen, but keep safe
                sub = ""
            self.df["Sub-Zone"] = face
            mask_empty = self.df["Sub-Zone"].eq("") | self.df["Sub-Zone"].str.lower().eq("nan")
            if not isinstance(body, str):
                self.df.loc[mask_empty, "Sub-Zone"] = body.loc[mask_empty]
        else:
            # last fallback if DB already has something named Sub-Zone
            self.df["Sub-Zone"] = self.df.get("Sub-Zone", "").fillna("").astype(str).str.strip()

        # Procedure/Type
        self.df["Procedure"] = self.df["procedure_title"].fillna("").astype(str).str.strip()
        self.df["Type"] = self.df["treatment_type"].fillna("").astype(str).str.strip()

        # Normalize core columns
        for col in ["Type", "Region", "Sub-Zone", "Procedure"]:
            self.df[col] = self.df[col].astype(str).fillna("").str.strip()

        self.df["_region_norm"] = self.df["Region"].apply(_norm)
        self.df["_subzone_norm"] = self.df["Sub-Zone"].apply(_norm)
        self.df["_type_norm"] = self.df["Type"].apply(_norm_type_value)
        # Build searchable DB sub-zone vocabulary for fuzzy matching
        self._db_subzone_terms = self._build_db_subzone_terms()


    def get_regions(self) -> List[str]:
        regions = [r for r in self.df["Region"].dropna().unique().tolist() if str(r).strip() and str(r).strip().lower() != "nan"]
        return sorted(regions)

    def get_sub_zones(self, region: str) -> List[str]:
        r = _norm(region)
        sub = self.df[self.df["_region_norm"].eq(r)]["Sub-Zone"].dropna().unique().tolist()
        out = []
        for s in sub:
            ss = str(s).strip()
            if ss and ss.lower() != "nan":
                out.append(ss)
        return sorted(out)
    # ---------------- Sub-zone fuzzy matching ----------------

    def _split_db_subzone(self, s: str) -> List[str]:
        """
        DB often has composite strings like:
          "Cheeks, Cheekbones" or "Jawline / Jowls"
        Split into individual normalized tokens.
        """
        s = _norm(s)
        if not s:
            return []
        parts = re.split(r"[,\|/]|(?:\band\b)", s)
        parts = [_norm(p) for p in parts]
        return [p for p in parts if p]

    def _build_db_subzone_terms(self) -> List[str]:
        """
        Build a unique list of normalized sub-zone terms present in DB.
        Includes both full strings and split components.
        """
        terms = set()

        for raw in self.df["Sub-Zone"].fillna("").astype(str).tolist():
            raw = raw.strip()
            if not raw or raw.lower() == "nan":
                continue

            terms.add(_norm(raw))
            for p in self._split_db_subzone(raw):
                terms.add(p)

        return sorted([t for t in terms if t])

    def _resolve_db_subzones(self, requested_sub_zone: str, max_matches: int = 5) -> List[str]:
        """
        Generalized closest-match: map requested sub_zone (from hardcoded list)
        to the closest DB sub-zone term(s).

        Examples:
          "teartrough" -> ["tear trough"]
          "eyes"       -> returns closest DB terms (e.g. "eyelids", etc.) if present
        """
        q = _norm(requested_sub_zone)
        if not q:
            return []

        # 1) Exact match
        if q in self._db_subzone_terms:
            return [q]

        # 2) Substring-based matches (handles "teartrough" vs "tear trough")
        substring_hits = [t for t in self._db_subzone_terms if (q in t or t in q)]
        if substring_hits:
            substring_hits = sorted(substring_hits, key=len)
            return substring_hits[:max_matches]

        # 3) Fuzzy match
        return difflib.get_close_matches(q, self._db_subzone_terms, n=max_matches, cutoff=0.62)

    def _subzone_mask(self, sub_zone: str) -> "pd.Series":
        """
        Boolean mask of rows whose DB sub-zone matches the requested sub_zone
        using resolved closest DB terms.
        """
        keys = self._resolve_db_subzones(sub_zone)
        if not keys:
            return pd.Series([False] * len(self.df), index=self.df.index)

        m = pd.Series([False] * len(self.df), index=self.df.index)
        for k in keys:
            # Match whole-ish word boundaries within composite DB strings
            pat = r"(^|[\s,;/\-|])" + re.escape(k) + r"($|[\s,;/\-|])"
            m = m | self.df["_subzone_norm"].str.contains(pat, na=False, regex=True)

        return m

    def get_concerns_for_subzone(self, sub_zone: str) -> List[str]:
        """
        Get all unique concerns from database for a given sub-zone.
        Extracts concerns from the 'concerns' column.
        """
        # sz = _norm(sub_zone)
        # filtered_df = self.df[self.df["_subzone_norm"].eq(sz) | self.df["_subzone_norm"].str.contains(sz, na=False)]
        filtered_df = self.df[self._subzone_mask(sub_zone)]

        all_concerns = set()
        for _, row in filtered_df.iterrows():
            concerns_text = _first_present(row, ["concerns", "aesthetic_concerns", "Aesthetic Concerns"])
            if concerns_text and concerns_text != "Not found in database.":
                # Split by common separators: comma, semicolon, pipe, newline
                items = concerns_text.replace(";", ",").replace("|", ",").replace("\n", ",").split(",")
                for item in items:
                    cleaned = item.strip()
                    if cleaned and len(cleaned) > 2:  # Filter out very short items
                        all_concerns.add(cleaned)
        
        return sorted(list(all_concerns))

    def get_region_from_subzone(self, sub_zone: str) -> str:
        """
        Find the region that contains the given sub-zone.
        Returns empty string if not found.
        """
        # sz = _norm(sub_zone)
        # matches = self.df[self.df["_subzone_norm"].eq(sz) | self.df["_subzone_norm"].str.contains(sz, na=False)]
        matches = self.df[self._subzone_mask(sub_zone)]

        if len(matches) > 0:
            region = matches.iloc[0]["Region"]
            return str(region).strip()
        
        return ""

    def search_by_concerns(
        self,
        region: str,
        sub_zone: str,
        type_choice: str,
        concerns: List[str],
        retrieval_k: int = 12,
        final_k: int = 5,
    ) -> dict:
        """
        Search for procedures based on array of concerns.
        Returns only procedure names (not full details).
        
        Args:
            region: Selected region
            sub_zone: Selected sub-zone
            type_choice: Treatment preference (Surgical/Non-Surgical/Both)
            concerns: List of concern strings
            retrieval_k: Number of candidates to retrieve
            final_k: Number of final recommendations
            
        Returns:
            dict with:
            - mismatch: bool
            - notice: str (if mismatch)
            - recommended_procedures: List[str] (procedure names only)
            - suggested_region_subzones: List[dict] (if mismatch)
        """
        region = (region or "").strip()
        sub_zone = (sub_zone or "").strip()
        
        if not region or not sub_zone:
            return {
                "mismatch": False,
                "notice": "Region and sub-zone are required",
                "recommended_procedures": [],
                "suggested_region_subzones": []
            }
        
        if not concerns:
            return {
                "mismatch": False,
                "notice": "At least one concern is required",
                "recommended_procedures": [],
                "suggested_region_subzones": []
            }
        
        # Combine concerns into issue text for semantic search
        issue_text = ", ".join(concerns)
        
        # Use the existing recommend method
        result = self.recommend(
            region=region,
            sub_zone=sub_zone,
            type_choice=type_choice,
            issue_text=issue_text,
            retrieval_k=retrieval_k,
            final_k=final_k,
        )
        
        # Transform result to return only procedure names
        if result["status"] == "mismatch":
            return {
                "mismatch": True,
                "notice": "Your selected sub-zone does not match your concerns.",
                "recommended_procedures": [],
                "suggested_region_subzones": result.get("suggested_region_subzones", [])
            }
        
        # Extract procedure_title values — deduplicate by title + db_sub_zone
        # (same rule as _load_db: same title in same zone = duplicate, different zone = unique)
        procedure_titles = []
        seen_title_zone = set()
        for proc in result.get("recommended_procedures", []):
            if isinstance(proc, dict) and "procedure_title" in proc:
                title = proc["procedure_title"].strip()
                zone = proc.get("db_sub_zone", "").strip().lower()
                key = (title.lower(), zone)
                if key not in seen_title_zone:
                    seen_title_zone.add(key)
                    procedure_titles.append(title)

        # Sort final procedure titles alphabetically (A-Z)
        procedure_titles = sorted(procedure_titles, key=lambda t: t.lower())

        return {
            "mismatch": False,
            "notice": "",
            "recommended_procedures": procedure_titles,
            "suggested_region_subzones": []
        }

    # ---------------- Embeddings ----------------

    def _row_to_text(self, row: pd.Series) -> str:
        """
        Build semantic text from DB fields (for embeddings).
        Keep it compact but informative so issue-only similarity works.
        """
        proc = _db_str(row.get("procedure_title", ""))
        reg = _db_str(row.get("main_zone", ""))
        sub = _db_str(row.get("Sub-Zone", ""))
        typ = _db_str(row.get("treatment_type", ""))

        short_desc = _first_present(row, ["short_description", "procedure_description", "description"])
        concerns = _first_present(row, ["concerns", "aesthetic_concerns", "Aesthetic Concerns"])
        techniques = _first_present(row, ["techniques_brands_variants", "Technique / Technology / Brand", "techniques"])

        expected = _first_present(row, ["expected_results", "expected_result"])
        sidefx = _first_present(row, ["potential_side_effects", "side_effects", "risks"])

        parts = [
            f"Procedure: {proc}",
            f"Type: {typ}",
            f"Region: {reg}",
            f"Sub-Zone: {sub}",
            f"Description: {short_desc}",
            f"Concerns: {concerns}",
            f"Techniques: {techniques}",
            f"Expected: {expected}",
            f"Side effects: {sidefx}",
        ]
        return " | ".join([p for p in parts if p and not p.endswith(": ")])

    def _load_or_build_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, "rb") as f:
                    data = pickle.load(f)
                emb = np.array(data["embeddings"], dtype=np.float32)
                txt = list(data["texts"])
                if emb.shape[0] == len(self.df) and len(txt) == len(self.df):
                    return emb, txt
            except Exception:
                pass
        return self._build_embeddings()

    def _build_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        texts = [self._row_to_text(self.df.iloc[i]) for i in range(len(self.df))]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        payload = {"created_at": time.time(), "texts": texts, "embeddings": embeddings.tolist()}
        with open(self.embeddings_cache_path, "wb") as f:
            pickle.dump(payload, f)
        return embeddings, texts

    # ---------------- Retrieval ----------------

    # def _candidate_indices(self, region: str, sub_zone: str, type_norm: str) -> np.ndarray:
    #     r = _norm(region)
    #     sz = _norm(sub_zone)

    #     m = self.df["_region_norm"].eq(r)
    #     if sz:
    #         m = m & (self.df["_subzone_norm"].eq(sz) | self.df["_subzone_norm"].str.contains(sz, na=False))
    #     if type_norm in {"surgical", "non-surgical"}:
    #         m = m & self.df["_type_norm"].eq(type_norm)

    #     return np.where(m.values)[0]
    def _candidate_indices(self, region: str, sub_zone: str, type_norm: str) -> np.ndarray:
        r = _norm(region)

        m = self.df["_region_norm"].eq(r)
        if sub_zone and sub_zone.strip():
            m = m & self._subzone_mask(sub_zone)

        if type_norm in {"surgical", "non-surgical"}:
            m = m & self.df["_type_norm"].eq(type_norm)

        return np.where(m.values)[0]

    def _semantic_over(self, idxs: np.ndarray, query: str, top_k: int) -> List[RetrievedCandidate]:
        if idxs.size == 0:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        sims = cosine_similarity(q_emb, self.embeddings[idxs])[0]
        order = sims.argsort()[::-1]

        out: List[RetrievedCandidate] = []
        for pos in order[: max(top_k, 1) * 10]:
            row_index = int(idxs[pos])
            row = self.df.iloc[row_index]

            proc = _db_str(row.get("procedure_title", "")) or _db_str(row.get("Procedure", ""))
            reg = _db_str(row.get("main_zone", "")) or _db_str(row.get("Region", ""))
            sub = _db_str(row.get("Sub-Zone", "")) or _db_str(row.get("face_subzone", "")) or _db_str(row.get("body_subzone", ""))
            typ = _db_str(row.get("treatment_type", "")) or _db_str(row.get("Type", ""))

            out.append(
                RetrievedCandidate(
                    row_index=row_index,
                    similarity=float(sims[pos]),
                    procedure=_na_db(proc),
                    region=_na_db(reg),
                    sub_zone=_na_db(sub),
                    type=_na_db(typ),

                    short_description=_na_db(_first_present(row, ["short_description", "procedure_description", "description"])),
                    concerns=_na_db(_first_present(row, ["concerns", "aesthetic_concerns", "Aesthetic Concerns"])),
                    techniques=_na_db(_first_present(row, ["techniques_brands_variants", "Technique / Technology / Brand", "techniques"])),

                    expected_results=_na_db(_first_present(row, ["expected_results"])),
                    procedure_duration_hours=_na_db(_first_present(row, ["procedure_duration_hours"])),
                    downtime_days=_na_db(_first_present(row, ["downtime_days"])),
                    results_visible_timeline=_na_db(_first_present(row, ["results_visible_timeline"])),
                    result_duration=_na_db(_first_present(row, ["result_duration"])),
                    potential_side_effects=_na_db(_first_present(row, ["potential_side_effects"])),
                    anesthesia_type=_na_db(_first_present(row, ["anesthesia_type"])),
                    hospital_stay=_na_db(_first_present(row, ["hospital_stay"])),
                    protocol_type=_na_db(_first_present(row, ["protocol_type"])),
                    session_frequency=_na_db(_first_present(row, ["session_frequency"])),

                    average_cost_min_eur=_na_db(_first_present(row, ["average_cost_min_eur"])),
                    average_cost_max_eur=_na_db(_first_present(row, ["average_cost_max_eur"])),
                    average_cost_min_chf=_na_db(_first_present(row, ["average_cost_min_chf"])),
                    average_cost_max_chf=_na_db(_first_present(row, ["average_cost_max_chf"])),
                )
            )
            if len(out) >= top_k:
                break
        return out

    def _global_semantic(self, issue_text: str, top_k: int = 15) -> List[RetrievedCandidate]:
        if not issue_text.strip():
            return []
        q_emb = self.model.encode([issue_text], convert_to_numpy=True).astype(np.float32)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        order = sims.argsort()[::-1]

        out: List[RetrievedCandidate] = []
        for idx in order[: max(top_k, 1) * 20]:
            row = self.df.iloc[int(idx)]
            # Build minimal candidate (details not required for mismatch suggestion list)
            proc = _db_str(row.get("procedure_title", "")) or _db_str(row.get("Procedure", ""))
            reg = _db_str(row.get("main_zone", "")) or _db_str(row.get("Region", ""))
            sub = _db_str(row.get("Sub-Zone", "")) or _db_str(row.get("face_subzone", "")) or _db_str(row.get("body_subzone", ""))
            typ = _db_str(row.get("treatment_type", "")) or _db_str(row.get("Type", ""))

            out.append(
                RetrievedCandidate(
                    row_index=int(idx),
                    similarity=float(sims[idx]),
                    procedure=_na_db(proc),
                    region=_na_db(reg),
                    sub_zone=_na_db(sub),
                    type=_na_db(typ),

                    short_description="Not found in database.",
                    concerns="Not found in database.",
                    techniques="Not found in database.",

                    expected_results="Not found in database.",
                    procedure_duration_hours="Not found in database.",
                    downtime_days="Not found in database.",
                    results_visible_timeline="Not found in database.",
                    result_duration="Not found in database.",
                    potential_side_effects="Not found in database.",
                    anesthesia_type="Not found in database.",
                    hospital_stay="Not found in database.",
                    protocol_type="Not found in database.",
                    session_frequency="Not found in database.",

                    average_cost_min_eur="Not found in database.",
                    average_cost_max_eur="Not found in database.",
                    average_cost_min_chf="Not found in database.",
                    average_cost_max_chf="Not found in database.",
                )
            )
            if len(out) >= top_k:
                break
        return out

    def _local_issue_only_best_sim(self, region: str, sub_zone: str, type_choice: str, issue_text: str) -> float:
        """
        Compute issue-only similarity inside selected region/sub-zone to detect irrelevance.
        """
        issue_text = (issue_text or "").strip()
        if not issue_text:
            return 0.0

        t = _norm_type_choice(type_choice)
        if t == "both":
            idx_s = self._candidate_indices(region, sub_zone, "surgical")
            idx_n = self._candidate_indices(region, sub_zone, "non-surgical")
            idxs = np.unique(np.concatenate([idx_s, idx_n])) if (idx_s.size or idx_n.size) else np.array([], dtype=int)
        else:
            idxs = self._candidate_indices(region, sub_zone, t)

        if idxs.size == 0:
            # region only
            if t == "both":
                idx_s = self._candidate_indices(region, "", "surgical")
                idx_n = self._candidate_indices(region, "", "non-surgical")
                idxs = np.unique(np.concatenate([idx_s, idx_n])) if (idx_s.size or idx_n.size) else np.array([], dtype=int)
            else:
                idxs = self._candidate_indices(region, "", t)

        if idxs.size == 0:
            return 0.0

        q_emb = self.model.encode([issue_text], convert_to_numpy=True).astype(np.float32)
        sims = cosine_similarity(q_emb, self.embeddings[idxs])[0]
        return float(np.max(sims)) if sims.size else 0.0

    def semantic_search(
        self,
        region: str,
        sub_zone: str,
        type_choice: str,
        issue_text: str,
        top_k: int = 12,
    ) -> List[RetrievedCandidate]:
        type_norm = _norm_type_choice(type_choice)

        query = f"Region: {region} | Sub-Zone: {sub_zone} | Preference: {type_choice} | Issue: {issue_text}"

        if type_norm == "both":
            idx_s = self._candidate_indices(region, sub_zone, "surgical")
            idx_n = self._candidate_indices(region, sub_zone, "non-surgical")
            per = max(3, top_k // 2)
            res = self._semantic_over(idx_s, query, per) + self._semantic_over(idx_n, query, per)
            res.sort(key=lambda x: x.similarity, reverse=True)
            # Deduplicate: a candidate is unique only if procedure_title + sub_zone differ.
            # This mirrors the same 4-field rule used in _load_db — same title in same zone = duplicate.
            seen_rows = set()
            seen_title_zone = set()
            out = []
            for c in res:
                title_zone_key = (c.procedure.strip().lower(), c.sub_zone.strip().lower())
                if c.row_index in seen_rows or title_zone_key in seen_title_zone:
                    continue
                seen_rows.add(c.row_index)
                seen_title_zone.add(title_zone_key)
                out.append(c)
                if len(out) >= top_k:
                    break
            return out

        idx = self._candidate_indices(region, sub_zone, type_norm)
        if idx.size == 0:
            idx = self._candidate_indices(region, "", type_norm)
        return self._semantic_over(idx, query, top_k)

    # ---------------- LLM rerank ----------------

    def _llm_rerank(self, issue_text: str, candidates: List[RetrievedCandidate], top_k: int) -> List[RetrievedCandidate]:
        if not candidates:
            return []

        cand_block = "\n".join([f"- {c.procedure} (Type: {c.type}, Sub-Zone: {c.sub_zone})" for c in candidates])
        prompt = f"""
Pick the best {top_k} procedure names for:
"{issue_text}"

Candidates:
{cand_block}

Return ONLY a comma-separated list of procedure names (exactly as written).
""".strip()

        raw = (self.llm.generate(prompt, temperature=0.1, max_tokens=90) or "").strip()
        names = [n.strip() for n in raw.split(",") if n.strip()]
        if not names:
            return candidates[:top_k]

        lookup = {c.procedure.lower(): c for c in candidates}
        out: List[RetrievedCandidate] = []
        for n in names:
            c = lookup.get(n.lower())
            if c and c not in out:
                out.append(c)
            if len(out) >= top_k:
                break

        # fill remainder
        for c in candidates:
            if len(out) >= top_k:
                break
            if c not in out:
                out.append(c)

        return out

    # ---------------- Formatting (DB details) ----------------

    def _format_cost(self, mn: str, mx: str, unit: str) -> str:
        if mn == "Not found in database." and mx == "Not found in database.":
            return "Not found in database."
        if mn == "Not found in database." and mx != "Not found in database.":
            return f"Up to {mx} {unit}"
        if mx == "Not found in database." and mn != "Not found in database.":
            return f"From {mn} {unit}"
        return f"{mn} – {mx} {unit}"

    def _format_procedure_block(self, c: RetrievedCandidate, idx: int) -> str:
        proc_type = _to_proc_type(c.type)

        desc = c.short_description
        if desc == "Not found in database." and c.concerns != "Not found in database.":
            desc = c.concerns

        cost_eur = self._format_cost(c.average_cost_min_eur, c.average_cost_max_eur, "EUR")
        cost_chf = self._format_cost(c.average_cost_min_chf, c.average_cost_max_chf, "CHF")

        lines = [
            f"## {c.procedure}",
            f"- **Procedure type:** {proc_type}",
            f"- **Region / Sub-Zone (DB):** {c.region} / {c.sub_zone}",
            f"- **Description (DB):** {desc}",
            f"- **Concerns (DB):** {c.concerns}",
            f"- **Techniques / brands (DB):** {c.techniques}",
            f"- **Expected results (DB):** {c.expected_results}",
            f"- **Protocol type (DB):** {c.protocol_type}",
            f"- **Session frequency (DB):** {c.session_frequency}",
            f"- **Anesthesia (DB):** {c.anesthesia_type}",
            f"- **Hospital stay (DB):** {c.hospital_stay}",
            f"- **Procedure duration (hours) (DB):** {c.procedure_duration_hours}",
            f"- **Downtime (days) (DB):** {c.downtime_days}",
            f"- **Results visible timeline (DB):** {c.results_visible_timeline}",
            f"- **Result duration / longevity (DB):** {c.result_duration}",
            f"- **Potential side effects (DB):** {c.potential_side_effects}",
            f"- **Average cost (EUR) (DB):** {cost_eur}",
            f"- **Average cost (CHF) (DB):** {cost_chf}",
        ]
        return "\n".join(lines).strip()

    def _format_final_answer(self, best: List[RetrievedCandidate]) -> str:
        blocks = [self._format_procedure_block(c, i + 1) for i, c in enumerate(best)]
        blocks.append("\n---\n**Safety disclaimer:** This tool is for informational purposes only and is not medical advice. Please consult a licensed clinician.")
        return "\n\n".join(blocks).strip()

    # ---------------- Main API ----------------

    def _candidate_to_result_obj(self, c: RetrievedCandidate) -> Dict[str, object]:
        """Convert a RetrievedCandidate into a JSON-serializable object for API responses."""
        proc_type = _to_proc_type(c.type)

        desc = c.short_description
        if desc == "Not found in database." and c.concerns != "Not found in database.":
            desc = c.concerns

        cost_eur = self._format_cost(c.average_cost_min_eur, c.average_cost_max_eur, "EUR")
        cost_chf = self._format_cost(c.average_cost_min_chf, c.average_cost_max_chf, "CHF")

        return {
            "procedure_title": c.procedure,
            "procedure_type": proc_type,
            "db_region": c.region,
            "db_sub_zone": c.sub_zone,
            "db_description": desc,
            "db_concerns": c.concerns,
            "db_techniques_brands": c.techniques,
            "expected_results": c.expected_results,
            "protocol_type": c.protocol_type,
            "session_frequency": c.session_frequency,
            "anesthesia_type": c.anesthesia_type,
            "hospital_stay": c.hospital_stay,
            "procedure_duration_hours": c.procedure_duration_hours,
            "downtime_days": c.downtime_days,
            "results_visible_timeline": c.results_visible_timeline,
            "result_duration": c.result_duration,
            "potential_side_effects": c.potential_side_effects,
            "average_cost": {
                "eur": cost_eur,
                "chf": cost_chf,
            },
            "similarity": round(float(c.similarity), 4),
        }

    def recommend(
        self,
        region: str,
        sub_zone: str,
        type_choice: str,
        issue_text: str,
        retrieval_k: int = 12,
        final_k: int = 5,
    ) -> Dict[str, object]:
        """
        Structured API response for backend integration (DigitalOcean etc.).

        Returns:
          - status: blocked | mismatch | ok
          - message: human-readable note
          - suggested_region_subzones: only when mismatch
          - recommended_procedures: list[dict] only when ok
          - debug: internal signals
        """
        region = (region or "").strip()
        sub_zone = (sub_zone or "").strip()
        issue_text = (issue_text or "").strip()

        if not region or not sub_zone:
            return {
                "status": "blocked",
                "message": "Please provide both region and sub_zone.",
                "recommended_procedures": [],
                "suggested_region_subzones": [],
                "debug": {"blocked": True, "reason": "missing_region_or_subzone"},
            }

        if len(issue_text) < self.min_issue_chars:
            return {
                "status": "blocked",
                "message": f"Please provide a problem statement (minimum {self.min_issue_chars} characters).",
                "recommended_procedures": [],
                "suggested_region_subzones": [],
                "debug": {"blocked": True, "reason": "missing_issue_text"},
            }

        candidates = self.semantic_search(region, sub_zone, type_choice, issue_text, top_k=int(retrieval_k))
        if not candidates:
            return {
                "status": "blocked",
                "message": "No matching procedures found for the selected region/sub-zone and problem statement. Please revise inputs.",
                "recommended_procedures": [],
                "suggested_region_subzones": [],
                "debug": {"blocked": True, "reason": "no_candidates"},
            }

        # mismatch detection (same logic as answer())
        global_cands = self._global_semantic(issue_text, top_k=15)
        global_best = global_cands[0].similarity if global_cands else 0.0
        local_best = candidates[0].similarity if candidates else 0.0

        selected_region_norm = _norm(region)
        selected_sub_norm = _norm(sub_zone)

        selected_in_global = any(
            _norm(c.region) == selected_region_norm and (
                selected_sub_norm in _norm(c.sub_zone) or _norm(c.sub_zone) in selected_sub_norm
            )
            for c in global_cands[:10]
        )

        local_issue_best = self._local_issue_only_best_sim(region, sub_zone, type_choice, issue_text)

        mismatch_strict = (
            (local_issue_best > 0.0 and local_issue_best < self.local_issue_min_sim)
            and (global_best >= self.global_issue_min_sim)
            and (not selected_in_global)
        )

        mismatch_delta = (
            (global_best >= self.global_issue_min_sim)
            and ((global_best - local_best) >= self.global_local_delta)
            and (not selected_in_global)
        )

        if mismatch_strict or mismatch_delta:
            suggestions = []
            seen = set()
            for c in global_cands:
                key = (c.region, c.sub_zone)
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append({"region": c.region, "sub_zone": c.sub_zone})
                if len(suggestions) >= 8:
                    break

            return {
                "status": "mismatch",
                "message": "Selected region/sub-zone does not match the problem statement. Please choose a more appropriate region/sub-zone.",
                "recommended_procedures": [],
                "suggested_region_subzones": suggestions,
                "debug": {
                    "mismatch": True,
                    "global_best": round(float(global_best), 4),
                    "local_best": round(float(local_best), 4),
                    "local_issue_best": round(float(local_issue_best), 4),
                    "candidate_count": len(candidates),
                },
            }

        best = self._llm_rerank(issue_text, candidates, top_k=int(final_k))
        if len(best) < int(final_k):
            for c in candidates:
                if c not in best:
                    best.append(c)
                if len(best) >= int(final_k):
                    break
        best = best[: int(final_k)]

        # Sort final top-k alphabetically by procedure title (A-Z)
        best = sorted(best, key=lambda c: c.procedure.lower())

        results = [self._candidate_to_result_obj(c) for c in best]

        return {
            "status": "ok",
            "message": "Success",
            "recommended_procedures": results,
            "suggested_region_subzones": [],
            "debug": {
                "mismatch": False,
                "candidate_count": len(candidates),
                "final_count": len(best),
                "candidates": [
                    {
                        "procedure": c.procedure,
                        "similarity": round(float(c.similarity), 4),
                        "type": c.type,
                        "region": c.region,
                        "sub_zone": c.sub_zone,
                    }
                    for c in candidates[:20]
                ],
            },
        }



    def answer(
        self,
        region: str,
        sub_zone: str,
        type_choice: str,
        issue_text: str,
        retrieval_k: int = 12,
        final_k: int = 5,
    ) -> Dict[str, object]:

        region = (region or "").strip()
        sub_zone = (sub_zone or "").strip()
        issue_text = (issue_text or "").strip()

        # Hard gate: must provide issue text
        if not region or not sub_zone:
            return {
                "answer_md": "Please select **Region** and **Sub-Zone** before running the search.",
                "sources": [],
                "_debug": {"mismatch": False, "blocked": True, "reason": "missing_region_or_subzone"},
            }

        if len(issue_text) < self.min_issue_chars:
            return {
                "answer_md": f"Please describe your issue/problem (minimum {self.min_issue_chars} characters) to get recommendations.",
                "sources": [],
                "_debug": {"mismatch": False, "blocked": True, "reason": "missing_issue_text"},
            }

        candidates = self.semantic_search(region, sub_zone, type_choice, issue_text, top_k=int(retrieval_k))
        if not candidates:
            return {
                "answer_md": "No matching procedures found for your selected Region/Sub-Zone and issue. Please revise your inputs.",
                "sources": [],
                "_debug": {"mismatch": False, "candidate_count": 0, "final_count": 0},
            }

        # ---------- mismatch detection ----------
        global_cands = self._global_semantic(issue_text, top_k=15)
        global_best = global_cands[0].similarity if global_cands else 0.0
        local_best = candidates[0].similarity if candidates else 0.0

        selected_region_norm = _norm(region)
        selected_sub_norm = _norm(sub_zone)

        selected_in_global = any(
            _norm(c.region) == selected_region_norm and (
                selected_sub_norm in _norm(c.sub_zone) or _norm(c.sub_zone) in selected_sub_norm
            )
            for c in global_cands[:10]
        )

        local_issue_best = self._local_issue_only_best_sim(region, sub_zone, type_choice, issue_text)

        mismatch_strict = (
            (local_issue_best > 0.0 and local_issue_best < self.local_issue_min_sim)
            and (global_best >= self.global_issue_min_sim)
            and (not selected_in_global)
        )

        mismatch_delta = (
            (global_best >= self.global_issue_min_sim)
            and ((global_best - local_best) >= self.global_local_delta)
            and (not selected_in_global)
        )

        if mismatch_strict or mismatch_delta:
            # suggest correct region/sub-zones based on issue text
            suggestions = []
            seen = set()
            for c in global_cands:
                key = (c.region, c.sub_zone)
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(key)
                if len(suggestions) >= 8:
                    break

            sug_lines = "\n".join([f"- **{r} → {sz}**" for (r, sz) in suggestions]) or "- (No suggestions found)"

            answer_md = f"""## Notice (Mismatch Detected)
Sorry for inconvenience. Your selected **Region/Sub-Zone** does not match your problem description.

### Your input
- **Selected Region:** {region}
- **Selected Sub-Zone:** {sub_zone}
- **Issue text:** {issue_text}

### Suggested Region/Sub-Zones (from your DB)
{sug_lines}

### Next step
Please select one of the suggested **Region/Sub-Zones** and run the search again.
""".strip()

            return {
                "answer_md": answer_md,
                "sources": [],
                "_debug": {
                    "mismatch": True,
                    "global_best": round(global_best, 4),
                    "local_best": round(local_best, 4),
                    "local_issue_best": round(local_issue_best, 4),
                    "candidate_count": len(candidates),
                },
            }
        # ---------------------------------------

        best = self._llm_rerank(issue_text, candidates, top_k=int(final_k))

        # Ensure exactly final_k if possible
        if len(best) < int(final_k):
            for c in candidates:
                if c not in best:
                    best.append(c)
                if len(best) >= int(final_k):
                    break
        best = best[: int(final_k)]

        # Sort final top-k alphabetically by procedure title (A-Z)
        best = sorted(best, key=lambda c: c.procedure.lower())

        answer_md = self._format_final_answer(best)

        return {
            "answer_md": answer_md,
            "sources": [],  # DB-only mode
            "_debug": {
                "mismatch": False,
                "candidate_count": len(candidates),
                "final_count": len(best),
                "candidates": [
                    {
                        "procedure": c.procedure,
                        "similarity": round(c.similarity, 4),
                        "type": c.type,
                        "region": c.region,
                        "sub_zone": c.sub_zone,
                    }
                    for c in candidates[:20]
                ],
            },
        }