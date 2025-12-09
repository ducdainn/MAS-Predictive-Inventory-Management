"""
EntityExtractor: identifies branches/products/regions mentioned in user queries.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.manager.database_manager import DatabaseManager
from agent.core.llm_provider import LLMProvider


class EntityExtractor:
    """
    Extracts entities from user questions: branch names, product names, regions.
    IMPROVEMENT: Enables context-aware optimization (e.g., "chi nhánh đà nẵng")
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        db_manager: DatabaseManager,
    ):
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.db = db_manager
        self.branch_cache: List[Dict[str, Any]] = []
        self.product_cache: List[Dict[str, Any]] = []
        self.regions: List[str] = []
        self._load_entity_cache()

    def extract_entities(self, question: str) -> Dict[str, Any]:
        """Extract entities from user question using LLM + fuzzy matching."""
        print("🔍 Extracting entities from question...")

        # Check if this is a "top N" ranking query - these are general analysis, not specific entities
        question_lower = question.lower()
        is_top_n_query = bool(re.search(r'\btop\s+\d+|top\s*10|top\s*5|top\s*20|hàng đầu|cao nhất', question_lower, re.IGNORECASE))
        
        if is_top_n_query:
            print("   ℹ️  Detected 'top N' ranking query - skipping specific entity extraction")
            return {
                'branch_names': [],
                'branch_codes': [],
                'product_names': [],
                'product_codes': [],
                'regions': [],
                'scope': 'all'
            }

        branch_names = self._get_branch_samples()
        regions = self.regions

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_extraction_prompt()),
            ("human", """
Question: {question}

Available branches (sample): {branch_names}
Available regions: {regions}

Extract entities as JSON:
{{
    "branch_names": ["exact or partial branch names mentioned"],
    "product_names": ["product names mentioned"],
    "regions": ["regions mentioned"],
    "scope": "specific" or "all"
}}
""")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "question": question,
                "branch_names": ", ".join(branch_names),
                "regions": ", ".join(regions)
            })

            entities = json.loads(result)
            entities['branch_codes'] = self._match_branches(entities.get('branch_names', []))
            entities['product_codes'] = self._match_products(entities.get('product_names', []))

            print(f"✅ Extracted: {len(entities.get('branch_codes', []))} branches, "
                  f"{len(entities.get('product_codes', []))} products")

            # SAFETY NET: nếu không tìm được chi nhánh nhưng câu hỏi nhắc tới "chi nhánh"
            # CHỈ search nếu KHÔNG phải "top N" query (đã xử lý ở trên)
            if not entities.get('branch_codes'):
                q_lower = question.lower()
                if ('chi nhánh' in q_lower or 'chi nhanh' in q_lower) and not is_top_n_query:
                    print("⚠️ No branch_codes from LLM, using vector search for branches...")
                    search_terms = self._extract_branch_keywords(question) or [question]
                    for term in search_terms:
                        db_candidates = self._search_branch_cache(term)
                        if db_candidates:
                            entities['branch_codes'] = [c['branch_code'] for c in db_candidates]
                            entities['branch_names'] = [c['branch_name'] for c in db_candidates]
                            entities['regions'] = entities.get('regions', [])
                            break

            return entities

        except Exception as e:
            print(f"⚠️ Entity extraction failed: {e}, using fallback")
            return self._fallback_extraction(question)

    def _get_extraction_prompt(self) -> str:
        return """You are an entity extractor for inventory management questions.

Extract:
1. Branch names (e.g., "đà nẵng", "hà nội", "chi nhánh 1")
2. Product names (e.g., "gạch 30x60", "sơn nước")
3. Regions (e.g., "miền trung", "tây nguyên", "đông nam bộ", "tây nam bộ", "hồ chí minh")
4. Scope: "specific" if question mentions specific branches/products/regions, "all" if general analysis

CRITICAL RULES:
- **"Top N" ranking queries** (e.g., "top 10", "top 5", "hàng đầu", "cao nhất") are GENERAL analysis queries
  → Do NOT extract specific branch/product names from these queries!
  → "Top 10 chi nhánh" means "rank ALL branches", not extract 10 specific branches
  → Return empty branch_names/product_names for "top N" queries
- If question says "theo vùng miền", "theo vung mien", "tất cả vùng", "tat ca vung" → scope="all" and regions=[]
- "theo vùng miền" means "by all regions" (grouping dimension), NOT a specific region filter
- Do NOT extract "miền trung" from "theo vùng miền" - these are different!
- Extract regions ONLY if user mentions a SPECIFIC region (e.g., "miền trung", "tây nguyên")
- Extract partial matches (e.g., "đà nẵng" matches "Chi nhánh Đà Nẵng 1")
- Case insensitive
- Return empty lists if nothing mentioned
- Be lenient with Vietnamese diacritics

Examples:
- "Top 10 chi nhánh có doanh thu cao nhất" → scope="all", branch_names=[], product_names=[] (ranking query)
- "Top 5 sản phẩm bán chạy" → scope="all", branch_names=[], product_names=[] (ranking query)
- "Phân tích doanh số theo vùng miền" → scope="all", regions=[]
- "Phân tích doanh số miền trung" → scope="specific", regions=["miền trung"]
- "Doanh số tất cả vùng" → scope="all", regions=[]
- "Chi nhánh Đà Nẵng" → scope="specific", branch_names=["đà nẵng"] (specific entity)

Return ONLY valid JSON, no explanations."""

    def _match_branches(self, mentioned_names: List[str]) -> List[int]:
        """Fuzzy match mentioned branch names to branch codes."""
        if not mentioned_names:
            return []

        matched_codes = []
        for mentioned in mentioned_names:
            query = mentioned.strip()
            if not query:
                continue
            results = self._fuzzy_search_branches(query)
            for result in results:
                matched_codes.append(result['branch_code'])
                print(f"   ✓ Matched '{mentioned}' → {result['branch_name']} (code: {result['branch_code']}, score={result['score']:.2f})")

        # Preserve order while removing duplicates
        return list(dict.fromkeys(matched_codes))

    def _match_products(self, mentioned_names: List[str]) -> List[str]:
        """Fuzzy match mentioned product names to product codes."""
        if not mentioned_names:
            return []

        matched_codes = []
        for mentioned in mentioned_names:
            query = mentioned.strip()
            if not query:
                continue
            results = self._fuzzy_search_products(query)
            if results:
                best = results[0]
                matched_codes.append(best['product_code'])
                print(f"   ✓ Matched '{mentioned}' → {best['product_name'][:50]} (code: {best['product_code']}, score={best['score']:.2f})")

        return list(dict.fromkeys(matched_codes))

    def _fallback_extraction(self, question: str) -> Dict[str, Any]:
        """Simple keyword-based fallback extraction."""
        question_lower = question.lower()
        
        entities = {
            'branch_names': [],
            'branch_codes': [],
            'product_names': [],
            'product_codes': [],
            'regions': [],
            'scope': 'all'
        }
        
        # Check for regions first
        if 'miền bắc' in question_lower or 'mien bac' in question_lower:
            entities['regions'].append('MIỀN BẮC')
            entities['scope'] = 'specific'
        if 'miền trung' in question_lower or 'mien trung' in question_lower:
            entities['regions'].append('MIỀN TRUNG')
            entities['scope'] = 'specific'
        if 'miền nam' in question_lower or 'mien nam' in question_lower:
            entities['regions'].append('MIỀN NAM')
            entities['scope'] = 'specific'
        
        # Use cache search as emergency fallback
        branch_candidates = self._search_branch_cache(question)
        if branch_candidates:
            entities['branch_codes'] = [c['branch_code'] for c in branch_candidates]
            entities['branch_names'] = [c['branch_name'] for c in branch_candidates]
            entities['scope'] = 'specific'

        product_candidates = self._fuzzy_search_products(question, top_k=5, min_ratio=0.5)
        if product_candidates:
            entities['product_codes'] = [c['product_code'] for c in product_candidates]
            entities['product_names'] = [c['product_name'] for c in product_candidates]
        
        entities['branch_codes'] = list(set(entities['branch_codes']))
        entities['branch_names'] = list(set(entities['branch_names']))
        
        return entities

    def _extract_branch_keywords(self, question: str) -> List[str]:
        """
        Heuristic: capture phrases after 'chi nhánh/chi nhanh' to query vector store.
        Keeps diacritics by slicing the original string using lowercase indices.
        """
        lowered = question.lower()
        keywords: List[str] = []
        for marker in ('chi nhánh', 'chi nhanh'):
            start = lowered.find(marker)
            if start == -1:
                continue
            orig_start = start + len(marker)
            snippet = question[orig_start:].strip()
            if not snippet:
                continue
            # Stop at first punctuation or keyword that usually ends the phrase
            snippet = re.split(r"[,.?;!]|(?:\b(tháng|quý|năm|của|ở|tại|trong)\b)", snippet, maxsplit=1, flags=re.IGNORECASE)[0]
            words = snippet.strip().split()
            if not words:
                continue
            # Use up to first 5 words to keep the branch label concise
            candidate = " ".join(words[:5]).strip()
            if candidate:
                keywords.append(candidate)
        return keywords

    def _load_entity_cache(self):
        """Load entire branch/product tables into memory for fast lookup."""
        branch_df = self.db.execute_query("SELECT branch_code, branch_name, region FROM branch ORDER BY branch_name", source="EntityExtractor._load_entity_cache")
        product_df = self.db.execute_query("SELECT product_code, product_name FROM product ORDER BY product_name", source="EntityExtractor._load_entity_cache")

        self.branch_cache = branch_df.to_dict("records")
        self.product_cache = product_df.to_dict("records")
        self.regions = sorted({row['region'] for row in self.branch_cache if row.get('region')})

        print(f"🔁 Loaded {len(self.branch_cache)} branches & {len(self.product_cache)} products into cache")

    def _get_branch_samples(self, limit: int = 10) -> List[str]:
        return [row['branch_name'] for row in self.branch_cache[:limit]]

    def _fuzzy_search_branches(self, query: str, min_ratio: float = 0.45, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._fuzzy_search(query, self.branch_cache, key='branch_name', value_keys=['branch_code', 'branch_name', 'region'], min_ratio=min_ratio, top_k=top_k)

    def _fuzzy_search_products(self, query: str, min_ratio: float = 0.45, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._fuzzy_search(query, self.product_cache, key='product_name', value_keys=['product_code', 'product_name'], min_ratio=min_ratio, top_k=top_k)

    def _fuzzy_search(self, query: str, dataset: List[Dict[str, Any]], key: str, value_keys: List[str], min_ratio: float, top_k: int) -> List[Dict[str, Any]]:
        query_norm = query.strip().lower()
        if not query_norm:
            return []
        scored = []
        for row in dataset:
            value = row.get(key)
            if not value:
                continue
            ratio = SequenceMatcher(None, query_norm, value.lower()).ratio()
            if ratio >= min_ratio:
                result = {k: row.get(k) for k in value_keys}
                result['score'] = ratio
                scored.append(result)
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    def _search_branch_cache(self, text: str) -> List[Dict[str, Any]]:
        """Search branch cache using keyword containment (for fallback)."""
        text_norm = text.lower().strip()
        matches = []
        for row in self.branch_cache:
            name = row.get('branch_name', '').lower()
            if text_norm in name:
                matches.append({'branch_code': row['branch_code'], 'branch_name': row['branch_name'], 'region': row.get('region'), 'score': 1.0})
        return matches[:5]



