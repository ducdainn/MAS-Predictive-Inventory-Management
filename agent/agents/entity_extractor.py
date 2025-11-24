"""
EntityExtractor: identifies branches/products/regions mentioned in user queries.
"""

import json
import unicodedata
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.manager.database_manager import DatabaseManager
from agent.core.llm_provider import LLMProvider


class EntityExtractor:
    """
    Extracts entities from user questions: branch names, product names, regions.
    IMPROVEMENT: Enables context-aware optimization (e.g., "chi nhánh đà nẵng")
    """

    def __init__(self, llm_provider: LLMProvider, db_manager: DatabaseManager):
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.db = db_manager
        self._load_entity_cache()

    def _load_entity_cache(self):
        """Load all branch and product names for fuzzy matching."""
        try:
            branches_df = self.db.execute_query("SELECT branch_code, branch_name, region FROM branch")
            self.branches = branches_df.to_dict('records')

            products_df = self.db.execute_query("SELECT product_code, product_name FROM product")
            self.products = products_df.to_dict('records')

            print(f"✅ Loaded {len(self.branches)} branches and {len(self.products)} products for entity matching")
        except Exception as e:
            print(f"⚠️ Could not load entity cache: {e}")
            self.branches = []
            self.products = []

    def extract_entities(self, question: str) -> Dict[str, Any]:
        """Extract entities from user question using LLM + fuzzy matching."""
        print("🔍 Extracting entities from question...")

        branch_names = [b['branch_name'] for b in self.branches[:20]]
        regions = list(set([b['region'] for b in self.branches]))

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
4. Scope: "specific" if question mentions specific branches/products, "all" if general

Rules:
- Extract partial matches (e.g., "đà nẵng" matches "Chi nhánh Đà Nẵng 1")
- Case insensitive
- Return empty lists if nothing mentioned
- Be lenient with Vietnamese diacritics

Return ONLY valid JSON, no explanations."""

    def _match_branches(self, mentioned_names: List[str]) -> List[int]:
        """Fuzzy match mentioned branch names to branch codes."""
        if not mentioned_names:
            return []

        matched_codes = []

        for mentioned in mentioned_names:
            mentioned_lower = mentioned.lower().strip()
            mentioned_normalized = self._normalize_vietnamese(mentioned_lower)

            for branch in self.branches:
                branch_name_lower = branch['branch_name'].lower()
                branch_name_normalized = self._normalize_vietnamese(branch_name_lower)

                if (mentioned_normalized in branch_name_normalized or
                        mentioned_lower in branch_name_lower):
                    matched_codes.append(branch['branch_code'])
                    print(f"   ✓ Matched '{mentioned}' → {branch['branch_name']} (code: {branch['branch_code']})")

        return list(set(matched_codes))

    def _match_products(self, mentioned_names: List[str]) -> List[str]:
        """Fuzzy match mentioned product names to product codes."""
        if not mentioned_names:
            return []

        matched_codes = []

        for mentioned in mentioned_names:
            mentioned_lower = mentioned.lower().strip()
            mentioned_normalized = self._normalize_vietnamese(mentioned_lower)

            for product in self.products:
                product_name_lower = product['product_name'].lower()
                product_name_normalized = self._normalize_vietnamese(product_name_lower)

                if (mentioned_normalized in product_name_normalized or
                        mentioned_lower in product_name_lower):
                    matched_codes.append(product['product_code'])
                    print(f"   ✓ Matched '{mentioned}' → {product['product_name'][:50]}...")
                    break

        return matched_codes

    def _normalize_vietnamese(self, text: str) -> str:
        """Remove Vietnamese accents for better matching."""
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')

    def _fallback_extraction(self, question: str) -> Dict[str, Any]:
        """Simple keyword-based fallback extraction."""
        question_lower = question.lower()
        question_normalized = self._normalize_vietnamese(question_lower)
        
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
        
        # Extract location keywords from question (2-3 word phrases)
        # Skip common words like "chi", "nhánh", "của", "tồn", "kho"
        skip_words = {'chi', 'nhánh', 'của', 'tồn', 'kho', 'tối', 'ưu', 'hóa', 'và', 'các', 'cho', 'với'}
        question_words = [w for w in question_lower.split() if w not in skip_words and len(w) > 2]
        
        # Build location phrases (2-3 consecutive words)
        location_phrases = []
        for i in range(len(question_words) - 1):
            # 2-word phrase
            phrase_2 = f"{question_words[i]} {question_words[i+1]}"
            location_phrases.append(phrase_2)
            # 3-word phrase if available
            if i + 2 < len(question_words):
                phrase_3 = f"{question_words[i]} {question_words[i+1]} {question_words[i+2]}"
                location_phrases.append(phrase_3)
        
        # Also add single important words (longer than 4 chars)
        important_words = [w for w in question_words if len(w) > 4]
        location_phrases.extend(important_words)
        
        # Normalize phrases for matching
        location_phrases_normalized = [self._normalize_vietnamese(p) for p in location_phrases]
        
        # Match branches using location phrases
        for branch in self.branches:
            branch_name_lower = branch['branch_name'].lower()
            branch_name_normalized = self._normalize_vietnamese(branch_name_lower)
            
            # Check if any location phrase appears in branch name
            matched = False
            for phrase, phrase_norm in zip(location_phrases, location_phrases_normalized):
                if (phrase in branch_name_lower or 
                    phrase_norm in branch_name_normalized):
                    entities['branch_codes'].append(branch['branch_code'])
                    entities['branch_names'].append(branch['branch_name'])
                    entities['scope'] = 'specific'
                    print(f"   ✓ Fallback matched: {branch['branch_name']} (phrase: '{phrase}')")
                    matched = True
                    break
            
            # If no phrase match, try matching important single words (only if they're long enough)
            if not matched and important_words:
                for word in important_words:
                    word_norm = self._normalize_vietnamese(word)
                    # Only match if word appears as a significant part of branch name
                    if (word in branch_name_lower or word_norm in branch_name_normalized):
                        # Additional check: word should not be too generic
                        if len(word) > 4:  # Only longer words to avoid false matches
                            entities['branch_codes'].append(branch['branch_code'])
                            entities['branch_names'].append(branch['branch_name'])
                            entities['scope'] = 'specific'
                            print(f"   ✓ Fallback matched: {branch['branch_name']} (word: '{word}')")
                            break
        
        entities['branch_codes'] = list(set(entities['branch_codes']))
        entities['branch_names'] = list(set(entities['branch_names']))
        
        return entities



