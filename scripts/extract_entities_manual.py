#!/usr/bin/env python3
"""
Manual extractor script: call Gemini via init_genai_client() directly (no LangChain)
Usage:
    python scripts/extract_entities_manual.py "Your text here"
Or:
    python scripts/extract_entities_manual.py --file path/to/file.txt

Temperature control:
    --temperature 0.0-2.0 (default: 0.3)
    Lower values (0.0-0.3): More focused, consistent answers
    Higher values (0.7-2.0): More creative, varied answers

This will print raw LLM output and a parsed list of entities/relationships.
"""
import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz

from src.llm.api import init_genai_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ANSI color codes for terminal
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"


class EntityMatch(NamedTuple):
    """Structured result for fuzzy matching"""
    query_entity: str
    match_id: str
    match_name: str
    confidence: float


class LLMEntityResult(NamedTuple):
    """Combined result from LLM entity processing"""
    identified_entity: Optional[str]
    extracted_entities: List[str]
    confidence: float


@dataclass
class EntityCache:
    """Cache for Neo4j entities to avoid duplicate queries"""
    node_mapping: Dict[str, str]  # node_id -> name
    known_entities: set
    entity_names: List[str]  # for fuzzy matching
    last_updated: datetime
    
    def is_valid(self, ttl_minutes: int = 60) -> bool:
        """Check if cache is still valid"""
        age_minutes = (datetime.now() - self.last_updated).total_seconds() / 60
        return age_minutes < ttl_minutes


class EntityCacheManager:
    """Manages caching of Neo4j entities to prevent duplicate queries"""
    
    def __init__(self):
        self._cache: Optional[EntityCache] = None
        self.logger = logging.getLogger(__name__)
    
    def get_entities(self, driver, force_refresh: bool = False) -> EntityCache:
        """Get cached entities or load from Neo4j if cache is invalid"""
        if not force_refresh and self._cache and self._cache.is_valid():
            self.logger.debug("Using cached entities")
            return self._cache
        
        self.logger.info("Loading entities from Neo4j...")
        try:
            # Single query to get both node mapping and entity set
            db = os.getenv("NEO4J_DATABASE", "neo4j")
            with driver.session(database=db) as session:
                result = session.run("MATCH (n:Entity) RETURN id(n) AS node_id, n.name AS name")
                records = list(result)
                
            node_mapping = {str(record["node_id"]): record["name"] for record in records}
            known_entities = {record["name"] for record in records}
            entity_names = list(known_entities)
            
            self._cache = EntityCache(
                node_mapping=node_mapping,
                known_entities=known_entities, 
                entity_names=entity_names,
                last_updated=datetime.now()
            )
            
            self.logger.info(f"Loaded and cached {len(known_entities)} entities from Neo4j")
            return self._cache
            
        except Exception as e:
            self.logger.error(f"Error loading entities from Neo4j: {e}")
            # Return empty cache if error
            self._cache = EntityCache(
                node_mapping={},
                known_entities=set(),
                entity_names=[],
                last_updated=datetime.now()
            )
            return self._cache
    
    def invalidate_cache(self):
        """Force cache refresh on next request"""
        self._cache = None
        self.logger.info("Entity cache invalidated")


class FuzzyMatchingService:
    """Unified fuzzy matching service for both chat and interactive modes"""
    
    def __init__(self, entity_cache: EntityCache):
        self.entity_cache = entity_cache
        self.logger = logging.getLogger(__name__)
    
    def find_matches(self, entities: List[str], confidence_threshold: float = 50.0) -> List[EntityMatch]:
        """Find fuzzy matches for entities with confidence scoring"""
        if not entities or not self.entity_cache.entity_names:
            return []
        
        matches = []
        for entity in entities:
            if not entity.strip():
                continue
                
            try:
                # Use ratio scorer with case-insensitive matching for better entity matching
                entity_lower = entity.lower()
                entity_names_lower = [name.lower() for name in self.entity_cache.entity_names]
                
                match = process.extractOne(entity_lower, entity_names_lower, scorer=fuzz.ratio)
                if match and match[1] >= confidence_threshold:
                    closest_match_lower, score, index = match
                    # Get the original case version
                    closest_match = self.entity_cache.entity_names[index]
                    
                    # Find node_id for this match
                    match_id = None
                    for node_id, name in self.entity_cache.node_mapping.items():
                        if name == closest_match:
                            match_id = node_id
                            break
                    
                    if match_id:
                        matches.append(EntityMatch(
                            query_entity=entity,
                            match_id=match_id,
                            match_name=closest_match,
                            confidence=float(score)
                        ))
                        
            except Exception as e:
                self.logger.warning(f"Error matching entity '{entity}': {e}")
                continue
        
        return matches
    
    def enhance_query(self, query: str, entities: List[str], confidence_threshold: float = 70.0) -> Tuple[str, List[EntityMatch]]:
        """Enhance query by replacing entities with best fuzzy matches"""
        matches = self.find_matches(entities, confidence_threshold=50.0)  # Lower threshold for finding
        
        if not matches:
            return query, []
        
        # Only use high-confidence matches for replacement
        good_matches = [m for m in matches if m.confidence >= confidence_threshold]
        
        enhanced_query = query
        for match in good_matches:
            enhanced_query = enhanced_query.replace(match.query_entity, match.match_name, 1)
        
        return enhanced_query, matches


class QueryProcessor:
    """Optimized query processor that combines LLM calls and reduces redundancy"""
    
    def __init__(self, client, entity_cache: EntityCache):
        self.client = client
        self.entity_cache = entity_cache
        self.logger = logging.getLogger(__name__)
    
    def process_entity_identification(self, 
                                    question: str, 
                                    current_active_entity: Optional[str] = None,
                                    conversation_history: str = "",
                                    model: str = "gemini-2.5-flash-lite") -> LLMEntityResult:
        """Combined entity identification and extraction in single LLM call"""
        try:
            # First try LLM entity identification
            identified_entity = self._llm_identify_active_entity(
                question=question,
                current_active_entity=current_active_entity,
                conversation_history=conversation_history,
                model=model
            )
            
            # If LLM found a clear entity with high confidence, we're done
            if identified_entity and self._is_high_confidence_entity(identified_entity):
                return LLMEntityResult(
                    identified_entity=identified_entity,
                    extracted_entities=[identified_entity],
                    confidence=0.9
                )
            
            # Otherwise, do comprehensive entity extraction
            extracted_entities = self._extract_entities_from_query(question, model)
            
            # Combine all potential entities for fuzzy matching
            all_entities = set(extracted_entities) if extracted_entities else set()
            
            # Include identified entity if it exists (even if not in DB) for fuzzy matching
            if identified_entity:
                all_entities.add(identified_entity)
            
            # Add regex-based potential entities
            potential_entities = re.findall(r'\b[A-Z][\w\-\(\)]+\b', question)
            all_entities.update(potential_entities)
            
            # Convert back to list
            final_entities = list(all_entities)
            
            return LLMEntityResult(
                identified_entity=identified_entity,
                extracted_entities=final_entities,
                confidence=0.7 if identified_entity else (0.6 if extracted_entities else 0.4)
            )
            
        except Exception as e:
            self.logger.error(f"Error in entity processing: {e}")
            return LLMEntityResult(
                identified_entity=None,
                extracted_entities=[],
                confidence=0.0
            )
    
    def _llm_identify_active_entity(self, question: str, current_active_entity: Optional[str], 
                                   conversation_history: str, model: str) -> Optional[str]:
        """LLM-based entity identification"""
        try:
            context = f"Current active entity: {current_active_entity}" if current_active_entity else "No current active entity"
            
            if conversation_history.strip():
                context += f"\n\nRecent conversation:\n{conversation_history.strip()}"
            
            # Provide sample entities for context
            sample_entities = list(self.entity_cache.known_entities)[:10] if self.entity_cache.known_entities else []
            entity_examples = ", ".join(sample_entities) if sample_entities else "No entities available"
            
            prompt = f"""Analyze this Vietnamese question and identify the PRIMARY entity being discussed.

Question: {question}

Context: {context}

Available entity examples: {entity_examples}

Instructions:
- Return ONLY the entity name if you can identify one clearly
- Return 'NONE' if the question is general or doesn't focus on a specific entity
- Handle pronouns (nó, này, đó, thiết bị này) by using the current active entity context
- For short questions without explicit entity, consider if they refer to the current active entity
- Prefer exact matches from available entities
- Consider the current context and conversation history

Entity (or NONE):"""
            
            response = call_gemini(self.client, prompt, model=model, temperature=0.1)
            result = response.strip() if response else ""
            
            if result and result != "NONE" and not any(word in result.lower() for word in ['none', 'không', 'general']):
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in LLM entity identification: {e}")
            return None
    
    def _extract_entities_from_query(self, question: str, model: str) -> List[str]:
        """Extract entities from query using LLM"""
        try:
            prompt = f"Extract entity names from this Vietnamese question: {question}\n\nReturn only entity names, one per line:"
            response = call_gemini(self.client, prompt, model=model, temperature=0.1)
            
            if response:
                entities, _ = parse_llm_output(response)
                return [e.strip() for e in entities if e.strip()]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def _is_high_confidence_entity(self, entity: str) -> bool:
        """Check if entity is high-confidence (exists in known entities)"""
        return entity in self.entity_cache.known_entities


# History persistence
HISTORY_FILE = "chatbot_history.json"


def load_history_from_file() -> Dict[str, Any]:
    """Load conversation history from JSON file with data validation"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Validate and repair data structure
            validated_data = validate_history_data(data)
            return validated_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in history file: {e}")
            # Backup corrupted file
            backup_file = f"{HISTORY_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.rename(HISTORY_FILE, backup_file)
                logger.info(f"Corrupted history backed up to {backup_file}")
            except:
                pass
            return create_empty_history()
            
        except Exception as e:
            logger.warning(f"Could not load history file: {e}")
            return create_empty_history()
    return create_empty_history()


def validate_history_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and repair history data structure"""
    try:
        # Ensure required keys exist
        if not isinstance(data, dict):
            return create_empty_history()
            
        # Validate basic structure
        required_keys = ["session_id", "created_at", "last_updated", "conversation_count", "history", "extracted_entities", "metadata"]
        for key in required_keys:
            if key not in data:
                if key == "history":
                    data[key] = []
                elif key == "extracted_entities":
                    data[key] = []
                elif key == "metadata":
                    data[key] = {}
                else:
                    data[key] = create_empty_history()[key]
        
        # Validate history list
        if not isinstance(data["history"], list):
            data["history"] = []
        else:
            # Clean up invalid conversations
            valid_conversations = []
            for conv in data["history"]:
                if isinstance(conv, dict) and "question" in conv and "answer" in conv:
                    # Ensure required fields
                    conv.setdefault("timestamp", datetime.now().isoformat())
                    conv.setdefault("entities_found", [])
                    conv.setdefault("question_length", len(str(conv.get("question", ""))))
                    conv.setdefault("answer_length", len(str(conv.get("answer", ""))))
                    
                    # Ensure entities_found is a list
                    if not isinstance(conv["entities_found"], list):
                        conv["entities_found"] = []
                        
                    valid_conversations.append(conv)
            data["history"] = valid_conversations
        
        # Update counts
        data["conversation_count"] = len(data["history"])
        
        # Validate extracted_entities
        if not isinstance(data["extracted_entities"], list):
            data["extracted_entities"] = []
        else:
            # Clean up entities list
            data["extracted_entities"] = [str(e) for e in data["extracted_entities"] if e]
            
        # Update metadata
        if not isinstance(data["metadata"], dict):
            data["metadata"] = {}
        data["metadata"]["total_questions"] = data["conversation_count"]
        data["metadata"]["total_entities"] = len(data["extracted_entities"])
        
        return data
        
    except Exception as e:
        logger.error(f"Error validating history data: {e}")
        return create_empty_history()


def create_empty_history() -> Dict[str, Any]:
    """Create empty history structure"""
    return {
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "conversation_count": 0,
        "history": [],
        "extracted_entities": [],
        "extracted_relationships": [],
        "active_entity": None,  # Store active entity for session persistence
        "metadata": {
            "temperature": 0.3,
            "model": "gemini-2.5-flash-lite",
            "total_questions": 0,
            "total_entities": 0,
        },
    }


def save_history_to_file(history_data: Dict[str, Any]):
    """Save conversation history to JSON file"""
    try:
        history_data["last_updated"] = datetime.now().isoformat()
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"History saved to {HISTORY_FILE}")
    except Exception as e:
        logger.error(f"Could not save history file: {e}")


def add_conversation_to_history(
    history_data: Dict[str, Any],
    question: str,
    answer: str,
    entities: list,
    timestamp: str,
):
    """Add a Q&A pair to history with data validation"""
    try:
        # Validate and sanitize inputs
        safe_question = str(question).strip() if question else ""
        safe_answer = str(answer).strip() if answer else ""
        safe_entities = [str(e).strip() for e in entities if e and str(e).strip()] if isinstance(entities, list) else []
        safe_timestamp = str(timestamp) if timestamp else datetime.now().isoformat()

        conversation = {
            "timestamp": safe_timestamp,
            "question": safe_question,
            "answer": safe_answer,
            "entities_found": safe_entities,
            "question_length": len(safe_question),
            "answer_length": len(safe_answer),
        }

        # Ensure history_data structure exists
        if "history" not in history_data:
            history_data["history"] = []
        if "extracted_entities" not in history_data:
            history_data["extracted_entities"] = []
        if "metadata" not in history_data:
            history_data["metadata"] = {}

        history_data["history"].append(conversation)
        history_data["conversation_count"] = len(history_data["history"])
        history_data["metadata"]["total_questions"] = history_data["conversation_count"]

        # Update entities safely
        for entity in safe_entities:
            if entity and entity not in history_data["extracted_entities"]:
                history_data["extracted_entities"].append(entity)

        history_data["metadata"]["total_entities"] = len(history_data["extracted_entities"])
        
    except Exception as e:
        logger.error(f"Error adding conversation to history: {e}")
        # Create minimal valid entry
        if "history" not in history_data:
            history_data["history"] = []
        history_data["history"].append({
            "timestamp": datetime.now().isoformat(),
            "question": "Error processing question",
            "answer": "Error processing answer", 
            "entities_found": [],
            "question_length": 0,
            "answer_length": 0,
        })

    # Keep only last 50 conversations to prevent file from growing too large
    if len(history_data["history"]) > 50:
        history_data["history"] = history_data["history"][-50:]
        history_data["conversation_count"] = len(history_data["history"])

    save_history_to_file(history_data)


def get_recent_history(history_data: Dict[str, Any], limit: int = 5) -> str:
    """Get recent conversation history as formatted string with robust error handling"""
    try:
        if not history_data or not isinstance(history_data, dict):
            return ""
        
        history_list = history_data.get("history", [])
        if not history_list or not isinstance(history_list, list):
            return ""

        # Safely get recent conversations
        recent = history_list[-limit:] if len(history_list) >= limit else history_list
        formatted_history = ""

        for i, conv in enumerate(recent, 1):
            try:
                # Safely extract conversation data
                if not isinstance(conv, dict):
                    continue
                    
                question = conv.get('question', 'N/A')
                answer = conv.get('answer', 'N/A')
                entities_found = conv.get('entities_found', [])
                
                # Ensure strings are properly handled
                question_str = str(question)[:100] if question else "N/A"
                answer_str = str(answer)[:100] if answer else "N/A"
                
                formatted_history += f"[Lần {i}] Q: {question_str}\n"
                formatted_history += f"[Lần {i}] A: {answer_str}...\n"
                
                # Handle entities safely
                if entities_found and isinstance(entities_found, list):
                    safe_entities = [str(e) for e in entities_found[:3] if e]  # Limit to 3 entities
                    if safe_entities:
                        formatted_history += f"[Lần {i}] Thực thể: {', '.join(safe_entities)}\n"
                        
                formatted_history += "\n"
                
            except Exception as e:
                logger.warning(f"Error processing conversation {i}: {e}")
                continue

        return formatted_history.strip()
        
    except Exception as e:
        logger.error(f"Error in get_recent_history: {e}")
        return ""


def show_history_stats(history_data: Optional[Dict[str, Any]] = None):
    """Show statistics about the conversation history.

    Accepts an optional history_data dict. If not provided, will try to use a
    global `history_data` variable, and finally will attempt to load from file.
    This keeps existing calls (no-arg) working while allowing explicit data.
    """
    # Prefer the passed-in history_data
    if history_data is None:
        # Try global variable next
        if "history_data" in globals():
            history_data = globals().get("history_data")
        else:
            # Fallback to loading from file
            try:
                history_data = load_history_from_file()
            except Exception:
                history_data = None

    if not history_data:
        print(f"{Colors.YELLOW}📝 No history data available.{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}{Colors.BLUE}📊 History Statistics:{Colors.RESET}")
    print(f"{Colors.CYAN}{'-'*40}{Colors.RESET}")
    try:
        print(
            f"{Colors.WHITE}Session ID: {Colors.CYAN}{history_data['session_id']}{Colors.RESET}"
        )
        print(
            f"{Colors.WHITE}Created: {Colors.CYAN}{history_data['created_at'][:19]}{Colors.RESET}"
        )
        print(
            f"{Colors.WHITE}Last Updated: {Colors.CYAN}{history_data['last_updated'][:19]}{Colors.RESET}"
        )
        print(
            f"{Colors.WHITE}Total Conversations: {Colors.CYAN}{history_data['conversation_count']}{Colors.RESET}"
        )
        print(
            f"{Colors.WHITE}Total Entities: {Colors.CYAN}{len(history_data.get('extracted_entities', []))}{Colors.RESET}"
        )
        print(
            f"{Colors.WHITE}File Size: {Colors.CYAN}{os.path.getsize(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else 0} bytes{Colors.RESET}"
        )
    except Exception as e:
        print(f"{Colors.RED}Error reading history data: {e}{Colors.RESET}")

    print(f"{Colors.CYAN}{'-'*40}{Colors.RESET}\n")


def print_banner():
    """Print a beautiful welcome banner"""
    banner = f"""
{Colors.CYAN}{'='*60}{Colors.RESET}
{Colors.BOLD}{Colors.CYAN}Chatbot{Colors.RESET}
{Colors.CYAN}{'='*60}{Colors.RESET}
{Colors.GREEN}🚀 Interactive Terminal Chatbot Mode{Colors.RESET}
{Colors.YELLOW}💡 Ask questions about your documents in Vietnamese{Colors.RESET}
{Colors.CYAN}{'='*60}{Colors.RESET}
"""
    print(banner)


def print_help(temperature: float = 0.3, extracted_entities: Optional[set] = None):
    """Print available commands"""
    entities_count = len(extracted_entities) if extracted_entities else 0

    help_text = f"""
{Colors.BOLD}{Colors.BLUE}📋 Available Commands:{Colors.RESET}
{Colors.CYAN}/help{Colors.RESET}     - Show this help message
{Colors.CYAN}/clear{Colors.RESET}    - Clear entities and reset to first session
{Colors.CYAN}/reset{Colors.RESET}    - Complete reset (delete all history)
{Colors.CYAN}/history{Colors.RESET}  - Show conversation history
{Colors.CYAN}/entities{Colors.RESET} - Show extracted entities ({entities_count} found)
{Colors.CYAN}/stats{Colors.RESET}    - Show history statistics
{Colors.CYAN}/active{Colors.RESET}   - Show current active entity status
{Colors.CYAN}/setactive <entity>{Colors.RESET} - Manually set active entity
{Colors.CYAN}/export [file]{Colors.RESET} - Export history to JSON file
{Colors.CYAN}/exit{Colors.RESET}     - Exit the chatbot
{Colors.CYAN}/quit{Colors.RESET}     - Exit the chatbot
{Colors.CYAN}/bye{Colors.RESET}      - Exit the chatbot

{Colors.BOLD}{Colors.BLUE}🌡️ Temperature Settings:{Colors.RESET}
{Colors.WHITE}Current: {Colors.CYAN}{temperature}{Colors.RESET} (balanced - focused yet natural){Colors.RESET}
{Colors.WHITE}• 0.0-0.3: Factual, consistent answers{Colors.RESET}
{Colors.WHITE}• 0.4-0.7: Balanced creativity{Colors.RESET}
{Colors.WHITE}• 0.8-2.0: Highly creative, varied responses{Colors.RESET}

{Colors.BOLD}{Colors.BLUE}💬 How to use:{Colors.RESET}
{Colors.WHITE}• Type your question in Vietnamese{Colors.RESET}
{Colors.WHITE}• Ask about technical specifications, devices, or documents{Colors.RESET}
{Colors.WHITE}• Examples: "LS-BE-001 có thông số gì?", "Thiết bị nào đo nhiệt độ?"{Colors.RESET}
"""
    print(help_text)


def format_user_message(message: str, timestamp: str) -> str:
    """Format user message with nice styling"""
    return f"""
{Colors.BOLD}{Colors.BLUE}👤 You [{timestamp}]{Colors.RESET}
{Colors.BLUE}┌─ {message}{Colors.RESET}
{Colors.BLUE}└─{Colors.RESET}"""


def format_bot_message(
    message: str, timestamp: str, processing_time: Optional[float] = None
) -> str:
    """Format bot message with nice styling"""
    time_info = f" ({processing_time:.1f}s)" if processing_time else ""
    return f"""
{Colors.BOLD}{Colors.GREEN}🤖 Bot [{timestamp}]{time_info}{Colors.RESET}
{Colors.GREEN}┌─ {message}{Colors.RESET}
{Colors.GREEN}└─{Colors.RESET}"""


def format_cypher_query(query: str) -> str:
    """Format Cypher query display"""
    return f"""
{Colors.BOLD}{Colors.YELLOW}🔍 Generated Cypher Query:{Colors.RESET}
{Colors.YELLOW}{query}{Colors.RESET}"""


def format_processing_step(title: str, content: str, icon: str = "⚙️") -> str:
    """Format processing step display with consistent styling"""
    return f"""
{Colors.BOLD}{Colors.MAGENTA}{icon} {title}:{Colors.RESET}
{Colors.CYAN}{content}{Colors.RESET}"""


def format_query_flow(original: str, enhanced: Optional[str] = None, active_entity: Optional[str] = None) -> str:
    """Format the complete query transformation flow"""
    lines = []
    lines.append(f"\n{Colors.BOLD}{Colors.BLUE}🔄 Query Processing Flow:{Colors.RESET}")
    lines.append(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    
    # Original query
    lines.append(f"{Colors.WHITE}📝 Original query:{Colors.RESET} {original}")
    
    # Active entity context
    if active_entity:
        lines.append(f"{Colors.YELLOW}🎯 Active entity:{Colors.RESET} {active_entity}")
    
    # Enhanced query
    if enhanced and enhanced != original:
        lines.append(f"{Colors.GREEN}✨ Enhanced query:{Colors.RESET} {enhanced}")
    elif enhanced:
        lines.append(f"{Colors.GRAY}✨ Enhanced query:{Colors.RESET} (no changes needed)")
    
    lines.append(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    return "\n".join(lines)


def get_best_active_entity_from_history(extracted_entities: set, known_entities: Optional[set] = None) -> Optional[str]:
    """Select the best active entity from extracted entities history
    Prioritizes real device entities from Neo4j over common words/greetings
    """
    if not extracted_entities:
        return None
    
    # Convert to list and reverse to check recent entities first
    entities_list = list(extracted_entities)
    entities_list.reverse()  # Most recent first
    
    # If we have known entities from Neo4j, use smart prioritization
    if known_entities:
        # First pass: Look for real entities from database (most recent first)
        for entity in entities_list:
            if is_real_entity(entity, known_entities) and not is_greeting_word(entity):
                logger.info(f"Selected real entity from history: {entity}")
                return entity
        
        # Second pass: Look for device-pattern entities if no real entities found
        for entity in entities_list:
            if is_device_entity(entity) and not is_greeting_word(entity):
                logger.info(f"Selected device-pattern entity from history: {entity}")
                return entity
    
    # Fallback: Use prioritization logic without Neo4j (for compatibility)
    best_entity = prioritize_entities(entities_list, known_entities)
    if best_entity:
        logger.info(f"Selected prioritized entity from history: {best_entity}")
        return best_entity
    
    logger.info("No suitable active entity found in history")
    return None


def fetch_entities_from_neo4j(driver) -> set:
    """Fetch all entity names from Neo4j database"""
    try:
        query = "MATCH (e:Entity) RETURN DISTINCT e.name as name LIMIT 10000"
        with driver.session() as session:
            result = session.run(query)
            entities = {record["name"] for record in result if record["name"]}
            logger.info(f"Fetched {len(entities)} entities from Neo4j")
            return entities
    except Exception as e:
        logger.error(f"Error fetching entities from Neo4j: {e}")
        return set()


def is_real_entity(entity: str, known_entities: set) -> bool:
    """Check if entity exists in the Neo4j database"""
    if not entity or not known_entities:
        return False
    
    entity_clean = entity.strip()
    # Direct match
    if entity_clean in known_entities:
        return True
    
    # Case-insensitive match
    entity_lower = entity_clean.lower()
    for known_entity in known_entities:
        if known_entity.lower() == entity_lower:
            return True
    
    # Also try with normalized case for device IDs
    entity_upper = entity_clean.upper()
    for known_entity in known_entities:
        if known_entity.upper() == entity_upper:
            return True
    
    return False


def is_device_entity(entity: str) -> bool:
    """Check if entity looks like a device/product ID - DEPRECATED: Use is_real_entity instead"""
    if not entity or len(entity.strip()) < 3:
        return False
    
    entity = entity.strip().upper()
    # Pattern for device IDs: letters-letters-numbers (e.g., LS-BE-001)
    device_patterns = [
        r'^[A-Z]{1,3}-[A-Z]{1,3}-\d{3}$',  # LS-BE-001 format
        r'^[A-Z]{2,4}-\d{3,4}$',           # AB-001 format
        r'^[A-Z]+\d{3,}$',                  # ABC123 format
    ]
    
    for pattern in device_patterns:
        if re.match(pattern, entity):
            return True
    return False


def is_greeting_word(entity: str) -> bool:
    """Check if entity is a greeting or common word"""
    if not entity:
        return False
    
    entity_lower = entity.lower().strip()
    greeting_words = {
        'chào', 'hello', 'hi', 'xin chào', 'mỹ', 'việt', 'nam',
        'tôi', 'bạn', 'anh', 'chị', 'ông', 'bà', 'em'
    }
    
    return entity_lower in greeting_words


def llm_identify_active_entity(
    client, 
    question: str, 
    known_entities: set, 
    current_active_entity: Optional[str] = None,
    conversation_history: Optional[str] = None,
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.1
) -> Optional[str]:
    """
    Use LLM to intelligently identify the main entity being discussed in the question.
    This replaces the manual entity extraction logic.
    """
    
    try:
        # Create a sample of known entities for context (avoid overwhelming the LLM)
        entity_sample = list(known_entities)[:50] if known_entities else []
        entity_examples = ", ".join(entity_sample[:20])  # Show first 20 as examples
        
        # Build conversation context if available
        context_section = ""
        if conversation_history:
            context_section = f"""
Ngữ cảnh cuộc trò chuyện gần đây:
{conversation_history}

Thực thể đang được thảo luận: {current_active_entity or 'Chưa xác định'}
"""
        else:
            context_section = f"""
Thực thể đang được thảo luận: {current_active_entity or 'Chưa xác định'}
"""
        
        prompt = f"""Bạn là chuyên gia phân tích ngôn ngữ tự nhiên. Nhiệm vụ của bạn là xác định CHÍNH XÁC thực thể chính mà người dùng đang hỏi về.

Câu hỏi: "{question}"

{context_section}

Danh sách một số thực thể có sẵn trong hệ thống:
{entity_examples}
(Lưu ý: Còn nhiều thực thể khác nữa trong hệ thống)

HƯỚNG DẪN PHÂN TÍCH:

1. TÌM THỰC THỂ CHÍNH:
   - Xác định tên thiết bị, mã sản phẩm, hoặc thực thể cụ thể trong câu hỏi
   - Ưu tiên thực thể được nhắc đến trực tiếp, rõ ràng nhất
   - Tránh nhầm lẫn với từ chào hỏi (chào, xin chào, hello, hi)

2. XỬ LÝ ĐẠI TỪ:
   - Nếu câu hỏi có đại từ (nó, này, đó, thiết bị này), tham khảo ngữ cảnh
   - Nếu không có ngữ cảnh rõ ràng, trả về None

3. KIỂM TRA TÍNH HỢP LỆ:
   - Chỉ trả về thực thể nếu có cơ sở rõ ràng từ câu hỏi
   - Tránh đoán mò hoặc suy luận quá mức

4. CÁC TRƯỜNG HỢP ĐẶC BIỆT:
   - Câu hỏi chung (như "có những loại nào?"): trả về None
   - Câu chào hỏi: trả về None
   - Câu hỏi về danh mục/thống kê: trả về None

ĐỊNH DẠNG TRẢ LỜI:
Chỉ trả về TÊN CHÍNH XÁC của thực thể hoặc "None" nếu không xác định được.

Ví dụ tốt:
- "LS-BE-001 có thông số gì?" → LS-BE-001
- "Nó hoạt động như thế nào?" (khi đang nói về iPhone 15) → iPhone 15
- "Có những thiết bị nào?" → None

THỰC THỂ ĐƯỢC NHẬN DIỆN:"""

        response = generate_answer(client, prompt, model=model, temperature=temperature)
        
        # Clean and validate response
        entity = response.strip() if response else ""
        
        # Handle common negative responses
        if entity.lower() in ['none', 'null', 'không có', 'không xác định', 'không rõ']:
            return None
            
        # Remove quotes if present
        entity = entity.strip('"\'')
        
        # Skip if it's a greeting or common word
        if is_greeting_word(entity) or entity.lower() in ['mỹ', 'chào', 'xin', 'hello', 'hi']:
            return None
            
        # Validate against known entities if available
        if known_entities and entity:
            # Exact match first
            if is_real_entity(entity, known_entities):
                return entity
            
            # Fuzzy match for slight variations
            from rapidfuzz import process
            match = process.extractOne(entity, list(known_entities), score_cutoff=80)
            if match:
                return match[0]
        
        # If not in known entities but seems like a valid entity name, return it
        # (This allows for new entities not yet in the database)
        if entity and len(entity) > 2 and not entity.isspace():
            return entity
            
        return None
        
    except Exception as e:
        logger.error(f"Error in LLM entity identification: {e}")
        return None


def prioritize_entities(entities: list, known_entities: Optional[set] = None) -> Optional[str]:
    """Legacy function - kept for backward compatibility. Use llm_identify_active_entity instead."""
    if not entities:
        return None
    
    # Filter out greetings first
    non_greeting_entities = [e for e in entities if not is_greeting_word(e)]
    if not non_greeting_entities:
        return None
    
    # If we have known entities from Neo4j, prioritize them
    if known_entities:
        # Prioritize real entities from database
        real_entities = [e for e in non_greeting_entities if is_real_entity(e, known_entities)]
        if real_entities:
            return real_entities[0]  # Return first real entity
    
    # Fallback to device pattern matching
    device_entities = [e for e in non_greeting_entities if is_device_entity(e)]
    if device_entities:
        return device_entities[0]  # Return first device entity
    
    # Return first non-greeting entity
    return non_greeting_entities[0]


def format_section_header(title: str, icon: str = "🔸") -> str:
    """Format a section header with consistent styling"""
    return f"\n{Colors.BOLD}{Colors.BLUE}{icon} {title}{Colors.RESET}"


def format_processing_step(title: str, content: str, icon: str = "⚙️") -> str:
    """Format processing step display with consistent styling"""
    return f"""
{Colors.BOLD}{Colors.MAGENTA}{icon} {title}:{Colors.RESET}
{Colors.CYAN}{content}{Colors.RESET}"""


def format_error_message(error: str) -> str:
    """Format error message"""
    return f"""
{Colors.BOLD}{Colors.RED}❌ Error:{Colors.RESET}
{Colors.RED}{error}{Colors.RESET}"""


def show_typing_indicator():
    """Show typing indicator"""
    print(f"{Colors.CYAN}🤔 Bot is thinking...{Colors.RESET}", end="", flush=True)
    time.sleep(0.5)
    print("\r" + " " * 30 + "\r", end="", flush=True)


def print_goodbye():
    """Print goodbye message"""
    goodbye = f"""
{Colors.BOLD}{Colors.GREEN}👋 Goodbye! Thanks for using VnEmisoft BAS{Colors.RESET}
{Colors.CYAN}Have a great day! 🎉{Colors.RESET}
"""
    print(goodbye)


def show_extracted_entities(
    extracted_entities: Optional[set] = None,
    extracted_relationships: Optional[set] = None,
):
    """Show all extracted entities and relationships"""
    if not extracted_entities and not extracted_relationships:
        print(f"{Colors.YELLOW}📝 No entities extracted yet.{Colors.RESET}")
        return

    print(
        f"\n{Colors.BOLD}{Colors.MAGENTA}🏷️  Extracted Entities & Relationships:{Colors.RESET}"
    )
    print(f"{Colors.CYAN}{'-'*60}{Colors.RESET}")

    if extracted_entities:
        print(
            f"{Colors.BOLD}{Colors.GREEN}📍 Entities ({len(extracted_entities)}):{Colors.RESET}"
        )
        entities_list = sorted(list(extracted_entities))
        for i, entity in enumerate(entities_list, 1):
            print(f"  {i:2d}. {entity}")
        print()

    if extracted_relationships:
        print(
            f"{Colors.BOLD}{Colors.BLUE}🔗 Relationships ({len(extracted_relationships)}):{Colors.RESET}"
        )
        relationships_list = sorted(list(extracted_relationships))
        for i, rel in enumerate(relationships_list, 1):
            print(f"  {i:2d}. {rel}")
        print()

    print(f"{Colors.CYAN}{'-'*60}{Colors.RESET}\n")


def print_history(
    history: str,
    extracted_entities: Optional[set] = None,
    extracted_relationships: Optional[set] = None,
):
    """Print conversation history"""
    if not history.strip():
        print(f"{Colors.YELLOW}📝 No conversation history yet.{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}{Colors.BLUE}📚 Conversation History:{Colors.RESET}")
    print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")

    # Split history into Q&A pairs
    lines = history.strip().split("\n")
    for line in lines:
        if line.startswith("Q: "):
            print(f"{Colors.BLUE}❓ {line[3:]}{Colors.RESET}")
        elif line.startswith("A: "):
            print(f"{Colors.GREEN}💡 {line[3:]}{Colors.RESET}")
        elif line.startswith("Entities: "):
            entities = line[10:]
            if entities.strip():
                print(f"{Colors.YELLOW}🏷️  {entities}{Colors.RESET}")

    print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}\n")

    # Show summary of extracted entities
    if extracted_entities:
        print(
            f"{Colors.BOLD}{Colors.MAGENTA}📊 Extracted Entities Summary:{Colors.RESET}"
        )
        print(
            f"{Colors.MAGENTA}Total entities: {len(extracted_entities)}{Colors.RESET}"
        )
        if extracted_entities:
            sample_entities = list(extracted_entities)[:10]
            print(f"{Colors.MAGENTA}Sample: {', '.join(sample_entities)}{Colors.RESET}")
        print()


PROMPT_TEMPLATE = (
    "Extract entities (nodes) and their relationships (edges) from the text below. "
    "Entities and relationships MUST be in Vietnamese. "
    "Include all technical specifications, parameters, manufacturers, origins, installation locations, and other relevant details as entities and relationships. "
    "Common relationship types: ĐƯỢC_SẢN_XUẤT_BỞI, CÓ_XUẤT_XỨ_TỪ, ĐƯỢC_LẮP_ĐẶT_TẠI, CÓ_THÔNG_SỐ, ĐO, HỖ_TRỢ, ỨNG_DỤNG, CÓ_GIÁ_TRỊ, etc. "
    "For technical specifications sections, parse tables and create relationships like (Device, CÓ_THÔNG_SỐ, Parameter), (Parameter, CÓ_GIÁ_TRỊ, Value). "
    "Examples: "
    "- (LS-BE-001, CÓ_THÔNG_SỐ, Dải đo khoảng cách tối đa) "
    "- (Dải đo khoảng cách tối đa, CÓ_GIÁ_TRỊ, 0,5 - 3.000 m) "
    "- (LS-BE-001, ĐƯỢC_SẢN_XUẤT_BỞI, BlueEco) "
    "Format: "
    "Entities: "
    "- {Entity}: {Type} "
    "Relationships: "
    "- ({Entity1}, {RelationshipType}, {Entity2}) "
)


def call_gemini(
    client,
    prompt_text: str,
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.3,
) -> str:
    """Call the GenAI client to generate text. This wraps genai.Client usage.

    The `genai` client API varies by version: this script attempts to use the
    `client.generate_text(...)` if available, otherwise falls back to `client.batch`/legacy.
    """
    try:
        # Use the same surface as other parts of the project: client.models.generate_content
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(
                model=model, contents=prompt_text, config={"temperature": temperature}
            )
            # resp may have .text or .output or .candidates; try common fields
            if getattr(resp, "text", None):
                return resp.text
            if getattr(resp, "output", None):
                return getattr(resp, "output")
            # Fallback to string
            return str(resp)

        # If API surface differs, try generate_content at top-level
        if hasattr(client, "generate_content"):
            resp = client.generate_content(
                model=model, contents=prompt_text, temperature=temperature
            )
            if getattr(resp, "text", None):
                return resp.text
            return str(resp)

        raise RuntimeError(
            "GenAI client does not expose a supported generate_content API"
        )

    except Exception as e:
        logger.error(f"Error calling GenAI client: {e}")
        raise


def parse_llm_output(result: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    entity_pattern = r"- (.+): (.+)"
    entities = re.findall(entity_pattern, result)
    entity_dict = {
        entity.strip(): entity_type.strip() for entity, entity_type in entities
    }
    entity_list = list(entity_dict.keys())

    relationship_pattern = r"- \(([^,]+), ([^,]+), ([^)]+)\)"
    relationships = re.findall(relationship_pattern, result)
    relationship_list = []
    for subject, relation, object_ in relationships:
        rel = relation.strip().replace(" ", "_").upper()
        relationship_list.append((subject.strip(), rel, object_.strip()))

    return entity_list, relationship_list


# ---- Neo4j utilities (no LangChain) ----
def get_neo4j_driver_from_env():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        # Optional ping
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
            s.run("RETURN 1 AS ok").single()
        return driver
    except Exception as e:
        logger.error(f"Failed to connect Neo4j at {uri}: {e}")
        raise


def fetch_schema_summary(driver) -> str:
    """Build a lightweight schema summary for prompt guidance.
    Lists relationship types and notes that nodes use label Entity(name).
    """
    rel_types: List[str] = []
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
            # Neo4j 5.x: CALL db.relationshipTypes()
            try:
                records = s.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
                rel_types = [r[0] for r in records]
            except Exception:
                # Fallback for older versions
                records = s.run(
                    "MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationshipType"
                )
                rel_types = [r["relationshipType"] for r in records]
    except Exception as e:
        logger.warning(f"Failed to fetch relationship types: {e}")

    rel_list = ", ".join(sorted(set(rel_types))) if rel_types else "(unknown)"
    schema = (
        "Nodes: (:Entity { name: STRING })\n" f"Relationships types present: {rel_list}"
    )
    return schema


def generate_cypher_prompt(schema: str, question: str) -> str:
    """Prompt adapted from OptimizedGraphRAGService._get_cypher_template, without LangChain."""
    return f"""
Task: Generate a Cypher statement to query a Neo4j graph database.
Instructions:
- Analyze the question and extract relevant graph components.
- Use only the relationship types and properties from the provided schema.
- Schema:
{schema}
- Return only the Cypher query (no extra text).
- Prefer Vietnamese names as-is; compare with toLower() for case-insensitive matching.
- Include LIMIT 100 at the end unless a single value is expected.

Examples:
# Nhiệt độ không khí có thông số gì?
MATCH (device:Entity)-[:CÓ_THÔNG_SỐ]->(param:Entity)
WHERE toLower(device.name) = toLower('Nhiệt độ không khí')
RETURN param.name LIMIT 100

# LS-BE-001 có thông số?
MATCH (device:Entity)-[:CÓ_THÔNG_SỐ]->(param:Entity)
WHERE toLower(device.name) = toLower('LS-BE-001')
OPTIONAL MATCH (param)-[:CÓ_GIÁ_TRỊ]->(value:Entity)
RETURN param.name, value.name LIMIT 100

The question is:
{question}
""".strip()


def strip_code_fences(text: str) -> str:
    # Remove ```cypher ... ``` fences if model returns fenced code
    text = text.strip()
    m = re.match(r"```[a-zA-Z]*\n([\s\S]*?)\n```", text)
    return m.group(1).strip() if m else text


def generate_cypher_via_llm(
    client, schema: str, question: str, model: str, temperature: float = 0.3
) -> str:
    prompt = generate_cypher_prompt(schema, question)
    output = call_gemini(
        client, prompt_text=prompt, model=model, temperature=temperature
    )
    return strip_code_fences(output)


def run_cypher(driver, query: str):
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    with driver.session(database=db) as s:
        records = s.run(query)
        # Return list of dicts for display
        return [r.data() for r in records]


def find_closest_entities(
    entities: List[str], node_mapping: Dict[int, str]
) -> List[Tuple[str, int, str, float]]:
    """
    Finds the closest matching entities in node_mapping for a list of query entities.

    Parameters:
        entities (list): List of entity names to match.
        node_mapping (dict): Mapping of node IDs to entity names.

    Returns:
        list: A list of tuples [(query_entity, closest_match_id, closest_match_name, score)].
    """
    results = []
    node_names = list(node_mapping.values())
    if not node_names:
        return results

    for entity in entities:
        match = process.extractOne(entity, node_names)
        if not match:
            continue
        closest_match, score, index = match
        closest_match_id = list(node_mapping.keys())[int(index)]
        results.append((entity, closest_match_id, closest_match, float(score)))

    return results


def load_node_mapping_from_neo4j(driver) -> Dict[int, str]:
    """
    Load node mapping (node_id -> name) from Neo4j for fuzzy matching.
    """
    try:
        db = os.getenv("NEO4J_DATABASE", "neo4j")
        with driver.session(database=db) as session:
            nodes_query = "MATCH (n:Entity) RETURN id(n) AS node_id, n.name AS name"
            nodes = session.run(nodes_query)
            node_mapping = {record["node_id"]: record["name"] for record in nodes}
        logger.info(f"Loaded {len(node_mapping)} nodes for fuzzy matching")
        return node_mapping
    except Exception as e:
        logger.error(f"Error loading node mapping: {e}")
        return {}


# ---- Answer synthesis (no LangChain) ----
def format_records_as_context(records: List[dict], max_items: int = 25) -> str:
    if not records:
        return "(no results)"
    lines: List[str] = []
    for i, row in enumerate(records[:max_items], 1):
        # Convert dict row to simple key: value pairs
        kv = ", ".join(f"{k}: {v}" for k, v in row.items())
        lines.append(f"{i}. {kv}")
    return "\n".join(lines)


def build_answer_prompt(
    question: str,
    context: str,
    conversation_context: str = "",
    known_entities: Optional[set] = None,
) -> str:
    # Check if we have actual data or just empty results
    has_neo4j_data = context and context.strip() != "(no results)"
    
    if has_neo4j_data:
        sys_instr = (
            "Bạn là trợ lý AI của Reecotech. "
            "Chỉ sử dụng thông tin trong kết quả truy vấn (context) dưới đây để trả lời. "
            "Không bịa đặt, không tham chiếu nguồn bên ngoài. "
            "Trả lời ngắn gọn, có cấu trúc, bằng tiếng Việt."
        )
    else:
        sys_instr = (
            "Bạn là trợ lý AI của Reecotech. "
            "Database không tìm thấy kết quả cho câu hỏi này. "
            "Nếu có lịch sử cuộc trò chuyện, bạn có thể tham khảo NHƯNG phải ghi rõ 'Dựa trên thông tin từ cuộc trò chuyện trước'. "
            "Nếu không có lịch sử, trả lời 'Không tìm thấy thông tin về [entity] trong database'. "
            "Trả lời ngắn gọn, có cấu trúc, bằng tiếng Việt."
        )

    # Add known entities to system instruction if available
    if known_entities:
        entities_str = ", ".join(list(known_entities)[:10])  # Limit to 10 entities
        sys_instr += f"\n\nThực thể đã biết: {entities_str}"

    # Build conversation history section
    history_section = ""
    if conversation_context and conversation_context.strip():
        if has_neo4j_data:
            history_section = f"""
Lịch sử cuộc trò chuyện gần đây (chỉ để tham khảo):
{conversation_context}

Hướng dẫn: Ưu tiên thông tin từ database, chỉ dùng lịch sử để làm rõ ngữ cảnh.
"""
        else:
            history_section = f"""
Lịch sử cuộc trò chuyện gần đây:
{conversation_context}

⚠️ Lưu ý: Database không có kết quả. Nếu trả lời dựa trên lịch sử, phải ghi rõ nguồn.
"""

    data_status = "✅ Có dữ liệu từ database" if has_neo4j_data else "❌ Không có dữ liệu từ database"
    
    prompt = f"""
{sys_instr}

Câu hỏi: {question}

Trạng thái dữ liệu: {data_status}

Kết quả truy vấn (context):
{context}
{history_section}
Yêu cầu trả lời:
- Tóm tắt chính xác dựa trên context, tránh lan man
- Dùng gạch đầu dòng khi phù hợp, tối đa ~5 câu
- Nếu không đủ dữ liệu database, hãy nói rõ không tìm thấy thông tin phù hợp
- Nếu dùng thông tin từ lịch sử, phải ghi rõ "Dựa trên cuộc trò chuyện trước:"
"""
    return textwrap.dedent(prompt).strip()


def generate_answer(
    client, prompt: str, model: str = "gemini-2.5-flash-lite", temperature: float = 0.3
) -> str:
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(
                model=model, contents=prompt, config={"temperature": temperature}
            )
            if getattr(resp, "text", None):
                return resp.text.strip()
            return str(resp)
        if hasattr(client, "generate_content"):
            resp = client.generate_content(
                model=model, contents=prompt, temperature=temperature
            )
            if getattr(resp, "text", None):
                return resp.text.strip()
            return str(resp)
        raise RuntimeError(
            "GenAI client does not expose a supported generate_content API"
        )
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Xin lỗi, có lỗi khi tạo câu trả lời."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text",
        nargs="?",
        help="Text input: entity extraction (mode=extract) or question (mode=query)",
    )
    parser.add_argument("--file", "-f", help="Path to a text file to read input from")
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-2.5-flash-lite",
        help="Model to use (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.3,
        help="Temperature for response generation (0.0-2.0, default: 0.3). "
        "Lower = more focused/consistent, higher = more creative/varied. "
        "Recommended: 0.1-0.3 for factual queries, 0.7-1.0 for creative tasks",
    )
    parser.add_argument(
        "--mode",
        choices=["extract", "query", "chat", "interactive"],
        default="extract",
        help=(
            "Mode: 'extract' to extract entities; 'query' to generate+run Cypher; "
            "'chat' to answer in natural language; 'interactive' to run a terminal chatbot loop"
        ),
    )
    parser.add_argument(
        "--history",
        help="Optional prior conversation context (plain text)",
        default="",
    )
    args = parser.parse_args()

    # Interactive mode: terminal chatbot loop
    if args.mode == "interactive":
        # Load history from file
        history_data = load_history_from_file()
        history_data["metadata"]["temperature"] = args.temperature
        history_data["metadata"]["model"] = args.model

        # Initialize variables from loaded data
        extracted_entities = set(history_data["extracted_entities"])
        extracted_relationships = set(history_data["extracted_relationships"])
        history = get_recent_history(history_data)
        conversation_count = history_data["conversation_count"]

        # Track last active entity for pronoun resolution in follow-up questions
        active_entity: Optional[str] = None
        # We'll set this after loading Neo4j entities for proper validation
        
        print_banner()
        print_help(args.temperature, extracted_entities)
        show_history_stats()

        try:
            client = init_genai_client()
            driver = get_neo4j_driver_from_env()
            schema = fetch_schema_summary(driver)
            
            # Initialize optimized services
            cache_manager = EntityCacheManager()
            entity_cache = cache_manager.get_entities(driver)
            fuzzy_service = FuzzyMatchingService(entity_cache)
            query_processor = QueryProcessor(client, entity_cache)
            
            # Legacy variables for backward compatibility
            node_mapping = entity_cache.node_mapping
            known_entities = entity_cache.known_entities
            
            # NOW we can intelligently select active_entity with Neo4j validation
            # Only use stored/history active_entity if this is NOT the first session
            stored_active_entity = history_data.get("active_entity")
            
            if history_data["conversation_count"] > 0:
                # This is a continuing session - use intelligent selection
                if stored_active_entity and is_real_entity(stored_active_entity, known_entities) and not is_greeting_word(stored_active_entity):
                    active_entity = stored_active_entity
                    logger.info(f"Using stored active_entity from previous session: {active_entity}")
                else:
                    active_entity = get_best_active_entity_from_history(extracted_entities, known_entities)
                    if stored_active_entity != active_entity:
                        logger.info(f"Active entity changed from stored '{stored_active_entity}' to selected '{active_entity}'")
            else:
                # This is the first session - start clean
                active_entity = None
                logger.info("First session - starting with no active entity")
            
        except Exception as e:
            print(format_error_message(f"Failed to initialize services: {e}"))
            sys.exit(1)

        print(
            f"{Colors.BOLD}{Colors.GREEN}✅ Ready! Start asking questions about your documents.{Colors.RESET}"
        )
        print(
            f"{Colors.CYAN}🤖 Entity Recognition: LLM-powered (replaced manual patterns){Colors.RESET}"
        )
        print(
            f"{Colors.CYAN}📊 Loaded {len(known_entities)} entities from Neo4j database for validation.{Colors.RESET}"
        )
        
        # Show session status and active entity info
        is_first_session = history_data["conversation_count"] == 0
        session_status = "New session" if is_first_session else f"Continuing session ({history_data['conversation_count']} previous questions)"
        
        print(
            f"{Colors.CYAN}📈 Session: {session_status}{Colors.RESET}"
        )
        
        # Show active entity status with more detail
        if active_entity:
            is_real = is_real_entity(active_entity, known_entities)
            entity_type = "Neo4j entity" if is_real else "pattern entity" if is_device_entity(active_entity) else "other entity"
            print(
                f"{Colors.CYAN}🎯 Active entity: {Colors.BOLD}{Colors.GREEN}{active_entity}{Colors.RESET}{Colors.CYAN} ({entity_type}){Colors.RESET}"
            )
        else:
            reason = "new session" if is_first_session else "no suitable entity found"
            print(
                f"{Colors.CYAN}⚠️  No active entity ({reason}){Colors.RESET}"
            )
            
        print(
            f"{Colors.CYAN}Type your question or '/help' for commands.{Colors.RESET}\n"
        )

        while True:
            try:
                # Get user input with colored prompt
                user_input = input(
                    f"{Colors.BOLD}{Colors.BLUE}You> {Colors.RESET}"
                ).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print_goodbye()
                    break
                elif user_input.lower() == "/help":
                    print_help(args.temperature, extracted_entities)
                    continue
                elif user_input.lower() == "/entities":
                    show_extracted_entities(extracted_entities, extracted_relationships)
                    continue
                elif user_input.lower() == "/clear":
                    extracted_entities.clear()
                    extracted_relationships.clear()
                    active_entity = None  # Reset active entity
                    history_data["extracted_entities"] = []
                    history_data["active_entity"] = None  # Reset in history
                    history_data["conversation_count"] = 0  # Reset to first session state
                    save_history_to_file(history_data)
                    print(f"{Colors.GREEN}✅ Cleared all data and reset to first session state!{Colors.RESET}")
                    continue
                elif user_input.lower() == "/reset":
                    # Complete reset - like starting fresh
                    extracted_entities.clear()
                    extracted_relationships.clear()
                    active_entity = None
                    history_data = create_empty_history()
                    history_data["metadata"]["temperature"] = args.temperature
                    history_data["metadata"]["model"] = args.model
                    save_history_to_file(history_data)
                    print(f"{Colors.GREEN}✅ Complete reset - starting fresh session!{Colors.RESET}")
                    continue
                elif user_input.lower() == "/active":
                    # Show current active entity status
                    if active_entity:
                        is_real = is_real_entity(active_entity, known_entities)
                        is_greeting = is_greeting_word(active_entity)
                        entity_type = "Neo4j entity" if is_real else "pattern entity" if is_device_entity(active_entity) else "other entity"
                        status_color = Colors.GREEN if is_real else Colors.YELLOW if is_device_entity(active_entity) else Colors.RED
                        
                        print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 Current Active Entity:{Colors.RESET}")
                        print(f"  {Colors.WHITE}Entity:{Colors.RESET} {status_color}{active_entity}{Colors.RESET}")
                        print(f"  {Colors.WHITE}Type:{Colors.RESET} {entity_type}")
                        print(f"  {Colors.WHITE}In Neo4j:{Colors.RESET} {'✅ Yes' if is_real else '❌ No'}")
                        print(f"  {Colors.WHITE}Is Greeting:{Colors.RESET} {'⚠️ Yes' if is_greeting else '✅ No'}")
                    else:
                        print(f"{Colors.YELLOW}⚠️  No active entity set{Colors.RESET}")
                    continue
                elif user_input.lower().startswith("/setactive "):
                    # Manually set active entity
                    new_entity = user_input[11:].strip()
                    if new_entity:
                        if is_real_entity(new_entity, known_entities):
                            active_entity = new_entity
                            history_data["active_entity"] = active_entity
                            save_history_to_file(history_data)
                            print(f"{Colors.GREEN}✅ Active entity set to: {Colors.BOLD}{new_entity}{Colors.RESET}")
                        else:
                            print(f"{Colors.YELLOW}⚠️  Warning: '{new_entity}' is not in Neo4j database. Set anyway? (y/n): {Colors.RESET}", end="")
                            confirm = input().strip().lower()
                            if confirm in ['y', 'yes']:
                                active_entity = new_entity
                                history_data["active_entity"] = active_entity
                                save_history_to_file(history_data)
                                print(f"{Colors.GREEN}✅ Active entity set to: {Colors.BOLD}{new_entity}{Colors.RESET}")
                            else:
                                print(f"{Colors.CYAN}Operation cancelled.{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}❌ Usage: /setactive <entity_name>{Colors.RESET}")
                    continue

                # Handle greetings
                if is_greeting_word(user_input) or any(greeting in user_input.lower() for greeting in ['chào', 'hello', 'hi', 'xin chào']):
                    print(f"\n{Colors.BOLD}{Colors.GREEN}🤖 Bot [{datetime.now().strftime('%H:%M:%S')}]{Colors.RESET}")
                    print(f"┌─ Xin chào! Tôi có thể giúp bạn tìm hiểu về các thiết bị và thông số kỹ thuật.")
                    print(f"└─ Hãy hỏi về một thiết bị cụ thể, ví dụ: 'LS-BE-001'")
                    continue

                # Reset active entity if it's a greeting word
                if active_entity and is_greeting_word(active_entity):
                    logger.info(f"Resetting invalid active_entity: {active_entity}")
                    active_entity = None

                    print(
                        f"{Colors.GREEN}🧹 Chat history and extracted entities cleared! New session created.{Colors.RESET}\n"
                    )
                    continue
                elif user_input.lower() == "/history":
                    print_history(history, extracted_entities, extracted_relationships)
                    continue
                elif user_input.lower() == "/stats":
                    show_history_stats()
                    continue
                elif user_input.lower().startswith("/export"):
                    # Export history to a specific file
                    parts = user_input.split()
                    export_file = (
                        parts[1]
                        if len(parts) > 1
                        else f"chatbot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    try:
                        with open(export_file, "w", encoding="utf-8") as f:
                            json.dump(history_data, f, ensure_ascii=False, indent=2)
                        print(
                            f"{Colors.GREEN}✅ History exported to: {export_file}{Colors.RESET}\n"
                        )
                    except Exception as e:
                        print(f"{Colors.RED}❌ Export failed: {e}{Colors.RESET}\n")
                    continue

                # Process question
                conversation_count += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n{Colors.BOLD}{Colors.WHITE}{'='*80}{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.BLUE}💬 Processing Query:{Colors.RESET} {user_input}")
                if active_entity:
                    print(f"{Colors.BOLD}{Colors.GREEN}🎯 Session Context:{Colors.RESET} Active entity = {active_entity}")
                else:
                    print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  Session Context:{Colors.RESET} No active entity")
                print(f"{Colors.BOLD}{Colors.WHITE}{'='*80}{Colors.RESET}")
                
                start_time = time.time()

                # Show user message
                print(format_user_message(user_input, timestamp))

                # Show typing indicator
                show_typing_indicator()

                try:
                    # Step 1: Pass raw input directly to LLM for context-aware processing
                    # LLM handles pronoun resolution better with full conversation context
                    user_input_for_cypher = user_input

                    # Initialize identified_entity for later use
                    identified_entity = None

                    # Step 2: Optimized Entity Processing (combines identification + extraction)
                    print(format_section_header("Smart Entity Processing", "�"))
                    entity_processing_start = time.time()
                    
                    # Use optimized query processor for combined entity identification and extraction
                    llm_result = query_processor.process_entity_identification(
                        question=user_input_for_cypher,
                        current_active_entity=active_entity,
                        conversation_history=history,
                        model=args.model
                    )
                    
                    identified_entity = llm_result.identified_entity
                    extracted_entities_list = llm_result.extracted_entities
                    confidence = llm_result.confidence
                    
                    # Display processing results
                    if identified_entity:
                        print(f"{Colors.GREEN}✅ Primary entity identified: {Colors.BOLD}{identified_entity}{Colors.RESET} (confidence: {confidence:.1f})")
                        
                        # Validate the identified entity
                        if is_real_entity(identified_entity, known_entities):
                            print(f"{Colors.CYAN}🔍 Entity validation: {Colors.GREEN}✓ Found in Neo4j database{Colors.RESET}")
                            active_entity = identified_entity
                            extracted_entities.add(active_entity)
                            history_data["extracted_entities"] = list(extracted_entities)
                            history_data["active_entity"] = active_entity
                            save_history_to_file(history_data)
                            logger.info(f"Active entity set by LLM: {active_entity}")
                        else:
                            print(f"{Colors.YELLOW}⚠️  Entity validation: Not found in Neo4j database, but keeping as context{Colors.RESET}")
                            active_entity = identified_entity
                            history_data["active_entity"] = active_entity
                            save_history_to_file(history_data)
                            logger.info(f"Active entity set by LLM (not in DB): {active_entity}")
                    else:
                        print(f"{Colors.YELLOW}ℹ️  No specific entity identified - will use fuzzy matching{Colors.RESET}")

                    # Step 3: Smart Fuzzy Matching (only if needed)
                    enhanced_query = user_input_for_cypher
                    fuzzy_matches = []
                    
                    # Only do fuzzy matching if we don't have high-confidence entity or have extracted entities
                    if confidence < 0.8 or extracted_entities_list:
                        print(format_section_header("Smart Fuzzy Matching", "🔍"))
                        
                        # Use fuzzy matching service for optimized matching
                        enhanced_query, fuzzy_matches = fuzzy_service.enhance_query(
                            user_input_for_cypher, 
                            extracted_entities_list,
                            confidence_threshold=70.0
                        )
                        
                        if fuzzy_matches:
                            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
                            for match in fuzzy_matches:
                                color = Colors.GREEN if match.confidence > 70 else Colors.YELLOW if match.confidence > 50 else Colors.RED
                                print(f"{color}  '{match.query_entity}' → '{match.match_name}' (confidence: {match.confidence:.1f}%){Colors.RESET}")
                            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
                            
                            # Update active entity from best fuzzy match if LLM didn't find one
                            if not identified_entity:
                                best_match = max(fuzzy_matches, key=lambda m: m.confidence)
                                if best_match.confidence > 70:
                                    active_entity = best_match.match_name
                                    extracted_entities.add(active_entity)
                                    history_data["extracted_entities"] = list(extracted_entities)
                                    history_data["active_entity"] = active_entity
                                    save_history_to_file(history_data)
                                    print(f"{Colors.GREEN}🎯 Active entity set from fuzzy matching: {Colors.BOLD}{active_entity}{Colors.RESET}")
                                    logger.info(f"Active entity set from fuzzy matching: {active_entity}")
                            
                            # Show replacements made
                            if enhanced_query != user_input_for_cypher:
                                good_matches = [m for m in fuzzy_matches if m.confidence > 70]
                                for match in good_matches:
                                    print(f"{Colors.CYAN}🔄 Enhanced: '{match.query_entity}' → '{match.match_name}'{Colors.RESET}")
                        else:
                            print(f"{Colors.YELLOW}ℹ️  No fuzzy matches found{Colors.RESET}")
                    else:
                        print(f"{Colors.GREEN}⚡ Skipped fuzzy matching (high-confidence entity: {identified_entity}){Colors.RESET}")

                    entity_processing_time = time.time() - entity_processing_start
                    print(f"{Colors.GRAY}⏱️  Entity processing completed in {entity_processing_time:.2f}s{Colors.RESET}")
                    
                    # Display the complete query transformation flow
                    print(format_query_flow(user_input, enhanced_query, active_entity))

                    # Step 4: Generate Cypher query
                    print(format_section_header("Cypher Generation", "⚙️"))
                    cypher_start = time.time()
                    cypher = generate_cypher_via_llm(
                        client,
                        schema=schema,
                        question=enhanced_query,
                        model=args.model,
                        temperature=args.temperature,
                    )
                    # Try to extract an explicit entity name from the generated Cypher
                    # Only as secondary validation, don't override LLM decision
                    try:
                        m = re.search(
                            r"toLower\([^)]*\)\s*=\s*toLower\((?:'|\")([^'\"]+)(?:'|\")\)",
                            cypher,
                            flags=re.IGNORECASE,
                        )
                        if m:
                            extracted_from_cypher = m.group(1).strip()
                            # Only use if LLM didn't identify any entity and this is a real entity
                            if not identified_entity and extracted_from_cypher and is_real_entity(extracted_from_cypher, known_entities) and not is_greeting_word(extracted_from_cypher):
                                active_entity = extracted_from_cypher
                                extracted_entities.add(active_entity)
                                history_data["extracted_entities"] = list(extracted_entities)
                                history_data["active_entity"] = active_entity
                                save_history_to_file(history_data)
                                logger.info(f"Active entity set from Cypher (LLM backup): {active_entity}")
                    except Exception as e:
                        logger.error(f"Error extracting entity from Cypher: {e}")

                    # If generated Cypher didn't contain an explicit entity but we have active_entity,
                    # build a small intent->relation mapping to construct a safe Cypher query.
                    try:
                        has_entity_in_cypher = bool(
                            re.search(
                                r"toLower\([^)]*\)\s*=\s*toLower\(",
                                cypher,
                                flags=re.IGNORECASE,
                            )
                        )
                    except Exception:
                        has_entity_in_cypher = False

                    if not has_entity_in_cypher and active_entity:
                        q_lower = user_input.lower()
                        # Simple keyword -> relation templates
                        relation_templates = [
                            (
                                [
                                    "thông số",
                                    "cấu hình",
                                    "thông số kỹ thuật",
                                    "thông số?",
                                ],
                                "MATCH (d:Entity)-[:CÓ_THÔNG_SỐ]->(param:Entity)\nWHERE toLower(d.name) = toLower('{ent}')\nOPTIONAL MATCH (param)-[:CÓ_GIÁ_TRỊ]->(val:Entity)\nRETURN param.name, val.name LIMIT 100",
                            ),
                            (
                                ["lắp", "lắp đặt", "đặt tại", "vị trí", "ở đâu"],
                                "MATCH (d:Entity)-[:ĐƯỢC_LẮP_ĐẶT_TẠI]->(loc:Entity)\nWHERE toLower(d.name) = toLower('{ent}')\nRETURN loc.name LIMIT 100",
                            ),
                            (
                                [
                                    "sản xuất",
                                    "sản xuất bởi",
                                    "nhà sản xuất",
                                    "manufacturer",
                                ],
                                "MATCH (d:Entity)-[:ĐƯỢC_SẢN_XUẤT_BỞI]->(m:Entity)\nWHERE toLower(d.name) = toLower('{ent}')\nRETURN m.name LIMIT 100",
                            ),
                            (
                                ["kết nối", "kết nối với", "kết nối qua"],
                                "MATCH (d:Entity)-[:KẾT_NỐI_VỚI]->(t:Entity)\nWHERE toLower(d.name) = toLower('{ent}')\nRETURN t.name LIMIT 100",
                            ),
                            (
                                ["model"],
                                "MATCH (d:Entity)-[:CÓ_MODEL]->(m:Entity)\nWHERE toLower(d.name) = toLower('{ent}')\nRETURN m.name LIMIT 100",
                            ),
                        ]

                        forced = None
                        for keywords, tpl in relation_templates:
                            for kw in keywords:
                                if kw in q_lower:
                                    forced = tpl.format(ent=active_entity)
                                    break
                            if forced:
                                break

                        if forced:
                            cypher = forced
                            logger.info(
                                f"Using forced Cypher based on active_entity and intent: {cypher}"
                            )
                    cypher_time = time.time() - cypher_start

                    # Always show the generated/forced Cypher for testing purposes
                    print(f"\n{Colors.BOLD}{Colors.YELLOW}🎯 Final Cypher Query:{Colors.RESET}")
                    print(f"{Colors.YELLOW}{'-'*60}{Colors.RESET}")
                    print(f"{Colors.YELLOW}{cypher}{Colors.RESET}")
                    print(f"{Colors.YELLOW}{'-'*60}{Colors.RESET}")

                    # Execute query
                    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 Executing Query...{Colors.RESET}")
                    query_start = time.time()
                    records = run_cypher(driver, cypher)
                    query_time = time.time() - query_start

                    # Enhanced results display
                    print(f"\n{Colors.BOLD}{Colors.GREEN}📊 Query Results:{Colors.RESET}")
                    print(f"{Colors.GREEN}{'-'*40}{Colors.RESET}")
                    
                    try:
                        if not records:
                            print(f"{Colors.RED}❌ No results found (0 rows){Colors.RESET}")
                        else:
                            print(f"{Colors.GREEN}✅ Found {len(records)} result(s){Colors.RESET}")
                            
                            # Show a preview of results in a compact format
                            print(f"\n{Colors.BOLD}{Colors.CYAN}� Results Preview:{Colors.RESET}")
                            for i, row in enumerate(records[:5], 1):  # Show first 5 results
                                kv = ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
                                print(f"  {Colors.CYAN}{i}.{Colors.RESET} {kv}")
                            
                            if len(records) > 5:
                                print(f"  {Colors.GRAY}... and {len(records) - 5} more results{Colors.RESET}")
                            
                            # Detailed view (collapsible)
                            print(f"\n{Colors.BOLD}{Colors.MAGENTA}🔎 Detailed Results:{Colors.RESET}")
                            for i, row in enumerate(records, 1):
                                print(f"  {Colors.MAGENTA}{i}. {row}{Colors.RESET}")
                            print()
                    except Exception as e:
                        print(f"{Colors.RED}❌ Error displaying results: {e}{Colors.RESET}")
                        if not records:
                            print("No results returned for the Cypher query.")
                        else:
                            print(f"Query returned {len(records)} rows.")
                            for i, row in enumerate(records, 1):
                                print(f"Row {i}: {row}")

                    # Generate answer
                    context = format_records_as_context(records)
                    prompt = build_answer_prompt(
                        question=user_input,
                        context=context,
                        conversation_context=history,
                        known_entities=extracted_entities,
                    )

                    answer_start = time.time()
                    answer = generate_answer(
                        client, prompt, model=args.model, temperature=args.temperature
                    )
                    answer_time = time.time() - answer_start

                    total_time = time.time() - start_time

                    # Show bot response
                    print(
                        format_bot_message(
                            answer, datetime.now().strftime("%H:%M:%S"), total_time
                        )
                    )

                    # Show performance info (optional)
                    if total_time > 2.0:  # Only show if slow
                        print(f"\n{Colors.BOLD}{Colors.CYAN}⏱️  Performance Breakdown:{Colors.RESET}")
                        print(f"{Colors.CYAN}  Entity processing: {entity_processing_time:.1f}s{Colors.RESET}")
                        print(f"{Colors.CYAN}  Cypher generation: {cypher_time:.1f}s{Colors.RESET}")
                        print(f"{Colors.CYAN}  Query execution: {query_time:.1f}s{Colors.RESET}")
                        print(f"{Colors.CYAN}  Answer generation: {answer_time:.1f}s{Colors.RESET}")
                        print(f"{Colors.CYAN}  Total time: {total_time:.1f}s{Colors.RESET}")

                    # Extract entities from the answer for context building
                    try:
                        # Try to extract entities from the answer using the same prompt
                        entity_prompt = f"Extract entities and relationships from this text: {answer}"
                        entity_output = call_gemini(
                            client, entity_prompt, model=args.model, temperature=0.1
                        )
                        new_entities, new_relationships = parse_llm_output(
                            entity_output
                        )

                        # Update extracted entities and relationships
                        extracted_entities.update(new_entities)
                        extracted_relationships.update(
                            [f"({s}, {r}, {o})" for s, r, o in new_relationships]
                        )

                        # Update active_entity using priority logic (only if we don't have one or found a better one)
                        if new_entities:
                            try:
                                best_entity = prioritize_entities(list(new_entities), known_entities)
                                if best_entity and (not active_entity or is_real_entity(best_entity, known_entities)):
                                    old_active_entity = active_entity
                                    active_entity = best_entity
                                    # Update in history if changed
                                    if old_active_entity != active_entity:
                                        history_data["active_entity"] = active_entity
                                        save_history_to_file(history_data)
                                    logger.info(f"Active entity updated from answer entities: {active_entity}")
                            except Exception as e:
                                logger.debug(f"Error prioritizing entities: {e}")

                        # Show extracted entities if any new ones found
                        if new_entities:
                            print(
                                f"{Colors.YELLOW}📝 Extracted entities: {', '.join(new_entities[:5])}{'...' if len(new_entities) > 5 else ''}{Colors.RESET}"
                            )

                    except Exception as e:
                        # Silently handle entity extraction errors
                        logger.debug(f"Could not extract entities from answer: {e}")
                        pass

                    # Update history with entities context
                    entities_info = ""
                    if extracted_entities:
                        entities_info = (
                            f"Entities: {', '.join(list(extracted_entities)[:10])}\n"
                        )

                    # Save to persistent storage
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    add_conversation_to_history(
                        history_data,
                        user_input,
                        answer,
                        list(extracted_entities) if extracted_entities else [],
                        timestamp,
                    )
                    
                    # Persist active_entity to history
                    history_data["active_entity"] = active_entity
                    save_history_to_file(history_data)

                    # Update in-memory history for context
                    history += f"Q: {user_input}\nA: {answer}\n{entities_info}"

                except Exception as e:
                    error_msg = (
                        f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn: {str(e)}"
                    )
                    print(format_error_message(error_msg))
                    logger.error(f"Error processing question: {e}")

                print()  # Add spacing between conversations

            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}⚠️  Interrupted by user{Colors.RESET}")
                print_goodbye()
                break
            except EOFError:
                print_goodbye()
                break

        sys.exit(0)

    # Load input text either from --file or positional text (skip for interactive)
    text = None
    if args.mode != "interactive":
        if args.file:
            try:
                with open(args.file, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except Exception as e:
                logger.error(f"Failed to read file: {e}")
                sys.exit(1)
        elif args.text:
            text = args.text
        else:
            logger.error("Provide text or --file")
            parser.print_help()
            sys.exit(1)

    client = init_genai_client()

    if args.mode == "extract":
        assert text is not None, "Text must be provided for extract mode"
        prompt = PROMPT_TEMPLATE + f'\nText: "{text}"'
        logger.info("Calling Gemini for entity extraction...")
        output = call_gemini(
            client, prompt, model=args.model, temperature=args.temperature
        )

        print("\n----- RAW LLM OUTPUT -----\n")
        print(output)

        entities, relationships = parse_llm_output(output)

        print("\n----- PARSED ENTITIES -----\n")
        for e in entities:
            print(f"- {e}")

        print("\n----- PARSED RELATIONSHIPS -----\n")
        for s, r, o in relationships:
            print(f"- ({s}, {r}, {o})")
    elif args.mode == "query":
        assert text is not None, "Text must be provided for query mode"
        # mode == 'query': generate Cypher via LLM using schema, then run on Neo4j
        t0 = time.time()
        driver = get_neo4j_driver_from_env()
        schema = fetch_schema_summary(driver)
        t1 = time.time()
        logger.info(f"Fetched schema in {t1 - t0:.2f}s")

        t0 = time.time()
        cypher = generate_cypher_via_llm(
            client,
            schema=schema,
            question=text,
            model=args.model,
            temperature=args.temperature,
        )
        t1 = time.time()
        logger.info(f"Generated Cypher in {t1 - t0:.2f}s")

        print("\n----- GENERATED CYPHER -----\n")
        print(cypher)

        t0 = time.time()
        try:
            records = run_cypher(driver, cypher)
        except Exception as e:
            logger.error(f"Error running Cypher: {e}")
            raise
        t1 = time.time()
        logger.info(f"Executed Cypher in {t1 - t0:.2f}s")

        print("\n----- QUERY RESULTS -----\n")
        if not records:
            print("(no results)")
        else:
            # Simple pretty print rows
            for i, row in enumerate(records, 1):
                print(f"Row {i}: {row}")
    else:
        # mode == 'chat': optimized chat with unified fuzzy matching
        assert text is not None, "Text must be provided for chat mode"
        driver = get_neo4j_driver_from_env()
        schema = fetch_schema_summary(driver)
        
        # Initialize optimized services
        cache_manager = EntityCacheManager()
        entity_cache = cache_manager.get_entities(driver)
        fuzzy_service = FuzzyMatchingService(entity_cache)
        query_processor = QueryProcessor(client, entity_cache)
        
        print(f"{Colors.CYAN}📊 Loaded {len(entity_cache.known_entities)} entities for processing{Colors.RESET}")
        
        # Process query with optimized services
        llm_result = query_processor.process_entity_identification(
            question=text,
            model=args.model
        )
        
        # Enhanced query with fuzzy matching
        enhanced_query, fuzzy_matches = fuzzy_service.enhance_query(
            text, 
            llm_result.extracted_entities,
            confidence_threshold=70.0
        )
        
        if fuzzy_matches:
            print(f"{Colors.CYAN}🔍 Fuzzy Matching Results:{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*40}{Colors.RESET}")
            
            for match in fuzzy_matches:
                color = Colors.GREEN if match.confidence > 70 else Colors.YELLOW if match.confidence > 50 else Colors.RED
                print(f"{color}  '{match.query_entity}' → '{match.match_name}' (confidence: {match.confidence:.1f}%){Colors.RESET}")
            
            if enhanced_query != text:
                good_matches = [m for m in fuzzy_matches if m.confidence > 70]
                for match in good_matches:
                    print(f"{Colors.CYAN}🔄 Replaced '{match.query_entity}' with '{match.match_name}'{Colors.RESET}")
            
            print(f"{Colors.CYAN}{'-'*40}{Colors.RESET}")
        
        cypher = generate_cypher_via_llm(
            client,
            schema=schema,
            question=enhanced_query,
            model=args.model,
            temperature=args.temperature,
        )

        print("\n----- GENERATED CYPHER -----\n")
        print(cypher)

        try:
            records = run_cypher(driver, cypher)
        except Exception as e:
            logger.error(f"Error running Cypher: {e}")
            print("Xin lỗi, đã xảy ra lỗi khi truy vấn cơ sở dữ liệu.")
            return

        context = format_records_as_context(records)
        prompt = build_answer_prompt(
            question=enhanced_query, context=context, conversation_context=args.history
        )
        answer = generate_answer(
            client, prompt, model=args.model, temperature=args.temperature
        )

        print("\n----- ANSWER -----\n")
        print(answer)


if __name__ == "__main__":
    main()
