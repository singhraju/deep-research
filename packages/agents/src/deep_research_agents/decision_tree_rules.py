"""
Decision Tree Rule Engine for CoC AI Analyst.

This module loads and manages decision tree rules from YAML configuration,
providing rule lookup and formatting for LLM-based recommendation generation.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    from deep_research_utils.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class DecisionTreeRuleEngine:
    """
    Load and manage decision tree rules from YAML configuration.
    
    This engine loads rules from a YAML file and provides methods to retrieve
    and format rules for use in LLM-based recommendation generation.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Example:
        >>> engine = DecisionTreeRuleEngine("configs/decision_tree_rules.yaml")
        >>> rules = engine.get_rules_by_category("IP MedSurg (DNE)")
        >>> formatted = engine.format_rules_for_llm()
    """
    
    def __init__(self, yaml_path: str):
        """
        Initialize rule engine from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
        """
        self.yaml_path = Path(yaml_path)
        self.rules = self._load_rules_from_yaml()
        rule_count = self.get_total_rule_count()
        category_count = len(self.rules)
        logger.info(f"Loaded {rule_count} rules from {category_count} categories")
    
    def _load_rules_from_yaml(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse YAML file into structured format.
        
        Returns:
            Dictionary mapping category names to lists of rules
        """
        if not self.yaml_path.exists():
            logger.warning(f"Decision tree YAML not found: {self.yaml_path}")
            return {}
        
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not isinstance(data, dict):
                logger.error("Invalid YAML format: root element is not a dictionary")
                return {}
            
            service_categories = data.get('service_categories', {})
            
            if not isinstance(service_categories, dict):
                logger.error("Invalid YAML format: service_categories is not a dictionary")
                return {}
            
            logger.debug(f"Loaded {len(service_categories)} service categories")
            return service_categories
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load decision tree YAML: {e}")
            return {}
    
    def get_rules_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Retrieve rules for specific service category.
        
        Args:
            category: Service category name (e.g., "IP MedSurg (DNE)")
            
        Returns:
            List of rules for the category, or empty list if not found
        """
        return self.rules.get(category, [])
    
    def get_all_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return all rules across all categories.
        
        Returns:
            Dictionary mapping category names to lists of rules
        """
        return self.rules
    
    def get_total_rule_count(self) -> int:
        """
        Count total number of rules across all categories.
        
        Returns:
            Total number of rules
        """
        return sum(len(rules) for rules in self.rules.values())
    
    def get_category_names(self) -> List[str]:
        """
        Get list of all service category names.
        
        Returns:
            List of category names
        """
        return list(self.rules.keys())
    
    def format_rules_for_llm(self, categories: Optional[List[str]] = None) -> str:
        """
        Convert rules to LLM-friendly text format.
        
        Args:
            categories: Optional list of specific categories to format.
                       If None, formats all categories.
        
        Returns:
            Formatted text representation of rules
        """
        if categories:
            rules_to_format = {cat: self.rules.get(cat, []) for cat in categories if cat in self.rules}
        else:
            rules_to_format = self.rules
        
        if not rules_to_format:
            return "No decision tree rules available."
        
        formatted = []
        formatted.append("DECISION TREE RULES BY SERVICE CATEGORY:")
        formatted.append("=" * 80)
        
        for category, rules in rules_to_format.items():
            formatted.append(f"\n### {category}")
            formatted.append("-" * 80)
            
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                
                rule_type = rule.get('rule_type', 'unknown')
                
                if rule_type == 'trend_category':
                    trend_id = rule.get('trend_id', 'N/A')
                    formatted.append(f"\n[TREND {trend_id}]")
                else:
                    parent_id = rule.get('parent_trend_id', 'N/A')
                    formatted.append(f"\n  [Detail under Trend {parent_id}]")
                
                research = rule.get('research_considerations')
                if research:
                    formatted.append(f"    Research: {research}")
                
                why = rule.get('why')
                if why:
                    formatted.append(f"    Why: {why}")
                
                suggestions = rule.get('cost_of_care_suggestions')
                if suggestions:
                    formatted.append(f"    Suggestion: {suggestions}")
                
                flags = rule.get('flags', {})
                if isinstance(flags, dict):
                    flag_list = []
                    if flags.get('clinical'):
                        flag_list.append('Clinical')
                    if flags.get('network'):
                        flag_list.append('Network')
                    if flags.get('ops'):
                        flag_list.append('Ops')
                    if flag_list:
                        formatted.append(f"    Flags: {', '.join(flag_list)}")
        
        formatted.append("\n" + "=" * 80)
        return "\n".join(formatted)
    
    def format_rules_compact(self, categories: Optional[List[str]] = None) -> str:
        """
        Convert rules to compact LLM-friendly format (shorter version).
        
        Args:
            categories: Optional list of specific categories to format.
                       If None, formats all categories.
        
        Returns:
            Compact formatted text representation of rules
        """
        if categories:
            rules_to_format = {cat: self.rules.get(cat, []) for cat in categories if cat in self.rules}
        else:
            rules_to_format = self.rules
        
        if not rules_to_format:
            return "No decision tree rules available."
        
        formatted = []
        
        for category, rules in rules_to_format.items():
            formatted.append(f"\n{category}:")
            
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                
                rule_type = rule.get('rule_type', 'unknown')
                research = rule.get('research_considerations')
                why = rule.get('why')
                suggestions = rule.get('cost_of_care_suggestions')
                
                # Only include rules with cost suggestions (mainly detailed rules)
                if suggestions:
                    line_parts = []
                    if research:
                        line_parts.append(f"RESEARCH: {research}")
                    if why:
                        line_parts.append(f"WHY: {why}")
                    line_parts.append(f"SUGGESTION: {suggestions}")
                    
                    formatted.append(f"  - {' | '.join(line_parts)}")
        
        return "\n".join(formatted)


if __name__ == "__main__":
    # Test the rule engine
    import sys
    from pathlib import Path
    
    # Try to load from default location
    yaml_path = Path(__file__).resolve().parents[4] / "configs" / "decision_tree_rules.yaml"
    
    if not yaml_path.exists():
        print(f"YAML file not found: {yaml_path}")
        sys.exit(1)
    
    print(f"Loading rules from: {yaml_path}")
    engine = DecisionTreeRuleEngine(str(yaml_path))
    
    print(f"\nTotal rules: {engine.get_total_rule_count()}")
    print(f"Categories: {len(engine.get_category_names())}")
    print(f"\nCategory names:")
    for cat in engine.get_category_names():
        rule_count = len(engine.get_rules_by_category(cat))
        print(f"  - {cat}: {rule_count} rules")
    
    print("\n" + "=" * 80)
    print("Sample formatted output (first category):")
    print("=" * 80)
    first_category = engine.get_category_names()[0] if engine.get_category_names() else None
    if first_category:
        print(engine.format_rules_for_llm(categories=[first_category]))
