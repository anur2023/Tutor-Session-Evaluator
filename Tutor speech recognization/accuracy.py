import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import re
from typing import Dict, Tuple, List
import warnings
import torch 
warnings.filterwarnings("ignore")

class ProtocolComplianceChecker:
    def __init__(self, protocol_file: str = "./protocol_cleaned_file.json"):
        """
        Initialize the compliance checker with protocol guidelines
        
        Args:
            protocol_file: Path to JSON file containing protocol phrases
        """
        self.protocol_file = protocol_file
        self.protocol_guidelines = self._load_protocols()
        self.sbert_model = SentenceTransformer("all-mpnet-base-v2")
        self.protocol_embeddings = self._create_embeddings()
        self.category_weights = self._get_category_weights()
        self.thresholds = {
            'exact_match': 100,
            'high_compliance': 70,
            'medium_compliance': 40,
            'low_compliance': 20
        }

    def _load_protocols(self) -> Dict:
        """Load and validate protocol guidelines"""
        try:
            with open(self.protocol_file, "r", encoding="utf-8") as file:
                protocols = json.load(file)
            
            # Validate protocol structure
            if not isinstance(protocols, dict):
                raise ValueError("Protocol file should contain a dictionary of categories")
                
            for category, phrases in protocols.items():
                if not isinstance(phrases, list):
                    raise ValueError(f"Category '{category}' should contain a list of phrases")
                    
            return protocols
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Protocol file {self.protocol_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.protocol_file}")

    def _create_embeddings(self) -> Dict:
        """Create SBERT embeddings for all protocol phrases"""
        return {
            category: self.sbert_model.encode(phrases, convert_to_tensor=True)
            for category, phrases in self.protocol_guidelines.items()
        }

    def _get_category_weights(self) -> Dict:
        """Define importance weights for all protocol categories"""
        weights = {
            # Communication categories
            "Communication - Greeting the Student": 0.10,
            "Communication - Tutor Misbehavior": -0.20,  # Negative weight for bad behavior
            "Communication - Off-Topic Conversations": -0.15,
            
            # Teaching - Problem Identification
            "Teaching - Confirming the Problem": 0.08,
            "Teaching - Identifying the Student's Need": 0.08,
            "Teaching - Identifying Student's Requirement": 0.08,
            
            # Teaching - Concept Explanation
            "Teaching - Identifying Core Concepts": 0.10,
            "Teaching - Checking Student's Understanding of the Core Concept": 0.10,
            "Teaching - Revising or Teaching the Concept if Required": 0.10,
            "Teaching - Explaining the Concept or Procedure": 0.12,
            "Teaching - Visualizing the Concepts While Teaching": 0.05,
            
            # Teaching - Problem Solving
            "Teaching - Solving the Problem Step-by-Step": 0.15,
            "Teaching - Confirming Student's Understanding of Steps": 0.10,
            "Teaching - Encouraging Student to Solve Independently": 0.10,
            
            # Teaching - Conclusion
            "Teaching - Giving the Final Solution": 0.08,
            "Teaching - Confirming Student's Understanding at the End": 0.08,
            
            # Feedback
            "Teaching - Asking Student to Rate the Session": 0.05,
            "Teaching - Asking Student to Mark as Favourite Tutor": 0.03
        }
        
        # Verify all weights sum to ~1.0 (accounting for negative weights)
        total_positive = sum(w for w in weights.values() if w > 0)
        total_negative = sum(w for w in weights.values() if w < 0)
        
        if not 0.95 <= total_positive <= 1.05:  # Allow small rounding differences
            print(f"Warning: Positive weights sum to {total_positive} (should be ~1.0)")
        
        return weights

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for comparison"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text

    def _calculate_similarity(self, text: str, category: str) -> float:
        """Calculate semantic similarity score (0-100)"""
        text_embedding = self.sbert_model.encode(text, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(text_embedding, self.protocol_embeddings[category])
        max_similarity = torch.max(similarities).item()  # Get the maximum similarity value
        return round(max_similarity * 100, 2)  # Convert to percentage

    def _get_feedback(self, category: str, score: float) -> Dict:
        """Generate detailed feedback for a category"""
        feedback = {
            'score': score,
            'suggestions': [],
            'examples': []
        }
        
        if score >= self.thresholds['high_compliance']:
            feedback['verdict'] = "Excellent protocol compliance"
            feedback['suggestions'] = ["Keep using these effective phrases"]
        elif score >= self.thresholds['medium_compliance']:
            feedback['verdict'] = "Moderate compliance - could improve"
            feedback['suggestions'] = [
                "Try incorporating more protocol phrases",
                "Be more explicit in following guidelines"
            ]
        else:
            feedback['verdict'] = "Needs significant improvement"
            feedback['suggestions'] = [
                "Review the protocol guidelines",
                "Practice using recommended phrases"
            ]
        
        # Add example phrases (top 3 from protocol)
        feedback['examples'] = self.protocol_guidelines[category][:3]
        
        return feedback

    def _detect_negative_patterns(self, text: str) -> Dict:
        """Check for protocol violations"""
        violations = defaultdict(list)
        clean_text = self._preprocess_text(text)
        
        # Check for misbehavior phrases
        for phrase in self.protocol_guidelines.get("Communication - Tutor Misbehavior", []):
            if phrase in clean_text:
                violations["misbehavior"].append(phrase)
                
        # Check for off-topic conversations
        for phrase in self.protocol_guidelines.get("Communication - Off-Topic Conversations", []):
            if phrase in clean_text:
                violations["off_topic"].append(phrase)
                
        return dict(violations)

    def analyze_text(self, text: str) -> Dict:
        """
        Comprehensive analysis of text against protocols
        
        Returns:
            {
                "overall_score": float,
                "category_scores": dict,
                "feedback": dict,
                "violations": dict,
                "top_categories": list,
                "bottom_categories": list
            }
        """
        if not text.strip():
            raise ValueError("Empty input text provided")
            
        results = {
            "category_scores": {},
            "feedback": {},
            "violations": self._detect_negative_patterns(text),
            "text_metrics": {
                "length": len(text.split()),
                "unique_words": len(set(text.split()))
            }
        }

        clean_text = self._preprocess_text(text)
        
        # Calculate scores for each category
        for category in self.protocol_guidelines:
            # First check for exact matches
            exact_match = any(
                self._preprocess_text(phrase) in clean_text
                for phrase in self.protocol_guidelines[category]
            )
            
            if exact_match:
                results["category_scores"][category] = self.thresholds['exact_match']
            else:
                results["category_scores"][category] = self._calculate_similarity(clean_text, category)
            
            # Generate feedback
            results["feedback"][category] = self._get_feedback(
                category, results["category_scores"][category]
            )

        # Calculate weighted overall score
        total_weight = sum(abs(w) for w in self.category_weights.values())  # Use abs for negative weights
        weighted_sum = sum(
            score * self.category_weights[category]
            for category, score in results["category_scores"].items()
            if category in self.category_weights
        )
        results["overall_score"] = max(0, min(100, round(weighted_sum / total_weight, 2)))
        
        # Identify top and bottom performing categories
        sorted_cats = sorted(
            results["category_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        results["top_categories"] = sorted_cats[:3]
        results["bottom_categories"] = sorted_cats[-3:]
        
        return results

    def generate_report(self, analysis_results: Dict) -> str:
        """Generate human-readable report"""
        report = []
        report.append(f"PROTOCOL COMPLIANCE REPORT")
        report.append(f"Overall Score: {analysis_results['overall_score']}/100")
        report.append(f"Text Length: {analysis_results['text_metrics']['length']} words")
        report.append("\nTOP PERFORMING CATEGORIES:")
        
        for cat, score in analysis_results["top_categories"]:
            report.append(f"- {cat}: {score}/100")
            
        report.append("\nNEEDS IMPROVEMENT:")
        for cat, score in analysis_results["bottom_categories"]:
            report.append(f"- {cat}: {score}/100")
            
        if analysis_results["violations"]:
            report.append("\nPROTOCOL VIOLATIONS DETECTED:")
            for violation_type, phrases in analysis_results["violations"].items():
                report.append(f"- {violation_type}:")
                for phrase in phrases[:3]:  # Show max 3 examples
                    report.append(f"  * '{phrase}'")
        
        report.append("\nDETAILED FEEDBACK:")
        for category, feedback in analysis_results["feedback"].items():
            if feedback['score'] < self.thresholds['medium_compliance']:
                report.append(f"\n{category} ({feedback['score']}/100):")
                report.append(f"Verdict: {feedback['verdict']}")
                report.append("Suggestions:")
                for suggestion in feedback['suggestions']:
                    report.append(f"- {suggestion}")
                report.append("Example phrases:")
                for example in feedback['examples']:
                    report.append(f"* '{example}'")
        
        return "\n".join(report)


# if __name__ == "__main__":
#     try:
#         # Initialize the compliance checker
#         checker = ProtocolComplianceChecker()
        
#         # Path to your text file - change this to your actual file path
#         text_file_path = "./hinglish_transcription_cleaned.txt"
        
#         # Read the text file
#         try:
#             with open(text_file_path, "r", encoding="utf-8") as file:
#                 transcript_text = file.read()
                
#             if not transcript_text.strip():
#                 raise ValueError("The text file is empty")
                
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Text file not found at {text_file_path}")
#         except UnicodeDecodeError:
#             raise UnicodeDecodeError("Could not read file - please ensure it's a text file with UTF-8 encoding")
        
#         # Analyze the transcript
#         analysis = checker.analyze_text(transcript_text)
        
#         # Generate and print the report
#         report = checker.generate_report(analysis)
#         print(report)
        
#         # Save results to JSON file (same name as input with _results.json suffix)
#         output_file = text_file_path.replace(".txt", "_results.json")
#         try:
#             with open(output_file, "w", encoding="utf-8") as f:
#                 json.dump(analysis, f, indent=2)
#             print(f"\nResults saved to {output_file}")
#         except IOError as e:
#             print(f"\nWarning: Could not save results - {str(e)}")
        
#         # Optionally save report to text file
#         report_file = text_file_path.replace(".txt", "_report.txt")
#         try:
#             with open(report_file, "w", encoding="utf-8") as f:
#                 f.write(report)
#             print(f"Full report saved to {report_file}")
#         except IOError as e:
#             print(f"Warning: Could not save report - {str(e)}")
            
#     except Exception as e:
#         print(f"\nError during analysis: {str(e)}")
#         print("Please check:")
#         print("- The input file exists and is accessible")
#         print("- The file contains valid text (not binary data)")
#         print("- You have proper read/write permissions")