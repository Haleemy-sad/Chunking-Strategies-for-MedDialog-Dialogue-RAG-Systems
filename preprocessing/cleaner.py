"""
Text Preprocessing Module
Applies consistent cleaning to all medical dialogues
"""

import re
from typing import List, Dict, Any


class TextCleaner:
    """Preprocess medical dialogue text"""
    
    def __init__(self, preserve_medical_terms: bool = True):
        """
        Initialize the text cleaner
        
        Args:
            preserve_medical_terms: Whether to preserve medical terminology
        """
        self.preserve_medical_terms = preserve_medical_terms
        
        # Common medical abbreviations to preserve
        self.medical_abbrev = {
            'ECG', 'EKG', 'MRI', 'CT', 'BP', 'HR', 'IV', 'IM', 'SC',
            'mg', 'ml', 'mcg', 'kg', 'lb', 'cm', 'mm',
            'BID', 'TID', 'QID', 'PRN', 'PO', 'NPO',
            'COPD', 'CHF', 'MI', 'CVA', 'TIA', 'DM', 'HTN'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Preserve medical abbreviations by temporarily replacing them
        preserved = {}
        if self.preserve_medical_terms:
            for idx, abbrev in enumerate(self.medical_abbrev):
                if abbrev in text:
                    placeholder = f"__MEDABBREV{idx}__"
                    preserved[placeholder] = abbrev
                    text = text.replace(abbrev, placeholder)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-°/]', '', text)
        
        # Lowercase
        text = text.lower()
        
        # Restore medical abbreviations
        for placeholder, abbrev in preserved.items():
            text = text.replace(placeholder.lower(), abbrev)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def clean_utterance(self, utterance: Dict[str, str]) -> Dict[str, str]:
        """
        Clean a single utterance (speaker + text)
        
        Args:
            utterance: Dictionary with 'speaker' and 'text' keys
            
        Returns:
            Cleaned utterance dictionary
        """
        return {
            'speaker': utterance['speaker'].lower(),
            'text': self.clean_text(utterance['text'])
        }
    
    def clean_dialogue(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean all utterances in a dialogue
        
        Args:
            dialogue: Dialogue dictionary
            
        Returns:
            Dialogue with cleaned utterances
        """
        cleaned_dialogue = dialogue.copy()
        cleaned_dialogue['utterances'] = [
            self.clean_utterance(utt) for utt in dialogue['utterances']
        ]
        return cleaned_dialogue
    
    def clean_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean a list of dialogues
        
        Args:
            dialogues: List of dialogue dictionaries
            
        Returns:
            List of cleaned dialogues
        """
        return [self.clean_dialogue(d) for d in dialogues]
    
    def dialogue_to_text(self, dialogue: Dict[str, Any], 
                        include_speakers: bool = True) -> str:
        """
        Convert a dialogue to a single text string
        
        Args:
            dialogue: Dialogue dictionary
            include_speakers: Whether to include speaker labels
            
        Returns:
            Dialogue as text
        """
        texts = []
        for utt in dialogue['utterances']:
            if include_speakers:
                texts.append(f"{utt['speaker']}: {utt['text']}")
            else:
                texts.append(utt['text'])
        
        return " ".join(texts)
    
    def format_for_chunking(self, dialogue: Dict[str, Any]) -> str:
        """
        Format dialogue for chunking (preserves structure)
        
        Args:
            dialogue: Dialogue dictionary
            
        Returns:
            Formatted dialogue text
        """
        lines = []
        lines.append(f"[Dialogue ID: {dialogue['dialogue_id']}]")
        lines.append(f"[Specialty: {dialogue['specialty']}]")
        lines.append("")
        
        for idx, utt in enumerate(dialogue['utterances'], 1):
            speaker = utt['speaker'].capitalize()
            lines.append(f"Turn {idx} - {speaker}: {utt['text']}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the cleaner
    cleaner = TextCleaner()
    
    test_text = "Patient has    ELEVATED   BP (140/90).  ECG shows NORMAL sinus rhythm!!!"
    cleaned = cleaner.clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    
    test_utterance = {
        'speaker': 'Doctor',
        'text': 'I recommend   an MRI  scan   and blood work.'
    }
    cleaned_utt = cleaner.clean_utterance(test_utterance)
    print(f"\nOriginal utterance: {test_utterance}")
    print(f"Cleaned utterance: {cleaned_utt}")
