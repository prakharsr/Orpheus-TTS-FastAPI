import re
import logging

logger = logging.getLogger(__name__)

def split_text_into_sentences(text):
    """
    Split text into sentences while preserving dialogue integrity and narrative flow.
    
    This function handles:
    - Dialogue preservation (keeping quotes together)
    - Dialogue attribution (keeping "he said" with dialogue)
    - Paragraph boundaries
    - Complex punctuation within dialogue
    - Reasonable chunk sizes for TTS processing
    """
    # First, split by paragraphs to maintain document structure
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    all_segments = []
    
    for paragraph in paragraphs:
        segments = _split_paragraph_intelligently(paragraph)
        all_segments.extend(segments)
    
    # Post-process to ensure reasonable sizes and combine very short segments
    return _combine_short_segments(all_segments)

def _split_paragraph_intelligently(paragraph):
    """Split a single paragraph while preserving dialogue and narrative flow."""
    # If paragraph is short enough, return as-is
    if len(paragraph) <= 200:
        return [paragraph]
    
    # Find all dialogue blocks (text within quotes)
    dialogue_pattern = r'"([^"]*)"'
    dialogues = list(re.finditer(dialogue_pattern, paragraph))
    
    segments = []
    last_end = 0
    
    for dialogue_match in dialogues:
        # Get text before this dialogue
        before_dialogue = paragraph[last_end:dialogue_match.start()].strip()
        
        # Get the dialogue with quotes
        dialogue_text = dialogue_match.group(0)
        dialogue_start = dialogue_match.start()
        dialogue_end = dialogue_match.end()
        
        # Look for attribution after the dialogue
        # Common attribution patterns: "he said", "she cried", "Jack whispered", etc.
        attribution_pattern = r'^\s*([A-Z][a-zA-Z]*\.?\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*(?:\s+(?:said|cried|whispered|shouted|asked|replied|continued|added|muttered|declared|announced|exclaimed|sobbed|laughed|sighed|nodded|shook|clutched|tugged|looked|turned|moved|went|came|walked|ran|stood|sat|knelt|rose|smiled|frowned|gasped|breathed|swallowed|attempted|tried|began|finished)(?:\s+[a-zA-Z]+)*)?)'
        
        # Look ahead for attribution (up to 150 characters)
        text_after_dialogue = paragraph[dialogue_end:dialogue_end + 150]
        attribution_match = re.match(attribution_pattern, text_after_dialogue)
        
        # Look for attribution before the dialogue (within last 100 characters of before_dialogue)
        before_attribution_pattern = r'([A-Z][a-zA-Z]*\.?\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s*$'
        before_attribution_match = None
        if before_dialogue:
            before_attribution_match = re.search(before_attribution_pattern, before_dialogue[-100:])
        
        # Process text before dialogue
        if before_dialogue:
            if before_attribution_match:
                # Split the text, keeping attribution with dialogue
                attribution_start_in_before = before_dialogue.rfind(before_attribution_match.group(1))
                pre_attribution_text = before_dialogue[:attribution_start_in_before].strip()
                
                if pre_attribution_text:
                    # Split the pre-attribution text if it's long
                    pre_segments = _split_long_text(pre_attribution_text)
                    segments.extend(pre_segments)
            else:
                # No attribution before, split the before_dialogue text normally
                before_segments = _split_long_text(before_dialogue)
                segments.extend(before_segments)
        
        # Create the dialogue segment
        dialogue_segment = ""
        
        # Add preceding attribution if found
        if before_attribution_match:
            dialogue_segment += before_attribution_match.group(1) + " "
        
        # Add the dialogue
        dialogue_segment += dialogue_text
        
        # Add following attribution if found
        if attribution_match:
            dialogue_segment += " " + attribution_match.group(1).strip()
            last_end = dialogue_end + attribution_match.end()
        else:
            last_end = dialogue_end
        
        segments.append(dialogue_segment.strip())
    
    # Handle any remaining text after the last dialogue
    remaining_text = paragraph[last_end:].strip()
    if remaining_text:
        remaining_segments = _split_long_text(remaining_text)
        segments.extend(remaining_segments)
    
    # If no dialogues were found, just split the paragraph normally
    if not dialogues:
        return _split_long_text(paragraph)
    
    return segments

def _split_long_text(text, max_length=400):
    """Split long text on sentence boundaries, preserving meaning."""
    if len(text) <= max_length:
        return [text.strip()] if text.strip() else []
    
    # Split on sentence endings, but be careful with abbreviations
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences into reasonable chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence would exceed max_length
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def _combine_short_segments(segments, min_length=40, max_length=500):
    """Combine very short segments while keeping segments under max_length."""
    if not segments:
        return []
    
    combined = []
    current_segment = ""
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # If adding this segment would exceed max_length, finalize current segment
        if current_segment and len(current_segment) + len(segment) + 1 > max_length:
            combined.append(current_segment)
            current_segment = segment
        # If current segment is too short, try to combine with next
        elif len(current_segment) < min_length and len(current_segment) + len(segment) + 1 <= max_length:
            if current_segment:
                current_segment += " " + segment
            else:
                current_segment = segment
        # If the current segment is good length, add it and start new
        else:
            if current_segment:
                combined.append(current_segment)
            current_segment = segment
    
    # Add the last segment
    if current_segment:
        combined.append(current_segment)
    
    return combined

def create_batches(sentences, max_batch_chars=500):
    """Create batches by combining sentences up to max_batch_chars"""
    batches = []
    current_batch = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the batch size, start a new batch
        if len(current_batch) + len(sentence) > max_batch_chars and current_batch:
            batches.append(current_batch)
            current_batch = sentence
        else:
            # Add separator space if needed
            if current_batch:
                current_batch += " "
            current_batch += sentence
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    # logger.info(f"Created {len(batches)} batches from {len(sentences)} sentences")
    return batches 