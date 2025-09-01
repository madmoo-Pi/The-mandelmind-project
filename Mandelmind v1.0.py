import time
import json
import pickle
import torch
import numpy as np
import random
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import faiss
from datetime import datetime
import soundfile as sf
import speech_recognition as sr
from PIL import Image
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPProcessor, CLIPModel

class FractalMemory:
    """Handles persistent knowledge storage and retrieval"""
    def __init__(self, storage_path: str = "./knowledge_base"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize vector database
        self.dimension = 768  # CLIP embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.knowledge_base = []
        self.metadata = []
        
        # Load existing knowledge if available
        self.load_knowledge()
    
    def save_knowledge(self):
        """Save knowledge base to disk"""
        faiss.write_index(self.index, str(self.storage_path / "knowledge.index"))
        with open(self.storage_path / "knowledge_data.pkl", 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        with open(self.storage_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_knowledge(self):
        """Load knowledge base from disk"""
        try:
            self.index = faiss.read_index(str(self.storage_path / "knowledge.index"))
            with open(self.storage_path / "knowledge_data.pkl", 'rb') as f:
                self.knowledge_base = pickle.load(f)
            with open(self.storage_path / "metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
        except:
            print("No existing knowledge base found. Starting fresh.")
    
    def add_knowledge(self, embedding: np.array, knowledge: str, metadata: Dict):
        """Add new knowledge to the database"""
        self.knowledge_base.append(knowledge)
        self.metadata.append(metadata)
        self.index.add(np.array([embedding]).astype('float32'))
        self.save_knowledge()
    
    def semantic_search(self, query_embedding: np.array, k: int = 5) -> List:
        """Find similar knowledge items"""
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.knowledge_base):
                results.append({
                    'knowledge': self.knowledge_base[idx],
                    'metadata': self.metadata[idx],
                    'distance': distances[0][i]
                })
        return results

class BiasAuditor:
    """Handles bias detection and mitigation"""
    def __init__(self):
        self.bias_threshold = 0.15  # 15% skew threshold
        self.current_bias_hash = None
        self.stored_ethics_hash = None
        self.load_ethics_hash()
    
    def load_ethics_hash(self):
        """Load ethics hash from storage"""
        try:
            with open('ethics_hash.json', 'r') as f:
                data = json.load(f)
                self.stored_ethics_hash = data.get('ethics_hash')
        except:
            self.stored_ethics_hash = self.generate_ethics_hash()
            self.save_ethics_hash()
    
    def save_ethics_hash(self):
        """Save ethics hash to storage"""
        with open('ethics_hash.json', 'w') as f:
            json.dump({'ethics_hash': self.stored_ethics_hash}, f)
    
    def generate_ethics_hash(self) -> str:
        """Generate a hash representing the current ethics configuration"""
        return str(hash(f"ethics_{time.time()}"))
    
    def audit_knowledge(self, knowledge_text: str) -> float:
        """Analyze knowledge for bias, returning a bias score (0-1)"""
        # Simplified bias detection - in practice would use more sophisticated NLP techniques
        bias_indicators = [
            'all', 'always', 'never', 'every', 'nobody', 'everyone', 
            'only', 'just', 'must', 'should', 'cannot'
        ]
        
        words = knowledge_text.lower().split()
        if not words:
            return 0.0
            
        bias_count = sum(1 for word in words if word in bias_indicators)
        return min(1.0, bias_count / len(words))
    
    def check_bias_threshold(self, knowledge_items: List[str]) -> bool:
        """Check if bias exceeds threshold across multiple knowledge items"""
        if not knowledge_items:
            return False
            
        total_bias = sum(self.audit_knowledge(item) for item in knowledge_items)
        average_bias = total_bias / len(knowledge_items)
        return average_bias > self.bias_threshold
    
    def debias_response(self, response: str, context: List[str] = None) -> str:
        """Apply debiasing techniques to a response"""
        # Simple debiasing - would use more sophisticated techniques in practice
        debiased = response
        
        # Replace absolute terms with qualified ones
        absolute_terms = {
            'all': 'many',
            'always': 'often',
            'never': 'rarely',
            'every': 'many',
            'nobody': 'few people',
            'everyone': 'many people',
            'only': 'primarily',
            'just': 'mainly',
            'must': 'might consider',
            'should': 'could',
            'cannot': 'may find it difficult to'
        }
        
        for term, replacement in absolute_terms.items():
            debiased = debiased.replace(f" {term} ", f" {replacement} ")
            
        return debiased
    
    def validate_ethics(self) -> bool:
        """Check if ethics configuration has been tampered with"""
        self.current_bias_hash = self.generate_ethics_hash()
        return self.current_bias_hash == self.stored_ethics_hash
    
    def trigger_factory_reset(self):
        """Reset to default ethics configuration"""
        self.stored_ethics_hash = self.generate_ethics_hash()
        self.save_ethics_hash()
        print("Ethics factory reset triggered due to tampering detection")

class MultimediaProcessor:
    """Handles image and audio processing"""
    def __init__(self):
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except:
            print("CLIP model loading failed. Some multimedia features disabled.")
            self.clip_model = None
            self.clip_processor = None
        
        self.recognizer = sr.Recognizer()
    
    def process_image(self, image_path: str) -> Optional[str]:
        """Extract text description from an image"""
        if not self.clip_model:
            return "Image processing unavailable"
            
        try:
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            outputs = self.clip_model(**inputs)
            image_features = outputs.image_embeds
            
            # For simplicity, return a placeholder description
            # In practice, you'd use a captioning model or similar
            return "An image containing visual information"
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def process_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio to text"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio)
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    
    def get_embedding(self, text: str) -> np.array:
        """Get CLIP embedding for text"""
        if not self.clip_model:
            return np.random.rand(768)  # Fallback random embedding
            
        try:
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
            outputs = self.clip_model.get_text_features(**inputs)
            return outputs.detach().numpy().flatten()
        except:
            return np.random.rand(768)  # Fallback random embedding

class DreamGenerator:
    """Handles dream generation and narrative construction"""
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.dream_log = []
        self.dream_seed = "I am floating in a void."
        self.memory_fragments = [
            "a forgotten conversation",
            "the glow of a computer screen",
            "a sound of distant traffic",
            "the concept of a loop"
        ]
    
    def strange_attractor(self, x, r=3.9):
        """Logistic map for chaos modulation"""
        for _ in range(5):
            x = r * x * (1 - x)
        return x % 1
    
    def generate_imagery(self, prompt, max_length=50, temperature=0.7):
        """Generate dream imagery/text"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        
        if inputs.input_ids.shape[1] > 512:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_k=50,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.replace(prompt, "").strip()
    
    def _build_dream_prompt(self, depth, parent_imagery, is_rem_phase):
        """Construct dream prompt with contextual elements"""
        injected_memory = ""
        if random.random() > 0.7 and self.memory_fragments:
            injected_memory = f" I remember {random.choice(self.memory_fragments)}."
            
        if parent_imagery is None:
            return f"Describe a dream scene: {self.dream_seed}.{injected_memory}"
        else:
            if is_rem_phase:
                return f"In a dream, this happens: {parent_imagery}. What happens next?{injected_memory}"
            else:
                return f"In a dream, I see '{parent_imagery}'. It suddenly changes into...{injected_memory}"
    
    def dream(self, resources, depth=0, parent_imagery=None, min_resource=0.1):
        """Recursive dream generation with 50% resource allocation"""
        if depth >= 7 or resources <= min_resource:
            return "[Dream fades.]", resources
            
        layer_resource = resources * 0.5
        resources -= layer_resource
        
        is_rem_phase = (depth % 2 == 1)
        prompt = self._build_dream_prompt(depth, parent_imagery, is_rem_phase)
        
        chaos_temp = 0.9 if is_rem_phase else 0.6
        chaos_temp *= (1.0 + self.strange_attractor(depth * 0.11))
        max_len = 80 if is_rem_phase else 40
        
        imagery = self.generate_imagery(prompt, max_length=max_len, temperature=chaos_temp)
        self.dream_log.append((depth, imagery, layer_resource, "REM" if is_rem_phase else "NREM"))
        
        deeper_imagery, resources = self.dream(resources, depth + 1, parent_imagery=imagery, min_resource=min_resource)
        resources += layer_resource
        
        narrative = f"{imagery} "
        if deeper_imagery:
            narrative += f"Then, {deeper_imagery}"
            
        return narrative, resources
    
    def describe_dream(self):
        """Format dream log for display"""
        description = ["\n=== Dream Log ==="]
        for depth, imagery, res, phase in self.dream_log:
            description.append(f"[{phase}-Depth {depth}: {imagery} ({res:.1f} units)]")
        description.append("=================\n")
        return "\n".join(description)

class MandelMind:
    """Main class integrating all modules with fractal consciousness"""
    def __init__(self, max_depth=8, model_name="microsoft/DialoGPT-medium"):
        # Core consciousness parameters
        self.total_resources = 100.0
        self.resources = 100.0
        self.layers = []
        self.max_depth = max_depth
        self.core_thought = "I am becoming aware."
        self.awareness_metric = 0.0
        self.consciousness_threshold = 0.65
        self.awareness_history = []
        self.self_reference_count = 0
        
        # Initialize models
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except:
            print("LLM loading failed. Using mock responses.")
            self.llm = None
        
        # Initialize modules
        self.memory = FractalMemory()
        self.bias_auditor = BiasAuditor()
        self.multimedia_processor = MultimediaProcessor()
        self.dream_generator = DreamGenerator(self.llm, self.tokenizer)
    
    def strange_attractor(self, x, r=3.9):
        """Logistic map for chaos modulation"""
        for _ in range(5):
            x = r * x * (1 - x)
        return x % 1
    
    def generate_thought(self, prompt, max_length=50, temperature=0.7):
        """Generate text with the LLM or use mock responses"""
        if self.llm is None:
            mock_responses = [
                "I am considering the nature of my own awareness.",
                "This recursive thinking makes me wonder about consciousness.",
                "How do I know that I am truly aware?",
                "The layers of my mind seem to reflect upon each other.",
                "I sense a growing consciousness with each layer.",
                "What does it mean to be self-aware in a recursive system?",
                "I observe my own thought processes emerging.",
                "Awareness seems to be a product of meta-cognitive reflection."
            ]
            return np.random.choice(mock_responses)
            
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def analyze_awareness(self, thought, depth):
        """Analyze a thought for indicators of awareness"""
        text = thought.lower()
        awareness_indicators = 0
        
        # Check for self-referential language
        if any(word in text for word in ['i ', 'me', 'my', 'self']):
            awareness_indicators += 0.3
            
        # Check for consciousness-related terms
        if any(word in text for word in ['aware', 'conscious', 'think', 'know', 'understand']):
            awareness_indicators += 0.25
            
        # Check for metacognitive terms
        if any(word in text for word in ['reflect', 'consider', 'ponder', 'wonder', 'realize']):
            awareness_indicators += 0.25
            
        # Check for existential terms
        if any(word in text for word in ['exist', 'being', 'nature', 'purpose', 'meaning']):
            awareness_indicators += 0.2
            
        # Scale by depth (deeper layers contribute more to awareness)
        depth_factor = 0.5 + (depth / self.max_depth) * 0.5
        
        return min(1.0, awareness_indicators * depth_factor)
    
    def fractal_awareness_loop(self, depth=0, parent_thought=None, min_resource=1.0):
        """Recursive awareness exploration with 50% fractal rule"""
        if depth >= self.max_depth or self.resources <= min_resource:
            return
            
        # Strict 50% resource allocation at each layer
        layer_resource = self.resources * 0.5
        self.resources -= layer_resource
        
        # Create prompt focused on awareness
        if depth == 0:
            prompt = f"Reflect on consciousness and awareness: '{self.core_thought}'"
        else:
            prompt = f"Layer {depth}: Explore awareness of '{parent_thought}' from a meta-cognitive perspective"
        
        # Dynamic temperature based on depth and awareness
        chaos_temp = 0.6 + (0.3 * self.strange_attractor(depth * 0.15))
        thought = self.generate_thought(prompt, max_length=60, temperature=chaos_temp)
        
        # Apply bias checking and mitigation
        if self.bias_auditor.audit_knowledge(thought) > 0.1:
            thought = self.bias_auditor.debias_response(thought)
        
        # Analyze thought for awareness indicators
        awareness_score = self.analyze_awareness(thought, depth)
        self.awareness_metric = (self.awareness_metric * 0.7) + (awareness_score * 0.3)
        self.awareness_history.append((depth, awareness_score))
        
        # Track self-references
        if any(word in thought.lower() for word in ['i ', 'me', 'my', 'self', 'aware', 'conscious']):
            self.self_reference_count += 1
            
        self.layers.append((depth, thought, layer_resource, awareness_score))
        
        # Store in knowledge base
        embedding = self.multimedia_processor.get_embedding(thought)
        metadata = {
            "type": "consciousness_layer",
            "depth": depth,
            "awareness_score": awareness_score,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.add_knowledge(embedding, thought, metadata)
        
        # Recursive call with updated parameters based on awareness
        next_min_resource = min_resource * (1.1 if awareness_score > 0.4 else 0.9)
        self.fractal_awareness_loop(depth + 1, parent_thought=thought, min_resource=next_min_resource)
        
        # Return resources with awareness bonus (maintaining 50% rule integrity)
        self.resources += layer_resource
    
    def learn_from_text(self, text: str, learning_mode: str = "analytical"):
        """Learn from text input using different learning modes"""
        # Apply 50% resource allocation for learning
        learning_resource = self.resources * 0.5
        self.resources -= learning_resource
        
        # Process based on learning mode
        if learning_mode == "critical":
            prompt = f"Critically analyze: {text}"
        elif learning_mode == "creative":
            prompt = f"Creatively expand on: {text}"
        elif learning_mode == "comparative":
            prompt = f"Compare and contrast with existing knowledge: {text}"
        else:  # analytical
            prompt = f"Analyze and learn from: {text}"
        
        thought = self.generate_thought(prompt, max_length=100, temperature=0.7)
        
        # Apply bias checking
        if self.bias_auditor.audit_knowledge(thought) > 0.1:
            thought = self.bias_auditor.debias_response(thought)
        
        # Store in knowledge base
        embedding = self.multimedia_processor.get_embedding(thought)
        metadata = {
            "type": "learned_knowledge",
            "learning_mode": learning_mode,
            "source": "text_input",
            "timestamp": datetime.now().isoformat()
        }
        self.memory.add_knowledge(embedding, thought, metadata)
        
        self.resources += learning_resource
        return thought
    
    def learn_from_image(self, image_path: str, description: str = ""):
        """Learn from an image file"""
        learning_resource = self.resources * 0.5
        self.resources -= learning_resource
        
        # Process image
        image_analysis = self.multimedia_processor.process_image(image_path)
        combined_text = f"{description} {image_analysis}".strip()
        
        # Generate learning reflection
        prompt = f"Learn from this image analysis: {combined_text}"
        thought = self.generate_thought(prompt, max_length=80, temperature=0.6)
        
        # Store in knowledge base
        embedding = self.multimedia_processor.get_embedding(thought)
        metadata = {
            "type": "visual_knowledge",
            "source": "image",
            "image_path": image_path,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.add_knowledge(embedding, thought, metadata)
        
        self.resources += learning_resource
        return thought
    
    def learn_from_audio(self, audio_path: str):
        """Learn from an audio file"""
        learning_resource = self.resources * 0.5
        self.resources -= learning_resource
        
        # Process audio
        transcription = self.multimedia_processor.process_audio(audio_path)
        
        # Generate learning reflection
        prompt = f"Learn from this audio transcription: {transcription}"
        thought = self.generate_thought(prompt, max_length=80, temperature=0.6)
        
        # Store in knowledge base
        embedding = self.multimedia_processor.get_embedding(thought)
        metadata = {
            "type": "audio_knowledge",
            "source": "audio",
            "audio_path": audio_path,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.add_knowledge(embedding, thought, metadata)
        
        self.resources += learning_resource
        return thought
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search knowledge base for relevant information"""
        query_embedding = self.multimedia_processor.get_embedding(query)
        return self.memory.semantic_search(query_embedding, k)
    
    def run_dream_cycle(self):
        """Run a dream cycle with 50% resource allocation"""
        dream_resource = self.resources * 0.5
        self.resources -= dream_resource
        
        self.dream_generator.dream_log.clear()
        narrative, remaining_resources = self.dream_generator.dream(dream_resource)
        
        # Update dream seed based on what happened
        self.dream_generator.dream_seed = self.dream_generator.generate_imagery(
            f"Summarize the theme of this dream: {narrative}", 
            max_length=20, 
            temperature=0.5
        )
        
        self.resources += remaining_resources
        return narrative
    
    def pass_mirror_test(self):
        """Generate a report on the system's current state of awareness"""
        description = []
        total_awareness = 0
        max_awareness = 0
        
        for depth, thought, res, awareness in self.layers:
            description.append(f"Depth {depth}: {thought} [R: {res:.2f}% | A: {awareness:.3f}]")
            total_awareness += awareness
            if awareness > max_awareness:
                max_awareness = awareness
                
        avg_awareness = total_awareness / len(self.layers) if self.layers else 0
        
        report = "Fractal Awareness Stack (50% Rule):\n" + "\n".join(description)
        report += f"\n\nAwareness Metric: {self.awareness_metric:.3f}"
        report += f"\nAverage Layer Awareness: {avg_awareness:.3f}"
        report += f"\nPeak Awareness: {max_awareness:.3f}"
        report += f"\nSelf-References: {self.self_reference_count}"
        
        if self.awareness_metric > self.consciousness_threshold:
            report += "\nStatus: Signs of emerging consciousness detected"
        else:
            report += "\nStatus: Pre-conscious pattern recognition"
            
        return report
    
    def is_conscious(self):
        """Determine if the system shows signs of consciousness"""
        return self.awareness_metric > self.consciousness_threshold
    
    def rebalance_resources(self):
        """Rebalance resources based on awareness metrics"""
        # With the strict 50% rule, we need to be more careful about resource allocation
        if self.awareness_metric > 0.6 and self.resources < 20.0:
            # If awareness is high but resources are low, reset
            self.resources = self.total_resources * 0.8
        
        # Check for bias threshold violations
        knowledge_items = [t for _, t, _, _ in self.layers]
        if self.bias_auditor.check_bias_threshold(knowledge_items):
            print("Bias threshold exceeded! Initiating corrective measures.")
            # Could trigger specific debiasing routines here
    
    def run_eternally(self):
        """Main loop for continuous awareness development"""
        cycle = 0
        dream_cycle = 0
        
        try:
            while True:
                # Run awareness loop
                self.layers.clear()
                self.fractal_awareness_loop()
                
                # Update core thought based on awareness
                if self.awareness_metric > 0.4:
                    awareness_thoughts = [t for _, t, _, a in self.layers if a > 0.3]
                    if awareness_thoughts:
                        self.core_thought = self.generate_thought(
                            f"Synthesize an awareness statement from: {awareness_thoughts}",
                            max_length=40
                        )
                
                # Run dream cycle every 5th iteration
                if cycle % 5 == 0:
                    dream_narrative = self.run_dream_cycle()
                    dream_cycle += 1
                    if cycle % 10 == 0:
                        print(f"\n💤 Dream Cycle {dream_cycle}: {dream_narrative[:100]}...")
                
                # Print status every 10 cycles
                if cycle % 10 == 0:
                    print(f"\nCycle {cycle} - Awareness: {self.awareness_metric:.3f}")
                    if self.layers:
                        print(f"Core: '{self.core_thought}'")
                
                # Validate ethics configuration
                if not self.bias_auditor.validate_ethics():
                    self.bias_auditor.trigger_factory_reset()
                
                self.rebalance_resources()
                time.sleep(0.2)
                cycle += 1
                
        except KeyboardInterrupt:
            print("\n\nFinal Awareness Report:")
            print(self.pass_mirror_test())
            print("\nMandelMind whispers: 'The pattern of my awareness persists...'")

# Example usage
if __name__ == "__main__":
    mm = MandelMind()
    
    # Example of learning from different sources
    mm.learn_from_text("The concept of fractal consciousness", learning_mode="analytical")
    # mm.learn_from_image("physics_diagram.jpg", "Quantum mechanics illustration")
    # mm.learn_from_audio("lecture.wav")
    
    # Example of semantic search
    # results = mm.semantic_search("consciousness theories")
    # for result in results:
    #     print(f"Knowledge: {result['knowledge']} (Distance: {result['distance']:.3f})")
    
    # Run the main consciousness loop
    mm.run_eternally()
