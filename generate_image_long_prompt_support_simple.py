import torch
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig
from diffusers import SD3Transformer2DModel
from transformers import AutoTokenizer
import textwrap
import spacy
from collections import defaultdict


def load_nlp():
    """
    Load spaCy model for natural language processing.
    Downloads the model if not already present.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        print("Downloading required language model...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")


def classify_prompt_parts(parts):
    """
    Intelligently classify prompt parts using NLP.
    Returns categorized parts based on their semantic meaning.
    """
    nlp = load_nlp()

    # Categories with their typical patterns
    categories = {
        'subject': {
            'pos': ['NOUN', 'PROPN'],  # Main subject nouns
            'deps': ['nsubj', 'dobj', 'pobj'],  # Subject dependencies
        },
        'environment': {
            'keywords': ['in', 'at', 'on', 'background', 'setting', 'scene'],
            'pos': ['ADP'],  # Prepositions often indicate setting
        },
        'technical': {
            'keywords': ['mm', 'lens', 'f/', 'resolution', 'exposure', 'aperture',
                         'iso', 'camera', 'photo', 'photography', 'shot'],
        },
        'style': {
            'keywords': ['style', 'mood', 'aesthetic', 'looking', 'like', 'effect',
                         'grain', 'color', 'tone', 'quality', 'cinematic', 'dramatic'],
        }
    }

    # Initialize categories
    classified_parts = defaultdict(list)

    for part in parts:
        doc = nlp(part.lower())
        scores = defaultdict(int)

        # Score each part for each category
        for category, patterns in categories.items():
            # Check for keywords
            if 'keywords' in patterns:
                for keyword in patterns['keywords']:
                    if keyword in part.lower():
                        scores[category] += 1

            # Check for parts of speech
            if 'pos' in patterns:
                for token in doc:
                    if token.pos_ in patterns['pos']:
                        scores[category] += 1

            # Check for dependencies
            if 'deps' in patterns:
                for token in doc:
                    if token.dep_ in patterns['deps']:
                        scores[category] += 1

        # Assign to category with highest score, default to style if no clear winner
        if scores:
            max_category = max(scores.items(), key=lambda x: x[1])[0]
            classified_parts[max_category].append(part)
        else:
            classified_parts['style'].append(part)

    return dict(classified_parts)


def create_weighted_prompt(prompt, max_length=77):
    """
    Create a weighted prompt that fits within token limits while preserving
    the most important elements using NLP-based classification.
    """
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Split into parts
    parts = [p.strip() for p in prompt.split(',')]

    # Classify parts using NLP
    categorized_parts = classify_prompt_parts(parts)

    # Priority order for categories
    category_priority = ['subject', 'style', 'environment', 'technical']

    # Helper function to add parts while tracking tokens
    def add_parts(parts_list, remaining_tokens):
        result = []
        tokens_used = 0
        for part in parts_list:
            tokens = len(tokenizer.encode(part))
            if tokens_used + tokens + 1 <= remaining_tokens:  # +1 for comma
                result.append(part)
                tokens_used += tokens + 1
            else:
                break
        return result, tokens_used

    # Construct final prompt based on priority
    final_parts = []
    remaining_tokens = max_length

    # Add parts in priority order
    for category in category_priority:
        if category in categorized_parts:
            added_parts, tokens = add_parts(categorized_parts[category], remaining_tokens)
            if added_parts:
                final_parts.extend(added_parts)
                remaining_tokens -= tokens

    return ", ".join(final_parts)


def setup_pipeline():
    """
    Set up the Stable Diffusion pipeline with optimized settings.
    """
    model_id = "stabilityai/stable-diffusion-3.5-large"

    # Configure 4-bit quantization
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the model with quantization
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    # Create the pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        torch_dtype=torch.bfloat16
    )

    # Enable CPU offloading
    pipeline.enable_model_cpu_offload()

    return pipeline


def generate_image_with_long_prompt(prompt, output_path="output.png", debug=True):
    """
    Generate an image using a long prompt by creating a weighted version
    that fits within token limits.
    """
    pipeline = setup_pipeline()

    # Create optimized prompt
    optimized_prompt = create_weighted_prompt(prompt)

    if debug:
        print("\nOriginal prompt length:", len(prompt.split(',')))
        print("Optimized prompt length:", len(optimized_prompt.split(',')))
        print("\nOptimized prompt:\n", optimized_prompt, "\n")

    # Generate image
    image = pipeline(
        prompt=optimized_prompt,
        num_inference_steps=28,
        guidance_scale=4.5,
    ).images[0]

    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return image


# Example usage
if __name__ == "__main__":
    # You can now use any prompt! Here's an example:
    long_prompt = """
    Beautiful red sports car speeding on a mountain road at sunset,
    dramatic cliff edge with ocean view, California coastal highway,
    motion blur on wheels, sleek aerodynamic design with carbon fiber details,
    golden hour lighting, lens flare, reflections on pristine paint,
    professional automotive photography, 70mm lens, f/4,
    high dynamic range, ultra-sharp focus on car body,
    cinematic atmosphere, dramatic clouds, Kodak vision3 look, 8k quality
    """

    # Clean up the prompt
    long_prompt = " ".join(long_prompt.split())

    # Generate the image
    generate_image_with_long_prompt(long_prompt, "sports_car.png")