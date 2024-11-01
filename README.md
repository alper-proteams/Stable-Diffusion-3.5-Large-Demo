# Stable Diffusion 3.5 Large Image Generation

This project demonstrates a simple implementation to generate images using the Stable Diffusion 3.5 Large model. The script processes long prompts, optimizes them, and generates high-quality images.

## Requirements

- Python 3.7+
- PyTorch
- diffusers
- transformers
- spaCy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/stable-diffusion-3.5-large.git
    cd stable-diffusion-3.5-large
    ```

2. Install the required packages:
    ```sh
    pip install torch diffusers transformers spacy
    ```

3. Download the spaCy language model:
    ```sh
    python -m spacy download en_core_web_sm
    ```

## Usage

1. Prepare your prompt in a text file or directly in the script.

2. Run the script to generate an image:
    ```sh
    python generate_image_long_prompt_support.py
    ```

3. The generated image will be saved in the specified output path.

## Example

Here is an example prompt used in the script:
```python
long_prompt = """
Eerie, gothic book cover for Morella by Edgar Allan Poe. A ghostly woman with flowing dark hair and a piercing gaze stands in a dim, candle-lit room filled with shadows. Faded books and ancient symbols surround her, hinting at occult knowledge. Her translucent figure fades into darkness, symbolizing mystery and death. Title in an ornate, haunting typeface.
"""
```

The script will process this prompt, optimize it, and generate an image saved as `morella.png`.

## Notes

- This is a basic implementation intended for testing the new Stable Diffusion 3.5 Large model.
- The script includes enhanced prompt processing to ensure high-quality image generation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
