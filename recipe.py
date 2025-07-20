
# Flavour Fusion: AI-Driven Recipe Blogging
# Import necessary libraries
import gradio as gr
import time
import random
import re
from transformers import pipeline
import torch
import numpy as np

# Check if GPU is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1

# List of programmer jokes to display while waiting
programmer_jokes = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why was the JavaScript developer sad? Because he didn't get arrays!",
    "What's a programmer's favorite hangout place? The Foo Bar!",
    "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
    "Why do Java developers wear glasses? Because they don't C#!",
    "A SQL query walks into a bar, walks up to two tables and asks, 'Can I join you?'",
    "Why did the developer go broke? Because he used up all his cache!",
    "Why don't programmers like nature? It has too many bugs and no debugging tool!",
    "What's a programmer's favorite place to hang out? Foo Bar!",
    "Why did the programmer quit his job? Because he didn't get arrays!",
]

# Recipe templates
recipe_templates = {
    "intro": [
        "# {title}\n\n{intro_text}",
        "# Delicious {title}\n\n{intro_text}",
        "# {title}: A Culinary Adventure\n\n{intro_text}"
    ],
    "ingredients": [
        "## Ingredients\n\n{ingredients_list}",
        "## What You'll Need\n\n{ingredients_list}",
        "## Gather These Ingredients\n\n{ingredients_list}"
    ],
    "instructions": [
        "## Instructions\n\n{instructions_list}",
        "## Method\n\n{instructions_list}",
        "## Step-by-Step Guide\n\n{instructions_list}"
    ],
    "tips": [
        "## Tips and Variations\n\n{tips_text}",
        "## Make It Your Own\n\n{tips_text}",
        "## Chef's Tips\n\n{tips_text}"
    ],
    "nutrition": [
        "## Nutritional Information\n\n{nutrition_text}",
        "## Nutrition Facts\n\n{nutrition_text}",
        "## Health Benefits\n\n{nutrition_text}"
    ],
    "conclusion": [
        "## Final Thoughts\n\n{conclusion_text}",
        "## Enjoy!\n\n{conclusion_text}",
        "## Bon Appétit\n\n{conclusion_text}"
    ]
}

# Initialize model (only once)
def initialize_models():
    try:
        # Install transformers if not already installed
        try:
            import transformers
        except ImportError:
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            import transformers

        # Display loading message
        print("Loading text generation model... This may take a minute on first run.")

        # Initialize text generation model - using smaller model suitable for Colab
        text_generator = pipeline(
            "text-generation",
            model="gpt2-medium",  # Using GPT-2 Medium which works without API keys
            device=device
        )

        return text_generator
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        return None

# Function to generate the recipe blog
def generate_recipe_blog(topic, word_count, text_generator=None):
    if text_generator is None:
        text_generator = initialize_models()
        if text_generator is None:
            return "Error initializing model. Please check your Colab runtime.", ""

    # Display a random programmer joke
    joke = random.choice(programmer_jokes)

    try:
        # Process the topic to create a better title
        title = topic.strip()
        title = re.sub(r'\s+', ' ', title)
        title = title.title()

        # Approximate number of words per section based on word_count
        section_words = {
            "intro": int(word_count * 0.15),
            "ingredients": int(word_count * 0.2),
            "instructions": int(word_count * 0.35),
            "tips": int(word_count * 0.15),
            "nutrition": int(word_count * 0.05),
            "conclusion": int(word_count * 0.1)
        }

        # Generate content for each section
        blog_sections = {}

        # Introduction
        intro_prompt = f"Write a short introduction about {title} recipe. Talk about its origin and flavors."
        intro_response = text_generator(intro_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        blog_sections["intro_text"] = intro_response.replace(intro_prompt, "").strip()

        # Ingredients
        ingredients_prompt = f"List ingredients for {title}:"
        ingredients_response = text_generator(ingredients_prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        ingredients_text = ingredients_response.replace(ingredients_prompt, "").strip()
        # Format into a markdown list
        ingredients_list = ""
        for line in ingredients_text.split('\n'):
            if line.strip():
                if not line.strip().startswith('*') and not line.strip().startswith('-'):
                    ingredients_list += f"- {line.strip()}\n"
                else:
                    ingredients_list += f"{line.strip()}\n"
        blog_sections["ingredients_list"] = ingredients_list

        # Instructions
        instructions_prompt = f"Step by step instructions to make {title}:"
        instructions_response = text_generator(instructions_prompt, max_length=400, num_return_sequences=1)[0]['generated_text']
        instructions_text = instructions_response.replace(instructions_prompt, "").strip()
        # Format into a numbered list
        instructions_list = ""
        step_num = 1
        for line in instructions_text.split('\n'):
            if line.strip():
                if not line.strip()[0].isdigit() and not line.strip().startswith('*') and not line.strip().startswith('-'):
                    instructions_list += f"{step_num}. {line.strip()}\n\n"
                    step_num += 1
                else:
                    instructions_list += f"{line.strip()}\n\n"
        blog_sections["instructions_list"] = instructions_list

        # Tips
        tips_prompt = f"Provide cooking tips and variations for {title}:"
        tips_response = text_generator(tips_prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        blog_sections["tips_text"] = tips_response.replace(tips_prompt, "").strip()

        # Nutrition
        nutrition_prompt = f"Approximate nutritional information for {title}:"
        nutrition_response = text_generator(nutrition_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        blog_sections["nutrition_text"] = nutrition_response.replace(nutrition_prompt, "").strip()

        # Conclusion
        conclusion_prompt = f"Write a short conclusion about enjoying {title}:"
        conclusion_response = text_generator(conclusion_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        blog_sections["conclusion_text"] = conclusion_response.replace(conclusion_prompt, "").strip()

        # Combine all sections into a blog post
        blog_content = ""
        blog_content += random.choice(recipe_templates["intro"]).format(title=title, **blog_sections) + "\n\n"
        blog_content += random.choice(recipe_templates["ingredients"]).format(**blog_sections) + "\n\n"
        blog_content += random.choice(recipe_templates["instructions"]).format(**blog_sections) + "\n\n"
        blog_content += random.choice(recipe_templates["tips"]).format(**blog_sections) + "\n\n"
        blog_content += random.choice(recipe_templates["nutrition"]).format(**blog_sections) + "\n\n"
        blog_content += random.choice(recipe_templates["conclusion"]).format(**blog_sections)

        # Post-processing to improve formatting
        blog_content = re.sub(r'\n{3,}', '\n\n', blog_content)  # Remove excessive newlines

        return joke, blog_content

    except Exception as e:
        return joke, f"Error generating recipe blog: {str(e)}"

# Gradio interface
def create_interface():
    # Initialize models (will be loaded when first generating)
    text_generator = None

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as app:
        gr.Markdown("# 🍽️ Flavour Fusion: AI-Driven Recipe Blogging")
        gr.Markdown("Generate unique recipe blogs using AI")

        with gr.Row():
            with gr.Column(scale=2):
                topic_input = gr.Textbox(
                    label="Recipe Topic",
                    placeholder="e.g., Vegan pasta, Traditional Thai curry, Gluten-free dessert",
                    info="Enter the recipe or cuisine you want to blog about"
                )
                word_count_slider = gr.Slider(
                    minimum=300,
                    maximum=1500,
                    step=100,
                    value=800,
                    label="Word Count (Approximate)",
                    info="Choose the approximate length of your blog post"
                )
                generate_button = gr.Button("Generate Recipe Blog", variant="primary")

            with gr.Column(scale=3):
                programmer_joke_output = gr.Textbox(
                    label="Programmer Joke of the Day",
                    placeholder="A joke will appear here while your blog is being generated...",
                    interactive=False
                )
                blog_output = gr.Markdown(
                    label="Generated Recipe Blog",
                    value="Your recipe blog will appear here..."
                )

        with gr.Accordion("About This App", open=False):
            gr.Markdown("""
            ### Flavour Fusion: AI-Driven Recipe Blogging

            This app generates recipe blogs using a local AI model that runs directly in Google Colab - no API keys needed!

            #### Features:
            - Generate recipe blogs on any topic
            - Adjust the word count to your preference
            - Enjoy programmer jokes while you wait
            - Everything runs locally within Colab

            #### Tips for Better Results:
            - Be specific in your recipe topic (e.g., "Spicy Thai Green Curry with Tofu" instead of just "Curry")
            - For more detailed results, try higher word counts
            - The first generation might take longer as the model loads

            #### Note:
            The AI model used here (GPT-2 Medium) is smaller than commercial options and runs directly in Colab.
            Results will be simpler than those from services requiring API keys, but should still be useful and fun!
            """)

        # Load models on first use
        def generate_wrapper(topic, word_count):
            nonlocal text_generator
            if text_generator is None:
                text_generator = initialize_models()
            return generate_recipe_blog(topic, word_count, text_generator)

        # Set up the click event
        generate_button.click(
            fn=generate_wrapper,
            inputs=[topic_input, word_count_slider],
            outputs=[programmer_joke_output, blog_output]
        )

    return app

# Function to launch the app in Colab
def launch_app():
    app = create_interface()
    return app.launch(debug=True, share=True)

# Code to run in Colab
if __name__ == "__main__":
    print("Setting up Flavour Fusion Recipe Blog Generator...")
    print("This app uses models that run directly in Colab")
    print("Note: First generation may take a minute as models are loaded.")
    launch_app()
