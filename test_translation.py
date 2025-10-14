"""
Translation Test Script
Tests the deep-translator functionality to ensure Hindi translation works
"""

from deep_translator import GoogleTranslator

# Test translations
test_phrases = [
    "Welcome to the Plant Disease Recognition System!",
    "Upload an image of a plant",
    "Analyzing the image...",
    "Primary prediction",
    "Suggested Treatment",
    "Apply fungicides and rotate crops.",
    "No action needed; your plant is healthy!"
]

print("Testing English to Hindi translations:\n")
print("=" * 60)

for phrase in test_phrases:
    try:
        translator = GoogleTranslator(source='en', target='hi')
        hindi = translator.translate(phrase)
        print(f"\nEnglish: {phrase}")
        print(f"Hindi:   {hindi}")
    except Exception as e:
        print(f"\nEnglish: {phrase}")
        print(f"ERROR:   {e}")

print("\n" + "=" * 60)
print("\n✅ Translation test complete!")
print("\nIf you see Hindi text (Devanagari script) above, translation is working!")
print("If you see errors, there might be a network issue or API problem.")
