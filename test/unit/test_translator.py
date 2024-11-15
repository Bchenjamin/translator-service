import src.translator
from src.translator import translate_content
from unittest.mock import patch
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def eval_single_response_translation(expected_answer: str, llm_response: str) -> float:
  embeddings_expected = model.encode(expected_answer)
  embeddings_response = model.encode(llm_response)
  similarity = model.similarity(embeddings_expected, embeddings_response).item()
  return similarity

# Unit tests (adapted from the collab notebook)
def test_translate_french_message():
    is_english, translated_content = translate_content("Je voudrais un café, s'il vous plaît.")
    assert is_english == False, "Expected the content to be recognized as non-English"
    similarity_score = eval_single_response_translation("I would like a coffee, please.", translated_content)
    assert similarity_score >= 0.9, f"Similarity score: {similarity_score}"

def test_translate_japanese_message():
    is_english, translated_content = translate_content("これは非常に興味深い質問です。")
    assert is_english == False, "Expected the content to be recognized as non-English"
    similarity_score = eval_single_response_translation("This is a very interesting question.", translated_content)
    assert similarity_score >= 0.9, f"Similarity score: {similarity_score}"

def test_translate_german_message():
    is_english, translated_content = translate_content("Ich liebe es, neue Sprachen zu lernen.")
    assert is_english == False, "Expected the content to be recognized as non-English"
    similarity_score = eval_single_response_translation("I love learning new languages", translated_content)
    assert similarity_score >= 0.9, f"Similarity score: {similarity_score}"

def test_translate_chinese_message():
    is_english, translated_content = translate_content("你今天过得怎么样")
    assert is_english == False, "Expected the content to be recognized as non-English"
    similarity_score = eval_single_response_translation("How was your day today?", translated_content)
    assert similarity_score >= 0.9, f"Similarity score: {similarity_score}"

def test_translate_english_message():
    is_english, translated_content = translate_content("This is an English message!")
    assert is_english == True, "Expected the content to be recognized as English"
    similarity_score = eval_single_response_translation("This is an English message!", translated_content)
    assert similarity_score >= 0.9, f"Similarity score: {similarity_score}"

def test_translate_non_string_message():
    is_english, translated_content = translate_content(1)
    assert is_english == False
    assert translated_content == "Content is not a string"

def test_translate_empty_message():
    is_english, translated_content = translate_content("")
    assert is_english == False
    assert translated_content == "Content is an empty string"

def test_translate_gibberish_message():
    is_english, translated_content = translate_content("asdfawefasdf")
    assert is_english == False
    assert translated_content == "Translation failed"


# Integration Tests
@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_normal_response(mocker):
    mocker.return_value.choices[0].message.content = "It's a beautiful day for a walk"
    is_english, translated_content = translate_content("C'est une belle journée pour une promenade.")
    assert is_english == False, "Expected the content to be recognized as non-English"
    similarity_score = eval_single_response_translation("It's a beautiful day for a walk.", translated_content)
    assert similarity_score >= 0.9, f"Similarity score: {similarity_score}"

    mocker.return_value.choices[0].message.content = "This is a Chinese message."
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False, "Expected the content to be recognized as non-English"
    similarity_score = eval_single_response_translation("This is a Chinese message.", translated_content)
    assert similarity_score >= 0.9, f"Similarity score: {similarity_score}"


@patch.object(src.translator.client.chat.completions, 'create')
def test_llm_gibberish_response(mocker):
    mocker.return_value.choices[0].message.content = "afwe;lkfjawef"
    is_english, translated_content = translate_content("Test")
    assert is_english == False, "Expected the content to be not recognized as non-English"
    similarity_score = eval_single_response_translation("afwe;lkfjawef", translated_content)
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"

    mocker.return_value.choices[0].message.content = "1000"
    is_english, translated_content = translate_content("Test")
    assert is_english == False, "Expected the content to be not recognized as non-English"
    similarity_score = eval_single_response_translation("1000", translated_content)
    assert similarity_score >= 0.9, f"Similarity is actually {similarity_score}"



        
