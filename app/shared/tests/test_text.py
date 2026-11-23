"""Model reply text, whatever shape the client returns it in."""

from app.shared.text import content_text


def test_a_plain_string_is_returned_as_is():
    assert content_text("Hello.") == "Hello."


def test_gemini_text_blocks_join_without_extra_spaces():
    # langchain-google-genai 4.x returns Gemini 3 replies as content blocks,
    # sometimes split mid-sentence, with the thought signature alongside.
    content = [
        {"type": "text", "text": "Your trip is ", "extras": {"signature": "abc"}},
        {"type": "text", "text": "ready."},
    ]

    assert content_text(content) == "Your trip is ready."


def test_blocks_without_text_are_skipped():
    content = [{"type": "thinking", "thinking": "private"}, "Done."]

    assert content_text(content) == "Done."


def test_no_content_is_empty_text():
    assert content_text(None) == ""
