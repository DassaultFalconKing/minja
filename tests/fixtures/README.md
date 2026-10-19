# Gemma 4 regression fixture

`gemma4-26-heretic-chat_template.jinja` is an unchanged byte copy of the local
production model's `chat_template.jinja` at:

`C:/llm/models/runtime/gemma4-26-heretic-google-current/chat_template.jinja`

SHA-256: `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`

This records the tested on-disk artifact; its own header describes its origin.
The tests do not assert that every Gemma 4 template has identical behavior.
The fixture is shared by the adjacent-string parser regression and the default
chat-template polyfill regression.
