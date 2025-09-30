"""Streamlit entrypoint shim.

Streamlit Cloud can be configured to run this file as the app. It simply imports
the existing `mai.py` Streamlit app so we don't duplicate UI code.
"""

import mai  # noqa: F401


if __name__ == "__main__":
    # When run locally with `python streamlit_app.py` nothing special is needed;
    # Streamlit executes the file and importing `mai` runs the app.
    pass
