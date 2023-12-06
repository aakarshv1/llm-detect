"""Welcome to Reflex! This file outlines the steps to create a basic app."""
from rxconfig import config

from model.classify import classify

import reflex as rx

docs_url = "https://reflex.dev/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"


class State(rx.State):
    """The app state."""

    prompt: str = ""

    def check_text(self):
        print(classify(self.prompt))
        pass

    pass


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.center(
                rx.vstack(
                    rx.heading("Check AI", font_size="1.5em"),
                    rx.text_area(
                        on_blur=State.set_prompt, placeholder="Text", width="100%"
                    ),
                    rx.center(
                        rx.button("Check", on_click=State.check_text, width="100%")
                    ),
                    shadow="lg",
                    padding="1em",
                    border_radius="lg",
                    width="100%",
                ),
                width="500%",
            ),
        ),
    )


# Add state and page to the app.
app = rx.App()
app.add_page(index)
app.compile()
