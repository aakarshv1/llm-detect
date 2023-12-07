"""Welcome to Reflex! This file outlines the steps to create a basic app."""
from rxconfig import config

from model.classify import classify
from model.dummy import dummy

import reflex as rx

docs_url = "https://reflex.dev/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"


class State(rx.State):
    """The app state."""

    prompt: str = ""
    percent: int = 0

    def check_text(self):
        percent = classify(self.prompt)
        print(self.prompt, percent)
        self.percent = int(100 * percent)


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
                        rx.button("Check", on_click=State.check_text, width="100%"),
                        width="100%",
                    ),
                    rx.vstack(
                        rx.cond(
                            State.percent > 50,
                            rx.alert(
                                rx.alert_icon(),
                                rx.alert_title("AI text detected"),
                                status="error",
                                width="100%",
                            ),
                            rx.alert(
                                rx.alert_icon(),
                                rx.alert_title("Human text detected"),
                                status="success",
                                width="100%",
                            ),
                        ),
                        rx.hstack(
                            rx.text(f"{State.percent}% confidence", width="14%"),
                            rx.progress(value=State.percent, width="100%"),
                            width="100%",
                        ),
                        width="100%",
                    ),
                    shadow="lg",
                    padding="1em",
                    border_radius="lg",
                    width="100%",
                ),
                width="400%",
            ),
        ),
    )


# Add state and page to the app.
app = rx.App()
app.add_page(index)
app.compile()
