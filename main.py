import os


if os.name == "nt":
    os.environ.setdefault("MUJOCO_GL", "wgl")

from app.application_entry import ApplicationEntry


with ApplicationEntry("unitree_go1_velocity") as application:
    application.train()
    application.save()
    application.play(num_steps=500, formats="gif")
    print(application)
    input("Press...")

# with ApplicationEntry(
#     "unitree_go1_velocity",
#     train_time="2026-09-15_14-30-00",
# ) as application:
#     application.play(num_steps=500, formats="gif")
