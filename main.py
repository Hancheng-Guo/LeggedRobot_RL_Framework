from app.application_entry import ApplicationEntry


with ApplicationEntry("unitree_go1_velocity") as application:

# with ApplicationEntry(
#     "unitree_go1_velocity",
#     train_time="2026-09-15_23-11-41"
# ) as application:
    
    application.train()
    application.save()
    application.play(num_steps=500, formats="gif")
    print(application)
    input("Press...")
