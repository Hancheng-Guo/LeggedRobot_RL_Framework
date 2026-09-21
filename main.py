from app.application_entry import ApplicationEntry


# with ApplicationEntry("unitree_go1_isaac_sim_test") as application:

with ApplicationEntry("unitree_go1_isaac_sim") as application:

# with ApplicationEntry(
#     "unitree_go1_isaac_sim",
#     train_time="2026-09-15_23-11-41",
# ) as application:
    
    application.train()
    application.save()
    application.play(num_steps=500, formats="gif")
    input("Press...")
