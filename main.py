from app import ApplicationEntry


# app_name = "unitree_go1_isaac_cuda_test"
# app_name = "unitree_go1_isaac_cuda_velocity"
app_name = "unitree_go1_mujoco_cpu_velocity"
train_time = None
device = None

with ApplicationEntry(app_name, train_time, device) as app:
    app.train()
    app.save()
    app.play(num_steps=500, formats="gif")
    app.close()    
