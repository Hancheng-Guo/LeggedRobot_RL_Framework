from app import ApplicationEntry


def main() -> None:
    # app_name = "unitree_go1_isaac_cuda_test"
    # app_name = "unitree_go1_isaac_cuda_velocity"
    # app_name = "unitree_go1_mujoco_cpu_velocity"
    app_name = "unitree_go1_mujoco_cpu_velocity_only"

    with ApplicationEntry(app_name) as app:
        app.train()
        app.save()
        app.play(num_steps=500, formats="gif")
        app.close()


if __name__ == "__main__":
    main()
