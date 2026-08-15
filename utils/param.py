
def update_attributes(instance, **kwargs) -> None:

    for attribute_name, attribute_value in kwargs.items():

        if attribute_value is not None:
            setattr(instance,attribute_name,attribute_value)

        elif getattr(instance, attribute_name, None) is None:
            raise ValueError(
                f"instance {instance!r} is "
                f"missing attribute {attribute_name!r}."
            )
