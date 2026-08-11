
def update_attributes(instance, **kwargs) -> None:

    for attribute_name, attribute_value in kwargs.items():

        if not hasattr(instance, attribute_name):

            raise ValueError(
                f"instance {instance!r} is "
                f"missing attribute {attribute_name!r}."
            )

        if attribute_value is not None:
            setattr(instance,attribute_name,attribute_value)

        elif getattr(instance, attribute_name) is None:
            raise ValueError(
                f"attribute {attribute_name!r} of "
                f"instance {instance!r} is missing value."
            )
            
