
def memory_accountant(
    param_shapes,
    param_bytes_per_element,
    grad_bytes_per_element,
    activation_shapes,
    activation_bytes_per_element,
    optimizer,
    optimizer_bytes_per_element
):
    # Calculate total number of elements
    def count_elements(shapes):
        total = 0

        for shape in shapes:
            elements = 1

            for dim in shape:
                elements *= dim

            total += elements

        return total

    # Count parameter and activation elements
    param_count = count_elements(param_shapes)
    activation_count = count_elements(activation_shapes)

    # Calculate memory for each category
    parameters = param_count * param_bytes_per_element

    gradients = param_count * grad_bytes_per_element

    activations = (
        activation_count * activation_bytes_per_element
    )

    # Optimizer state tensors
    state_count = {
        "sgd": 0,
        "adagrad": 1,
        "adam": 2
    }

    optimizer_state = (
        param_count
        * state_count[optimizer]
        * optimizer_bytes_per_element
    )

    total = (
        parameters
        + gradients
        + activations
        + optimizer_state
    )

    return {
        "parameters": parameters,
        "gradients": gradients,
        "activations": activations,
        "optimizer_state": optimizer_state,
        "total": total
    }
