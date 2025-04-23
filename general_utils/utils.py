def show_udef_methods(obj, depth=1, max_m=5, offset='', show_type=False):
    """
    Recursively displays user-defined methods of an object.

    This function iterates through the attributes of an object and prints the names
    of those that are considered user-defined methods (i.e., not starting with an
    underscore). It allows for customization of the depth of recursion, the maximum
    number of methods to display, indentation, and whether to show the type of each
    method.

    Args:
        obj: The object whose methods to display.
        depth (int, optional): The maximum depth of recursion. Defaults to 1.
        max_m (int, optional): The maximum number of methods to display at each level.
            Defaults to 5.
        offset (str, optional): The indentation string to use. Defaults to ''.
        show_type (bool, optional): Whether to display the type of each method.
            Defaults to False.
    """
    methods = list(filter(lambda x: not x.startswith('_'), dir(obj)))
    
    for i, m in enumerate(methods):
        if i >= max_m:
            print(offset + '...')
            break

        try:
            new_obj = getattr(obj, m)
            print(f"{offset}{m}" + f" ({type(new_obj).__name__})" if show_type else "")

            if depth > 1:
                new_offset = ''.join([' '] * len(offset))  + '|--'
                show_udef_methods(new_obj, depth - 1, max_m, new_offset, show_type)    
        except AttributeError:
            pass