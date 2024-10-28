from enum import Enum
from typing import Type, Any, Optional, get_origin, get_args, Union
import typer
from typer.models import OptionInfo
from pydantic import BaseModel
from inspect import Signature, Parameter
from functools import wraps


class CLIManager:
    """
    An extended class-based wrapper around Typer to dynamically generate CLI options from Pydantic models
    with support for enums and custom field types.
    """

    def __init__(self, *config_classes: Type[BaseModel]):
        self.config_classes = config_classes
        self.app = typer.Typer()
        self.params = self._generate_typer_options()
        self.configs: dict[str, BaseModel] = {}

    def _get_field_type(self, field) -> tuple[Any, str, Type]:
        """
        Determine the field type and generate appropriate help text.
        Returns (type, help_text, python_type)
        """
        field_type = field.annotation
        help_text = field.description or ""
        python_type = field_type

        # Handle Optional types
        if get_origin(field_type) is Optional:
            field_type = get_args(field_type)[0]
            python_type = Union[field_type, None]

        # Handle lists
        if get_origin(field_type) is list:
            item_type = get_args(field_type)[0]
            python_type = list[item_type]
            if issubclass(item_type, Enum):
                valid_values = [e.value for e in item_type]
                help_text += f" (valid values: {valid_values})"
            return (list[item_type], help_text, python_type)

        # Handle enums
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            valid_values = [e.value for e in field_type]
            help_text += f" (valid values: {valid_values})"
            return (field_type, help_text, field_type)

        # Handle custom types (like RangeField)
        if hasattr(field_type, "validate"):
            help_text += f" (format: number or start:end)"
            return (field_type, help_text, field_type)

        return (field_type, help_text, python_type)

    def _generate_typer_options(self) -> dict:
        """
        Generates Typer command options for each Pydantic dataclass field.
        Returns a dictionary of parameter name to (OptionInfo, type) tuples.
        """
        params = {}
        for config_class in self.config_classes:
            fields = config_class.model_fields.items()
            for name, field in fields:
                field_type, help_text, python_type = self._get_field_type(field)
                default_value = (
                    field.default if field.default is not None else typer.Option(None)
                )

                # Convert the name to CLI format (using hyphens)
                cli_name = name.replace("_", "-")

                # Base option parameters
                option_params = {
                    "help": help_text,
                    "default": default_value,
                }

                # Special handling for enums
                if isinstance(field_type, type) and issubclass(field_type, Enum):
                    option_params["case_sensitive"] = False

                # Special handling for custom types
                if hasattr(field_type, "validate"):
                    option_params["callback"] = field_type.validate

                # Create the OptionInfo with the appropriate parameters
                option = OptionInfo(**option_params)
                params[name] = (option, python_type)

        return params

    def register_command(self):
        """Register the command with the Typer app."""

        def decorator(f):
            # Generate a list of parameters to add to the function signature dynamically
            parameters = []

            for param_name, (option, param_type) in self.params.items():
                parameters.append(
                    Parameter(
                        param_name,
                        Parameter.POSITIONAL_OR_KEYWORD,
                        default=typer.Option(
                            default=option.default,
                            help=option.help,
                        ),
                        annotation=param_type,
                    )
                )

            # Define the function with these dynamically generated parameters
            @wraps(f)
            def new_func(*args, **kwargs):
                # Process kwargs to create Pydantic model instances
                for config_class in self.config_classes:
                    config_args = {
                        key: value
                        for key, value in kwargs.items()
                        if key in config_class.model_fields
                    }
                    self.configs[config_class.__name__] = config_class(**config_args)

                # Call the original function `f` with `self` as its argument
                return f(self)

            # Set the dynamically generated signature on `new_func`
            new_func.__signature__ = Signature(parameters)

            # Register the function with Typer
            self.app.command()(new_func)  # Register with the Typer app

            return new_func  # This allows decorator chaining

        return decorator

    def run(self):
        """Run the Typer app."""
        self.app()

    def get_config(self, config_name: str) -> Optional[BaseModel]:
        """Get a specific populated config by class name."""
        return self.configs.get(config_name)

    def get_all_configs(self) -> dict[str, BaseModel]:
        """Get all populated config objects."""
        return self.configs
