from enum import Enum
from typing import Type, Any, Optional, get_origin, get_args
import typer
from typer import Option
from pydantic import BaseModel


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

    def _get_field_type(self, field) -> tuple[Any, str]:
        """
        Determine the field type and generate appropriate help text.
        Returns (type, help_text)
        """
        field_type = field.annotation
        help_text = field.description or ""

        # Handle Optional types
        if get_origin(field_type) is Optional:
            field_type = get_args(field_type)[0]

        # Handle lists
        if get_origin(field_type) is list:
            item_type = get_args(field_type)[0]
            if issubclass(item_type, Enum):
                valid_values = [e.value for e in item_type]
                help_text += f" (valid values: {valid_values})"
            return (list[item_type], help_text)

        # Handle enums
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            valid_values = [e.value for e in field_type]
            help_text += f" (valid values: {valid_values})"
            return (field_type, help_text)

        # Handle custom types (like RangeField)
        if hasattr(field_type, "validate"):
            help_text += f" (format: number or start:end)"
            return (field_type, help_text)

        return (field_type, help_text)

    def _generate_typer_options(self) -> list:
        """
        Generates Typer command options for each Pydantic dataclass field.
        Now handles enums and custom field types.
        """
        params = []
        for config_class in self.config_classes:
            fields = config_class.model_fields.items()
            for name, field in fields:
                field_type, help_text = self._get_field_type(field)
                default_value = field.default if field.default is not None else ...

                # Special handling for enums
                if isinstance(field_type, type) and issubclass(field_type, Enum):
                    param = Option(
                        default_value,
                        help=help_text,
                        case_sensitive=False,
                    )
                # Special handling for custom types
                elif hasattr(field_type, "validate"):
                    param = Option(
                        default_value, help=help_text, callback=field_type.validate
                    )
                # Default handling
                else:
                    param = Option(default_value, help=help_text)

                params.append((name, param))
        return params

    def command(self, **kwargs: Any) -> None:
        """
        Main command to parse CLI arguments and instantiate the Pydantic models.
        """
        for config_class in self.config_classes:
            config_args = {
                key: value
                for key, value in kwargs.items()
                if key in config_class.model_fields
            }
            self.configs[config_class.__name__] = config_class(**config_args)

    def register_command(self):
        """Register the command with the Typer app."""
        self.app.command()(
            typer.main.create_command_from_function(self.command, self.params)
        )

    def run(self):
        """Run the Typer app."""
        self.app()

    def get_config(self, config_name: str) -> Optional[BaseModel]:
        """Get a specific populated config by class name."""
        return self.configs.get(config_name)

    def get_all_configs(self) -> dict[str, BaseModel]:
        """Get all populated config objects."""
        return self.configs
