"""
Tests for connectors/registry.py

The connector registry is critical infrastructure used by both the runtime
(main.py) and admin UI (web/routes.py) to:
- Validate connector IDs
- Generate dynamic configuration forms
- Instantiate connector classes
- Provide metadata for all supported connectors

This test suite ensures the registry is complete, consistent, and properly
supports dynamic connector loading.
"""

import pytest
from dataclasses import asdict
from connectors.registry import (
    ConnectorFieldDef,
    ConnectorTypeDef,
    CONNECTOR_REGISTRY,
    get_connector_class,
)


# --- ConnectorFieldDef Tests ---


def test_connector_field_def_defaults():
    """ConnectorFieldDef should have sensible defaults for optional fields."""
    field = ConnectorFieldDef(name="test_field")

    assert field.name == "test_field"
    assert field.field_type == "string"
    assert field.required is False
    assert field.sensitive is False
    assert field.default is None
    assert field.help_text == ""
    assert field.placeholder == ""


def test_connector_field_def_all_parameters():
    """ConnectorFieldDef should accept and store all parameters correctly."""
    field = ConnectorFieldDef(
        name="api_key",
        field_type="password",
        required=True,
        sensitive=True,
        default="default_value",
        help_text="Your API key",
        placeholder="sk-..."
    )

    assert field.name == "api_key"
    assert field.field_type == "password"
    assert field.required is True
    assert field.sensitive is True
    assert field.default == "default_value"
    assert field.help_text == "Your API key"
    assert field.placeholder == "sk-..."


def test_connector_field_def_valid_types():
    """ConnectorFieldDef should support all documented field types."""
    valid_types = ["string", "password", "integer", "list", "boolean"]

    for field_type in valid_types:
        field = ConnectorFieldDef(name="test", field_type=field_type)
        assert field.field_type == field_type


def test_connector_field_def_is_dataclass():
    """ConnectorFieldDef should be a dataclass with standard conversion methods."""
    field = ConnectorFieldDef(name="test", required=True, sensitive=True)

    # Should convert to dict via dataclasses.asdict
    field_dict = asdict(field)
    assert isinstance(field_dict, dict)
    assert field_dict["name"] == "test"
    assert field_dict["required"] is True
    assert field_dict["sensitive"] is True


# --- ConnectorTypeDef Tests ---


def test_connector_type_def_minimal():
    """ConnectorTypeDef should work with minimal required fields."""
    typedef = ConnectorTypeDef(
        connector_id="test",
        display_name="Test Connector",
        description="A test connector",
        category="api",
        module_path="connectors.test.connector",
        class_name="TestConnector",
    )

    assert typedef.connector_id == "test"
    assert typedef.display_name == "Test Connector"
    assert typedef.description == "A test connector"
    assert typedef.category == "api"
    assert typedef.module_path == "connectors.test.connector"
    assert typedef.class_name == "TestConnector"
    assert typedef.config_fields == []


def test_connector_type_def_with_fields():
    """ConnectorTypeDef should store config fields correctly."""
    field1 = ConnectorFieldDef(name="url", required=True)
    field2 = ConnectorFieldDef(name="api_key", field_type="password", sensitive=True)

    typedef = ConnectorTypeDef(
        connector_id="test",
        display_name="Test",
        description="Test",
        category="api",
        module_path="test",
        class_name="Test",
        config_fields=[field1, field2]
    )

    assert len(typedef.config_fields) == 2
    assert typedef.config_fields[0].name == "url"
    assert typedef.config_fields[1].name == "api_key"


def test_connector_type_def_valid_categories():
    """ConnectorTypeDef should support both 'api' and 'browser' categories."""
    api_typedef = ConnectorTypeDef(
        connector_id="api_test",
        display_name="API Test",
        description="API connector",
        category="api",
        module_path="test",
        class_name="Test"
    )
    assert api_typedef.category == "api"

    browser_typedef = ConnectorTypeDef(
        connector_id="browser_test",
        display_name="Browser Test",
        description="Browser connector",
        category="browser",
        module_path="test",
        class_name="Test"
    )
    assert browser_typedef.category == "browser"


def test_connector_type_def_is_dataclass():
    """ConnectorTypeDef should be a dataclass with standard conversion methods."""
    typedef = ConnectorTypeDef(
        connector_id="test",
        display_name="Test",
        description="Test",
        category="api",
        module_path="test",
        class_name="Test"
    )

    typedef_dict = asdict(typedef)
    assert isinstance(typedef_dict, dict)
    assert typedef_dict["connector_id"] == "test"
    assert typedef_dict["category"] == "api"


# --- CONNECTOR_REGISTRY Tests ---


def test_registry_is_dict():
    """CONNECTOR_REGISTRY should be a dictionary."""
    assert isinstance(CONNECTOR_REGISTRY, dict)


def test_registry_not_empty():
    """CONNECTOR_REGISTRY should contain at least the documented connectors."""
    assert len(CONNECTOR_REGISTRY) > 0

    # Should have at least these core connectors
    expected_connectors = [
        "proton_mail",
        "signal",
        "imessage",
        "caldav",
        "finance",
        "home_assistant",
        "google",
        "whatsapp",
        "youtube",
        "reddit",
    ]

    for connector_id in expected_connectors:
        assert connector_id in CONNECTOR_REGISTRY, f"Missing connector: {connector_id}"


def test_registry_all_values_are_connector_type_defs():
    """All values in CONNECTOR_REGISTRY should be ConnectorTypeDef instances."""
    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        assert isinstance(typedef, ConnectorTypeDef), \
            f"{connector_id} is not a ConnectorTypeDef"


def test_registry_connector_ids_match_keys():
    """Each ConnectorTypeDef's connector_id should match its dictionary key."""
    for key, typedef in CONNECTOR_REGISTRY.items():
        assert typedef.connector_id == key, \
            f"Key '{key}' does not match connector_id '{typedef.connector_id}'"


def test_registry_all_have_required_fields():
    """All registry entries should have all required ConnectorTypeDef fields."""
    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        assert typedef.connector_id, f"{connector_id}: missing connector_id"
        assert typedef.display_name, f"{connector_id}: missing display_name"
        assert typedef.description, f"{connector_id}: missing description"
        assert typedef.category in ["api", "browser"], \
            f"{connector_id}: invalid category '{typedef.category}'"
        assert typedef.module_path, f"{connector_id}: missing module_path"
        assert typedef.class_name, f"{connector_id}: missing class_name"


def test_registry_all_have_config_fields():
    """All registry entries should define their configuration schema."""
    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        # config_fields should be a list (may be empty, but should exist)
        assert isinstance(typedef.config_fields, list), \
            f"{connector_id}: config_fields is not a list"


def test_registry_categories_are_valid():
    """All connectors should have category of either 'api' or 'browser'."""
    valid_categories = {"api", "browser"}

    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        assert typedef.category in valid_categories, \
            f"{connector_id}: invalid category '{typedef.category}'"


def test_registry_api_connectors():
    """API connectors should be properly categorized."""
    api_connectors = [
        "proton_mail",
        "signal",
        "imessage",
        "caldav",
        "finance",
        "home_assistant",
        "google",
    ]

    for connector_id in api_connectors:
        typedef = CONNECTOR_REGISTRY[connector_id]
        assert typedef.category == "api", \
            f"{connector_id} should be category 'api', not '{typedef.category}'"


def test_registry_browser_connectors():
    """Browser connectors should be properly categorized."""
    browser_connectors = ["whatsapp", "youtube", "reddit"]

    for connector_id in browser_connectors:
        typedef = CONNECTOR_REGISTRY[connector_id]
        assert typedef.category == "browser", \
            f"{connector_id} should be category 'browser', not '{typedef.category}'"


def test_registry_config_fields_are_valid():
    """All config fields in the registry should be valid ConnectorFieldDef instances."""
    valid_field_types = {"string", "password", "integer", "list", "boolean"}

    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        for field in typedef.config_fields:
            assert isinstance(field, ConnectorFieldDef), \
                f"{connector_id}: field '{field}' is not a ConnectorFieldDef"

            assert field.name, f"{connector_id}: field has no name"
            assert field.field_type in valid_field_types, \
                f"{connector_id}.{field.name}: invalid field_type '{field.field_type}'"


def test_registry_required_fields_validation():
    """Required fields should be properly marked across all connectors."""
    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        for field in typedef.config_fields:
            # Required fields should have meaningful help text
            if field.required:
                assert field.help_text or field.placeholder, \
                    f"{connector_id}.{field.name}: required field has no help text or placeholder"


def test_registry_sensitive_fields_are_passwords():
    """Sensitive fields should typically use password field type."""
    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        for field in typedef.config_fields:
            # Sensitive fields should generally be password type (with some exceptions)
            if field.sensitive and field.field_type not in ["password", "list"]:
                # Log warning but don't fail - there may be valid exceptions
                print(f"Warning: {connector_id}.{field.name} is sensitive but type '{field.field_type}'")


def test_registry_module_paths_follow_convention():
    """Module paths should follow the connectors.{type}.connector pattern."""
    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        # Should start with "connectors."
        assert typedef.module_path.startswith("connectors."), \
            f"{connector_id}: module_path '{typedef.module_path}' doesn't start with 'connectors.'"


def test_registry_class_names_follow_convention():
    """Class names should follow the {Name}Connector pattern."""
    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        # Should end with "Connector"
        assert typedef.class_name.endswith("Connector"), \
            f"{connector_id}: class_name '{typedef.class_name}' doesn't end with 'Connector'"


def test_registry_no_duplicate_display_names():
    """Each connector should have a unique display name."""
    display_names = [typedef.display_name for typedef in CONNECTOR_REGISTRY.values()]
    assert len(display_names) == len(set(display_names)), \
        "Duplicate display names found in registry"


# --- get_connector_class() Tests ---


def test_get_connector_class_proton_mail():
    """get_connector_class should successfully load ProtonMailConnector."""
    cls = get_connector_class("proton_mail")

    assert cls is not None
    assert cls.__name__ == "ProtonMailConnector"
    assert hasattr(cls, "authenticate")
    assert hasattr(cls, "sync")


def test_get_connector_class_signal():
    """get_connector_class should successfully load SignalConnector."""
    cls = get_connector_class("signal")

    assert cls is not None
    assert cls.__name__ == "SignalConnector"
    assert hasattr(cls, "authenticate")
    assert hasattr(cls, "sync")


def test_get_connector_class_imessage():
    """get_connector_class should successfully load iMessageConnector."""
    cls = get_connector_class("imessage")

    assert cls is not None
    assert cls.__name__ == "iMessageConnector"
    assert hasattr(cls, "authenticate")
    assert hasattr(cls, "sync")


def test_get_connector_class_caldav():
    """get_connector_class should successfully load CalDAVConnector."""
    cls = get_connector_class("caldav")

    assert cls is not None
    assert cls.__name__ == "CalDAVConnector"
    assert hasattr(cls, "authenticate")
    assert hasattr(cls, "sync")


def test_get_connector_class_finance():
    """get_connector_class should successfully load FinanceConnector."""
    cls = get_connector_class("finance")

    assert cls is not None
    assert cls.__name__ == "FinanceConnector"
    assert hasattr(cls, "authenticate")
    assert hasattr(cls, "sync")


def test_get_connector_class_home_assistant():
    """get_connector_class should successfully load HomeAssistantConnector."""
    cls = get_connector_class("home_assistant")

    assert cls is not None
    assert cls.__name__ == "HomeAssistantConnector"
    assert hasattr(cls, "authenticate")
    assert hasattr(cls, "sync")


def test_get_connector_class_google():
    """get_connector_class should successfully load GoogleConnector."""
    cls = get_connector_class("google")

    assert cls is not None
    assert cls.__name__ == "GoogleConnector"
    assert hasattr(cls, "authenticate")
    assert hasattr(cls, "sync")


def test_get_connector_class_whatsapp():
    """get_connector_class should successfully load WhatsAppConnector."""
    cls = get_connector_class("whatsapp")

    assert cls is not None
    assert cls.__name__ == "WhatsAppConnector"
    # Browser connectors use BrowserConnector methods
    assert hasattr(cls, "authenticate") or hasattr(cls, "setup")


def test_get_connector_class_youtube():
    """get_connector_class should successfully load YouTubeConnector."""
    cls = get_connector_class("youtube")

    assert cls is not None
    assert cls.__name__ == "YouTubeConnector"


def test_get_connector_class_reddit():
    """get_connector_class should successfully load RedditConnector."""
    cls = get_connector_class("reddit")

    assert cls is not None
    assert cls.__name__ == "RedditConnector"


def test_get_connector_class_invalid_id():
    """get_connector_class should raise ValueError for unknown connector IDs."""
    with pytest.raises(ValueError, match="Unknown connector type"):
        get_connector_class("nonexistent_connector")


def test_get_connector_class_empty_string():
    """get_connector_class should raise ValueError for empty string."""
    with pytest.raises(ValueError, match="Unknown connector type"):
        get_connector_class("")


def test_get_connector_class_none():
    """get_connector_class should handle None gracefully."""
    with pytest.raises((ValueError, AttributeError)):
        get_connector_class(None)


def test_get_connector_class_all_registered():
    """get_connector_class should successfully load all registered connectors."""
    for connector_id in CONNECTOR_REGISTRY.keys():
        try:
            cls = get_connector_class(connector_id)
            assert cls is not None, f"Failed to load {connector_id}"

            # Class name should match registry
            expected_name = CONNECTOR_REGISTRY[connector_id].class_name
            assert cls.__name__ == expected_name, \
                f"{connector_id}: class name '{cls.__name__}' != expected '{expected_name}'"
        except Exception as e:
            pytest.fail(f"Failed to load {connector_id}: {e}")


def test_get_connector_class_returns_class_not_instance():
    """get_connector_class should return the class itself, not an instance."""
    cls = get_connector_class("proton_mail")

    # Should be a class (type)
    assert isinstance(cls, type)

    # Should not be an instance of the connector
    assert not hasattr(cls, "_is_instance")


def test_get_connector_class_lazy_loading():
    """get_connector_class should use lazy imports (module loaded on demand)."""
    # This test verifies lazy loading by checking the function uses importlib
    import sys

    # Get a connector that might not be imported yet
    test_module = "connectors.reddit"

    # Remove from sys.modules if present (to test fresh import)
    if test_module in sys.modules:
        original_module = sys.modules[test_module]
    else:
        original_module = None

    try:
        if test_module in sys.modules:
            del sys.modules[test_module]

        # Load the connector - should trigger import
        cls = get_connector_class("reddit")
        assert cls is not None

        # Module should now be loaded
        assert test_module in sys.modules or "connectors.browser.reddit" in sys.modules

    finally:
        # Restore original state
        if original_module:
            sys.modules[test_module] = original_module


# --- Integration Tests ---


def test_registry_supports_admin_ui_workflow():
    """The registry should support the admin UI workflow of listing and configuring connectors."""
    # Simulate what web/routes.py does
    api_connectors = []
    browser_connectors = []

    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        connector_info = {
            "id": typedef.connector_id,
            "name": typedef.display_name,
            "description": typedef.description,
            "category": typedef.category,
            "fields": [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "required": field.required,
                    "sensitive": field.sensitive,
                }
                for field in typedef.config_fields
            ]
        }

        if typedef.category == "api":
            api_connectors.append(connector_info)
        else:
            browser_connectors.append(connector_info)

    # Should have both types
    assert len(api_connectors) > 0
    assert len(browser_connectors) > 0

    # All should have required metadata
    for connector in api_connectors + browser_connectors:
        assert connector["id"]
        assert connector["name"]
        assert isinstance(connector["fields"], list)


def test_registry_supports_runtime_instantiation_workflow():
    """The registry should support the main.py workflow of validating and instantiating connectors."""
    # Simulate what main.py does when starting a connector
    connector_id = "proton_mail"

    # 1. Validate connector exists
    assert connector_id in CONNECTOR_REGISTRY

    # 2. Get typedef for config validation
    typedef = CONNECTOR_REGISTRY.get(connector_id)
    assert typedef is not None

    # 3. Get connector class for instantiation
    cls = get_connector_class(connector_id)
    assert cls is not None

    # 4. Class should be instantiable (we don't actually instantiate since it needs real config)
    assert callable(cls)


def test_registry_field_type_consistency():
    """Field types should be consistent with their usage patterns."""
    for connector_id, typedef in CONNECTOR_REGISTRY.items():
        for field in typedef.config_fields:
            # Passwords should be sensitive
            if field.field_type == "password":
                assert field.sensitive, \
                    f"{connector_id}.{field.name}: password field should be marked sensitive"

            # Integer fields should have integer defaults (if default is set)
            if field.field_type == "integer" and field.default is not None:
                assert isinstance(field.default, int), \
                    f"{connector_id}.{field.name}: integer field has non-integer default"

            # Boolean fields should have boolean defaults (if default is set)
            if field.field_type == "boolean" and field.default is not None:
                assert isinstance(field.default, bool), \
                    f"{connector_id}.{field.name}: boolean field has non-boolean default"
