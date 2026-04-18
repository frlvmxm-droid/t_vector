# -*- coding: utf-8 -*-
"""
Unit tests for exceptions.py — custom exception hierarchy.

Hierarchy under test:
  AppBaseError            (Exception)
  ├─ ModelLoadError       (AppBaseError)
  │  └─ SchemaError       (ModelLoadError)
  ├─ FeatureBuildError    (AppBaseError)
  ├─ PredictPipelineError (AppBaseError)
  └─ UnexpectedError      (AppBaseError)

Covers:
  - isinstance / issubclass relationships at every level
  - Catching a parent class catches all children
  - str() and args representation of each exception type
  - raise / except round-trips for every concrete class
  - AppBaseError catches all custom exceptions
  - All custom exceptions are also plain Exception subclasses
"""
import pytest
from exceptions import (
    AppBaseError,
    ModelLoadError,
    SchemaError,
    FeatureBuildError,
    PredictPipelineError,
    UnexpectedError,
)


# ---------------------------------------------------------------------------
# Inheritance: issubclass checks
# ---------------------------------------------------------------------------

class TestInheritanceIsSubclass:
    def test_app_base_error_is_exception(self):
        assert issubclass(AppBaseError, Exception)

    def test_model_load_error_is_app_base(self):
        assert issubclass(ModelLoadError, AppBaseError)

    def test_schema_error_is_model_load_error(self):
        assert issubclass(SchemaError, ModelLoadError)

    def test_schema_error_is_app_base(self):
        assert issubclass(SchemaError, AppBaseError)

    def test_schema_error_is_exception(self):
        assert issubclass(SchemaError, Exception)

    def test_feature_build_error_is_app_base(self):
        assert issubclass(FeatureBuildError, AppBaseError)

    def test_feature_build_error_is_exception(self):
        assert issubclass(FeatureBuildError, Exception)

    def test_predict_pipeline_error_is_app_base(self):
        assert issubclass(PredictPipelineError, AppBaseError)

    def test_predict_pipeline_error_is_exception(self):
        assert issubclass(PredictPipelineError, Exception)

    def test_unexpected_error_is_app_base(self):
        assert issubclass(UnexpectedError, AppBaseError)

    def test_unexpected_error_is_exception(self):
        assert issubclass(UnexpectedError, Exception)

    # Negative: siblings must NOT be subclasses of each other
    def test_model_load_error_not_feature_build(self):
        assert not issubclass(ModelLoadError, FeatureBuildError)

    def test_feature_build_error_not_model_load(self):
        assert not issubclass(FeatureBuildError, ModelLoadError)

    def test_predict_pipeline_not_model_load(self):
        assert not issubclass(PredictPipelineError, ModelLoadError)


# ---------------------------------------------------------------------------
# Inheritance: isinstance checks on raised instances
# ---------------------------------------------------------------------------

class TestInheritanceIsInstance:
    def test_model_load_error_isinstance_app_base(self):
        exc = ModelLoadError("msg")
        assert isinstance(exc, AppBaseError)
        assert isinstance(exc, Exception)

    def test_schema_error_isinstance_model_load(self):
        exc = SchemaError("schema mismatch")
        assert isinstance(exc, ModelLoadError)
        assert isinstance(exc, AppBaseError)
        assert isinstance(exc, Exception)

    def test_feature_build_error_isinstance_app_base(self):
        exc = FeatureBuildError("build failed")
        assert isinstance(exc, AppBaseError)
        assert isinstance(exc, Exception)

    def test_predict_pipeline_error_isinstance_app_base(self):
        exc = PredictPipelineError("pipeline error")
        assert isinstance(exc, AppBaseError)
        assert isinstance(exc, Exception)

    def test_unexpected_error_isinstance_app_base(self):
        exc = UnexpectedError("oops")
        assert isinstance(exc, AppBaseError)
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# Raise and catch — each exception type individually
# ---------------------------------------------------------------------------

class TestRaiseAndCatch:
    def test_raise_app_base_error(self):
        with pytest.raises(AppBaseError):
            raise AppBaseError("base error")

    def test_raise_model_load_error(self):
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("model not found")

    def test_raise_schema_error(self):
        with pytest.raises(SchemaError):
            raise SchemaError("wrong schema version")

    def test_raise_feature_build_error(self):
        with pytest.raises(FeatureBuildError):
            raise FeatureBuildError("missing column")

    def test_raise_predict_pipeline_error(self):
        with pytest.raises(PredictPipelineError):
            raise PredictPipelineError("dimension mismatch")

    def test_raise_unexpected_error(self):
        with pytest.raises(UnexpectedError):
            raise UnexpectedError("incident X")


# ---------------------------------------------------------------------------
# Catching parent catches child (Liskov / polymorphism)
# ---------------------------------------------------------------------------

class TestParentCatchesChild:
    def test_app_base_catches_model_load_error(self):
        with pytest.raises(AppBaseError):
            raise ModelLoadError("caught as AppBaseError")

    def test_app_base_catches_schema_error(self):
        with pytest.raises(AppBaseError):
            raise SchemaError("caught as AppBaseError")

    def test_app_base_catches_feature_build_error(self):
        with pytest.raises(AppBaseError):
            raise FeatureBuildError("caught as AppBaseError")

    def test_app_base_catches_predict_pipeline_error(self):
        with pytest.raises(AppBaseError):
            raise PredictPipelineError("caught as AppBaseError")

    def test_app_base_catches_unexpected_error(self):
        with pytest.raises(AppBaseError):
            raise UnexpectedError("caught as AppBaseError")

    def test_model_load_error_catches_schema_error(self):
        with pytest.raises(ModelLoadError):
            raise SchemaError("schema caught as ModelLoadError")

    def test_exception_catches_all_custom(self):
        """Built-in Exception catches every custom exception."""
        for exc_class in (
            AppBaseError, ModelLoadError, SchemaError,
            FeatureBuildError, PredictPipelineError, UnexpectedError,
        ):
            with pytest.raises(Exception):
                raise exc_class("caught as Exception")

    def test_except_block_catches_child_as_parent(self):
        """Use a plain try/except to verify child is caught by parent clause."""
        caught = None
        try:
            raise SchemaError("bad version")
        except ModelLoadError as e:
            caught = e
        assert caught is not None
        assert isinstance(caught, SchemaError)


# ---------------------------------------------------------------------------
# String representation and args
# ---------------------------------------------------------------------------

class TestStringRepresentation:
    def test_app_base_error_str(self):
        exc = AppBaseError("something went wrong")
        assert "something went wrong" in str(exc)

    def test_model_load_error_str(self):
        exc = ModelLoadError("file missing")
        assert "file missing" in str(exc)

    def test_schema_error_str(self):
        exc = SchemaError("version 99 not supported")
        assert "version 99 not supported" in str(exc)

    def test_feature_build_error_str(self):
        exc = FeatureBuildError("column 'text' absent")
        assert "column 'text' absent" in str(exc)

    def test_predict_pipeline_error_str(self):
        exc = PredictPipelineError("512 != 768")
        assert "512 != 768" in str(exc)

    def test_unexpected_error_str(self):
        exc = UnexpectedError("incident-42")
        assert "incident-42" in str(exc)

    def test_exception_args_single(self):
        exc = ModelLoadError("reason")
        assert exc.args == ("reason",)

    def test_exception_args_empty(self):
        exc = FeatureBuildError()
        assert exc.args == ()

    def test_exception_args_multiple(self):
        exc = PredictPipelineError("mismatch", 512, 768)
        assert exc.args == ("mismatch", 512, 768)
        assert "mismatch" in str(exc)
